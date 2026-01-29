#!/usr/bin/python3 -u
# MAX_SESSION_SHARE_COUNT=100
import os
from glob import glob
import re
from pathlib import Path, PosixPath
import time
from typing import List, Set
from itertools import chain
import hashlib

import requests
import zipfile
import subprocess


def find_model_dependency_loc(loc: PosixPath) -> Set[PosixPath]:
    dependencies = set()
    with open(f"{loc}/config.pbtxt") as f:
        dependencies.update(re.findall("model_name:.[\"|'](.*)[\"|']", f.read()))

    py_files = glob(f"{loc}/**/*py")
    for pyf in py_files:
        with open(pyf) as f:
            dependencies.update(re.findall('model_name="(.*)"', f.read()))

    return set(chain(*[find_model_paths(d) for d in dependencies]))


def find_model_paths(pattern: str) -> List[PosixPath]:
    if type(pattern) is str:
        return set([Path(x) for x in glob(f"**/{pattern}") if x[:4] != "repo"])
    elif type(pattern) is PosixPath:
        return set([pattern])
    else:
        raise ValueError(
            "Can't handle pattern: {pattern}. It's neither a string nor a path."
        )


def clean_repo(auto=False):
    tobedeleted_files = glob(f"repo/*")
    x = "n"
    if not auto:
        print("The following files will be deleted")
        print(tobedeleted_files)
        time.sleep(1)
        x = input("Enter [y] to confirm")
    if x == "y" or auto:
        [os.remove(x) for x in tobedeleted_files]
    else:
        print("Aborting, no files were harmed in the making of this message")


def symlink_model(loc: PosixPath):
    try:
        os.symlink(f"../{loc}", f"repo/{loc.name}")
        recursive_dependency_symlink(loc)
    except FileExistsError:
        pass


def recursive_dependency_symlink(pattern: str):
    dependencies = find_model_paths(pattern)
    dependencies.update(
        set(chain(*[list(find_model_dependency_loc(d)) for d in dependencies]))
    )
    for d in dependencies:
        symlink_model(d)


def md5sum(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url_zip, target_zip):
    path_zip = Path(target_zip)
    timeout = 5 * 30 
    start_time = time.time()
    with requests.get(url_zip, stream=True) as r:
        r.raise_for_status()
        with open(path_zip, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if time.time() - start_time > timeout:
                    raise TimeoutError("Download timed out")
                if chunk:
                    f.write(chunk)
    with zipfile.ZipFile(path_zip, "r") as f:
        f.extractall(path_zip.parent)


def find_and_download():
    for path_zen in glob(f"repo/**/.zenodo", recursive=True):
        path_zen = Path(path_zen)
        target_zip = Path(f"{path_zen.parent}/zenodo.zip")
        with open(path_zen) as f:
            url_zip = f.readline()
            content = [line.strip() for line in f.readlines()]
            for line in content:
                if line[:3] == "md5":
                    checksum_algorithm, checksum = line.split(":")
                else:
                    url_zip = line

                if target_zip.is_file() and md5sum(target_zip) == checksum:
                    print(f"Skipping download for {target_zip}, checksum matches")
                    break

                elif not target_zip.is_file():
                    print(f"Downloading model from {url_zip}")
                    try:
                        download_file(url_zip, target_zip)
                        if md5sum(target_zip) != checksum:
                            print(f"Checksum mismatch for {target_zip}, deleting file")
                            os.remove(target_zip)
                    except Exception as e:
                        print(f"Download failed: {e}")
                        if target_zip.is_file():
                            os.remove(target_zip)


if __name__ == "__main__":
    os.chdir("/models")
    clean_repo(True)
    try:
        recursive_dependency_symlink(os.environ["MODEL_PATTERN"])
    except KeyError:
        print("MODEL_PATTERN key not found linking all available models")
        recursive_dependency_symlink("*")
    find_and_download()

    triton_cmd = [
        "tritonserver",
        "--model-repository=/models/repo",
        "--allow-grpc=true",
        "--grpc-port=8500",
        "--allow-http=true",
        "--allow-metrics=true",
        "--allow-cpu-metrics=true",
        "--allow-gpu-metrics=true",
        "--metrics-port=8502",
        "--log-info=true",
        "--log-warning=true",
        "--log-error=true",
        "--rate-limit",
        "execution_count",
        "--cuda-memory-pool-byte-size",
        "0:536870912",
        "--grpc-infer-response-compression-level",
        "high",
    ]

    usi_proxy = subprocess.Popen(["/models/usi_proxy"])

    time.sleep(1)  # Allow for startup time of the proxy
    if usi_proxy.poll() is None:
        # .poll() returns None if the process is still running
        # that means the usi proxy likely started successfully
        triton = subprocess.Popen(triton_cmd + ["--http-port=8503"])
        while True:
            time.sleep(15)
            if triton.poll() is not None:
                print("Triton exited, killing USI Proxy")
                usi_proxy.kill()
                os._exit(1)
            elif usi_proxy.poll() is not None:
                print("USI Proxy died, restarting")
                usi_proxy = subprocess.Popen(["/models/usi_proxy"])
    else:
        # USI proxy didn't start. Start triton without it
        subprocess.run(triton_cmd + ["--http-port=8501"], check=True)
