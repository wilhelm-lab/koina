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


def find_and_download():
    for path_zen in glob(f"repo/**/.zenodo", recursive=True):
        path_zen = Path(path_zen)
        with open(path_zen) as f:
            url_zip = f.readline()
            checksum_algorithm, checksum = f.readline().strip().split(":")
        path_zip = Path(f"{path_zen.parent}/zenodo.zip")
        if not path_zip.is_file() or md5sum(path_zip) != checksum:
            print(
                f"MD5 mismatch or file not found. Downloading {url_zip} to {path_zip}"
            )
            with open(path_zip, "wb") as f:
                f.write(requests.get(url_zip).content)
            with zipfile.ZipFile(path_zip, "r") as f:
                f.extractall(path_zen.parent)
        else:
            print(f"Skipping. {path_zip} checksum matches {checksum}")


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
