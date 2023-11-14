#!/usr/bin/env python3
# MAX_SESSION_SHARE_COUNT=100
import os
from glob import glob
import re
from pathlib import Path, PosixPath
import time
from typing import List, Set
from itertools import chain

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
        return set([Path(x) for x in glob(f"[!repo]**/{pattern}")])
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


def find_and_download():
    for path_zen in glob(f"repo/**/.zenodo", recursive=True):
        path_zen = Path(path_zen)
        print(f"Downloading {path_zen}")
        with open(path_zen) as f:
            url_zip = f.read()
        path_zip = Path(f"{path_zen.parent}/tmp.zip")
        if not path_zip.is_file():
            with open(path_zip, "wb") as f:
                f.write(requests.get(url_zip).content)
            with zipfile.ZipFile(path_zip, "r") as f:
                f.extractall(path_zen.parent)
        else:
            print(f"Skipping. {path_zip} exists")


if __name__ == "__main__":
    os.chdir("/models")
    clean_repo(True)
    recursive_dependency_symlink(os.environ["MODEL_PATTERN"])
    find_and_download()

    subprocess.run(
        [
            "tritonserver",
            "--model-repository=/models/repo",
            "--allow-grpc=true",
            "--grpc-port=8500",
            "--allow-http=true",
            "--http-port=8501",
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
        ],
        check=False,
    )
