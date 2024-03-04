from koinapy import Koina
import numpy as np
import requests
from pathlib import Path
from glob import glob


def lib_test_available_http(model_name, server_http):
    req = requests.get(f"{server_http}/v2/models/{model_name}", timeout=1)
    assert req.status_code == 200


def lib_test_available_grpc(model_name, server_grpc):
    client = Koina(model_name, server_url=server_grpc, ssl=False)
    assert client._is_model_ready() is None


def lib_test_inference(model_name, server_grpc, atol=1e-6):
    files = glob(f"**/arr-{model_name}-*.npy", recursive=True)
    data = {Path(f).stem.split("-")[-1]: np.load(f) for f in files}

    client = Koina(model_name, server_url=server_grpc, ssl=False)
    preds = client.predict(data)

    for k in preds.keys():
        try:
            assert np.allclose(
                preds[k],
                data[k],
                rtol=0,
                atol=atol,
            )
        except TypeError:
            assert np.all(preds[k] == preds[k])
        except AssertionError as e:
            print(k)
            print(preds[k])
            print(data[k])
            raise e
