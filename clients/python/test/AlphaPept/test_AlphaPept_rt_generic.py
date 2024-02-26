from test.server_config import SERVER_GRPC, SERVER_HTTP
from koinapy import Koina
import numpy as np
import requests
from pathlib import Path
from glob import glob


# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    client = Koina(MODEL_NAME, server_url=SERVER_GRPC, ssl=False)
    assert client._is_model_ready() is None


def test_inference():
    files = glob(f"**/arr-{MODEL_NAME}-*", recursive=True)
    data = {Path(f).stem.split("-")[-1]:np.load(f) for f in files}

    client = Koina(MODEL_NAME, server_url=SERVER_GRPC, ssl=False)
    preds = client.predict(data)
    
    for k in preds.keys():
        assert np.allclose(
            preds[k],
            data[k],
            rtol=0,
            atol=1e-4,
        )
