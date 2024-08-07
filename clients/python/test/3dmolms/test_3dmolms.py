#from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
from pathlib import Path
import requests
import numpy as np

# # To ensure MODEL_NAME == test_<filename>.py
# MODEL_NAME = Path(__file__).stem.replace("test_", "")


# def test_available_http():
#     req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
#     assert req.status_code == 200


def test_available_grpc():
    SERVER_GRPC = "localhost:8502"

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)

def test_inference():
    structures = [
        "CCCCCCC"
    ]

    SERVER_HTTP = "http://localhost:8502"
    MODEL_NAME = "3dmolms"

    url = f"{SERVER_HTTP}/v2/models/{MODEL_NAME}/infer"
    
    params = {
        "id": "1",
        "inputs": [
            {
            "name": "charge_raw",
            "shape": [
                -1,
                1
            ],
            "datatype": "FP32",
            "data": [
                [1.0]
                ]
            }
        ]
    }

    r = requests.post(url, data=params)

    print(r.json())
    

def main():
    test_inference()

if __name__ == "__main__":
    main()