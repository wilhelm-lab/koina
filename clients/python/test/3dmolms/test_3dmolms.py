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
    SERVER_GRPC = "localhost:8500"
    MODEL_NAME = "3dmolms"

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)

def test_inference():
    structures = [
        "CCCCCCC"
    ]

    SERVER_HTTP = "http://localhost:8501"
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
            "datatype": "INT32",
            "data": [
                [1]
                ]
            }
        ]
    }

    r = requests.post(url, data=params)

    print(r.text)
    

def main():
    test_inference()
    #test_available_grpc()

if __name__ == "__main__":
    main()