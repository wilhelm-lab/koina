#from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
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

    bareserver = "localhost:8501"
    SERVER_HTTP = "http://localhost:8501"
    MODEL_NAME = "molecularnetworking2"

    url = f"{SERVER_HTTP}/v2/models/{MODEL_NAME}/infer"

    triton_client = httpclient.InferenceServerClient(url=bareserver)

    # Check if the server is live
    if not triton_client.is_server_live():
        print("Triton server is not live!")
        exit(1)

    input_data = np.array([[20]], dtype=np.int32)  # Replace with appropriate input
    input = httpclient.InferInput("charge_raw", input_data.shape, "INT32")
    input.set_data_from_numpy(input_data)

    output = httpclient.InferRequestedOutput("charge_norm")

    response = triton_client.infer(MODEL_NAME, inputs=[input], outputs=[output])

    output_data = response.as_numpy("charge_norm")

    print(output_data)


def main():
    test_inference()
    #test_available_grpc()

if __name__ == "__main__":
    main()