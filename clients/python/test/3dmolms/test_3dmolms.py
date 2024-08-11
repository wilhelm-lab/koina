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
    MODEL_NAME = "3dmolms"

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


def test_interference_realmodel():
    bareserver = "localhost:8501"
    #SERVER_HTTP = "http://localhost:8501"
    SERVER_HTTP = "http://ucr-lemon.duckdns.org:8501"
    MODEL_NAME = "3dmolms_torch"

    url = f"{SERVER_HTTP}/v2/models/{MODEL_NAME}/infer"

    triton_client = httpclient.InferenceServerClient(url=bareserver)

    # Check if the server is live
    if not triton_client.is_server_live():
        print("Triton server is not live!")
        exit(1)

    # Prepare input data
    x_data = np.random.rand(1, 21, 300).astype(np.float32)  # Example input for 'x'
    env_data = np.random.rand(1, 6).astype(np.float32)  # Example input for 'env'
    idx_base_data = np.array([[[[0]]]], dtype=np.int32)  # Example input for 'idx_base' with dynamic dimensions

    # Create Triton inputs
    x_input = httpclient.InferInput("x", x_data.shape, "FP32")
    x_input.set_data_from_numpy(x_data)

    env_input = httpclient.InferInput("env", env_data.shape, "FP32")
    env_input.set_data_from_numpy(env_data)

    idx_base_input = httpclient.InferInput("idx_base", idx_base_data.shape, "INT32")
    idx_base_input.set_data_from_numpy(idx_base_data)

    # Define the output
    output = httpclient.InferRequestedOutput("3dmolms_out")

    # Perform inference
    response = triton_client.infer(MODEL_NAME, inputs=[x_input, env_input, idx_base_input], outputs=[output])

    # Get the output data
    output_data = response.as_numpy("3dmolms_out")

    print(output_data)
    print(output_data.shape)




def main():
    #test_inference()
    test_interference_realmodel()
    #test_available_grpc()

if __name__ == "__main__":
    main()