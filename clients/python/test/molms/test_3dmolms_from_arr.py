#from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from pathlib import Path
import requests

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_arr(): 
    arr_dir = Path(__file__).parent
    x_data = np.load(os.path.join(arr_dir, 'arr_3dmolms_x.npy'))  # Load the x_data array from the file
    env_data = np.load(os.path.join(arr_dir, 'arr_3dmolms_env.npy'))  # Load the env_data array from the file
    idx_base_data = np.load(os.path.join(arr_dir, 'arr_3dmolms_idx_base.npy'))  # Load the idx_base_data array from the file
    y_data = np.load(os.path.join(arr_dir, 'arr_3dmolms_y.npy'))  # Load the y_data array from the file
    
    # Add a new dimension at the 0-axis (batch dimension)
    x_data = np.expand_dims(x_data, axis=0)
    env_data = np.expand_dims(env_data, axis=0)
    idx_base_data = np.expand_dims(np.expand_dims(idx_base_data, axis=0), axis=0).astype(np.int32)
    y_data = np.expand_dims(y_data, axis=0)
    return x_data, env_data, idx_base_data, y_data

def test_interference_realmodel():
    # bareserver = "localhost:8501"
    # SERVER_HTTP = "http://localhost:8501"
    bareserver = "ucr-lemon.duckdns.org:8501"
    SERVER_HTTP = "http://ucr-lemon.duckdns.org:8501"
    MODEL_NAME = "3dmolms_torch"

    url = f"{SERVER_HTTP}/v2/models/{MODEL_NAME}/infer"

    triton_client = httpclient.InferenceServerClient(url=bareserver)

    # Check if the server is live
    if not triton_client.is_server_live():
        print("Triton server is not live!")
        exit(1)

    # Prepare input data
    x_data, env_data, idx_base_data, y_data = load_arr()

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

    # Calculate cosine similarity
    output_data_flattened = output_data.reshape(-1, output_data.shape[-1])
    y_data_flattened = y_data.reshape(-1, y_data.shape[-1])
    cosine_sim = cosine_similarity(output_data_flattened, y_data_flattened)
    print("Cosine Similarity:", cosine_sim.item())



def main(): 
    test_interference_realmodel()

if __name__ == "__main__":
    main()