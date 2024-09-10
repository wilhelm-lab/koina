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
    MODEL_NAME = "3dmolms_ensemble"

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)

def test_inference():
    structures = [
        "CCCCCCC"
    ]

    #bareserver = "localhost:8501" # Local test
    bareserver = "lemon.wanglab.science:8501" # production test
    MODEL_NAME = "3dmolms_ensemble"

    triton_client = httpclient.InferenceServerClient(url=bareserver)

    # Check if the server is live
    if not triton_client.is_server_live():
        print("Triton server is not live!")
        exit(1)

    smiles_in = 'C/C(=C\\CNc1nc[nH]c2ncnc1-2)CO'
    smiles_in = np.array([smiles_in], dtype=object).reshape(-1, 1)
    precursor_type_in = '[M+H]+'
    precursor_type_in = np.array([precursor_type_in], dtype=object).reshape(-1, 1)
    collision_energy_in = np.array([[20.0]], dtype=np.float32)  

    SMILES = httpclient.InferInput("SMILES", smiles_in.shape, "BYTES")
    SMILES.set_data_from_numpy(smiles_in)

    precursor_type = httpclient.InferInput("precursor_type", precursor_type_in.shape, "BYTES")
    precursor_type.set_data_from_numpy(precursor_type_in)

    collision_energy = httpclient.InferInput("collision_energy", collision_energy_in.shape, "FP32")
    collision_energy.set_data_from_numpy(collision_energy_in)

    output = httpclient.InferRequestedOutput("me_out")

    response = triton_client.infer(MODEL_NAME, inputs=[SMILES, precursor_type, collision_energy], outputs=[output])

    output_data = response.as_numpy("me_out")

    # This should be around 16, as it should be 2 x 2 x 4 (though the torch model is not exact in 4x)

    print(len(output_data))
    print(output_data)


def main():
    #test_inference_torch()
    test_inference()
    #test_available_grpc()

if __name__ == "__main__":
    main()