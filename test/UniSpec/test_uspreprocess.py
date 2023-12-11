from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
import requests
from pathlib import Path


# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    
    # must add dummy second dimension
    INPUT = np.array(open("test/UniSpec/labels_input.txt").read().split("\n")).astype('object')[:,None]
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    in_INPUT = grpcclient.InferInput('labels', INPUT.shape, 'BYTES')
    in_INPUT.set_data_from_numpy(INPUT)
    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_INPUT],
        outputs=[
            grpcclient.InferRequestedOutput("intensities"),
        ],
    )

    intensities = result.as_numpy("intensities")

    assert intensities.shape == (50, 7919)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/UniSpec/test_output_tensor.npy"),
        rtol=0,
        atol=1e-5,
    )
