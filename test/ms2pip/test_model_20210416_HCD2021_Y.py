from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
from pathlib import Path
import requests
import numpy as np

# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    input = np.load("test/ms2pip/ms2pip_xgboost_input.npy").astype(np.float32)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("input__0", input.shape, "FP32")
    in_pep_seq.set_data_from_numpy(input)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq],
        outputs=[
            grpcclient.InferRequestedOutput("output__0"),
        ],
    )

    intensities = result.as_numpy("output__0")

    assert intensities.shape == (116,)

    assert np.allclose(
        intensities,
        np.load("test/ms2pip/predictions_y_ions.npy"),
        rtol=0,
        atol=1e-4,
    )
