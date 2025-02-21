from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
from pathlib import Path
import requests

# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    coefficients = np.load("test/Altimeter/arr_Altimeter_2024_filtered_coefs.npy")
    knots = np.load("test/Altimeter/arr_Altimeter_2024_filtered_knots.npy")
    ces = np.load("test/Altimeter/arr_Altimeter_2024_NCEs.npy")

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_coef = grpcclient.InferInput("coefficients", coefficients.shape, "FP32")
    in_coef.set_data_from_numpy(coefficients)

    in_knots = grpcclient.InferInput("knots", knots.shape, "FP32")
    in_knots.set_data_from_numpy(knots)
    
    in_ces = grpcclient.InferInput("inpce", ces.shape, "FP32")
    in_ces.set_data_from_numpy(ces)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_coef, in_knots, in_ces],
        outputs=[
            grpcclient.InferRequestedOutput("intensities")
        ],
    )

    intensities = result.as_numpy("intensities")

    assert intensities.shape == (4, 380)

    assert np.allclose(
        intensities,
        np.load("test/Altimeter/arr_Altimeter_2024_int.npy"),
        rtol=0,
        atol=1e-5,
    )

