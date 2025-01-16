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
    coefficients = np.load("test/Prosit/arr_Altimeter_2024_eval_splines_coef.npy")
    knots = np.load("test/Prosit/arr_Altimeter_2024_eval_splines_knots.npy")
    ces = np.load("test/Prosit/arr_Altimeter_2024_eval_splines_ces.npy")

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_coef = grpcclient.InferInput("coefficients", coefficients, "FP32")
    in_coef.set_data_from_numpy(coefficients)

    in_knots = grpcclient.InferInput("knots", knots.shape, "FP32")
    in_knots.set_data_from_numpy(knots)
    
    in_ces = grpcclient.InferInput("inpce", ces.shape, "INT32")
    in_ces.set_data_from_numpy(ces)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_coef, in_knots, in_ces],
        outputs=[
            grpcclient.InferRequestedOutput("intensities")
        ],
    )

    intensities = result.as_numpy("intensities")

    assert intensities.shape == (5, 200)

    assert np.allclose(
        intensities,
        np.load("test/Prosit/arr_Altimter_2024_eval_splines_int_raw.npy"),
        rtol=0,
        atol=1e-5,
    )

