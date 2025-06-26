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
    seq = np.load("test/Altimeter/arr_Altimeter_2024_seq.npy")
    charge = np.load("test/Altimeter/arr_Altimeter_2024_charge.npy")

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("inp", seq.shape, "FP32")
    in_pep_seq.set_data_from_numpy(seq)

    in_charge = grpcclient.InferInput("inpch", charge.shape, "FP32")
    in_charge.set_data_from_numpy(charge)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge],
        outputs=[
            grpcclient.InferRequestedOutput("coefficients"),
            grpcclient.InferRequestedOutput("knots"),
            grpcclient.InferRequestedOutput("AUCs"),
        ],
    )

    coefficients = result.as_numpy("coefficients")
    knots = result.as_numpy("knots")
    AUCs = result.as_numpy("AUCs")

    assert coefficients.shape == (4, 4, 380)
    assert knots.shape == (4, 8)
    assert AUCs.shape == (4, 380)

    assert np.allclose(
        coefficients,
        np.load("test/Altimeter/arr_Altimeter_2024_coefs.npy"),
        rtol=0,
        atol=1e-5,
    )
    
    assert np.allclose(
        knots,
        np.load("test/Altimeter/arr_Altimeter_2024_knots.npy"),
        rtol=0,
        atol=1e-5,
    )
    
    assert np.allclose(
        AUCs,
        np.load("test/Altimeter/arr_Altimeter_2024_AUCs.npy"),
        rtol=0,
        atol=1e-5,
    )
