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
    x = np.load("test/Deeplc/arr_x.npy")
    x_sum = np.load("test/Deeplc/arr_x_sum.npy")
    x_global = np.load("test/Deeplc/arr_x_global.npy")
    x_hc = np.load("test/Deeplc/arr_x_hc.npy")

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    inputs = [
        grpcclient.InferInput("input_141", x.shape, "FP32"),
        grpcclient.InferInput("input_142", x_sum.shape, "FP32"),
        grpcclient.InferInput("input_143", x_global.shape, "FP32"),
        grpcclient.InferInput("input_144", x_hc.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(x)
    inputs[1].set_data_from_numpy(x_sum)
    inputs[2].set_data_from_numpy(x_global)
    inputs[3].set_data_from_numpy(x_hc)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=inputs,
        outputs=[
            grpcclient.InferRequestedOutput("dense_323"),
        ],
    )

    preds = result.as_numpy("dense_323")

    assert preds.shape == (4, 1)

    assert np.allclose(
        preds,
        np.load("test/Deeplc/arr_preds.npy"),
        rtol=0,
        atol=1e-5,
    )
