from test.server_config import SERVER_GRPC, SERVER_HTTP
from pathlib import Path
from test.lib import lib_test_available_grpc, lib_test_available_http
import numpy as np
import tritonclient.grpc as grpcclient

# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    lib_test_available_http(MODEL_NAME, SERVER_HTTP)


def test_available_grpc():
    lib_test_available_grpc(MODEL_NAME, SERVER_GRPC)


def test_inference():

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    ces = np.array([[30]], dtype=np.float32)
    in_ces = grpcclient.InferInput("collision_energies", [1, 1], "FP32")
    in_ces.set_data_from_numpy(ces)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_ces],
        outputs=[
            grpcclient.InferRequestedOutput("norm_collision_energy"),
        ],
    )

    normce = result.as_numpy("norm_collision_energy")

    ground_truth = np.array([0.30])

    assert np.allclose(normce[0], ground_truth, atol=1e-4)
