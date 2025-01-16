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

    charge = np.array([[3], [1]], dtype=np.int32)
    in_charge = grpcclient.InferInput("precursor_charges", [len(charge), 1], "INT32")
    in_charge.set_data_from_numpy(charge)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_charge],
        outputs=[
            grpcclient.InferRequestedOutput("precursor_charges_in:0"),
        ],
    )

    one_hot_charge = result.as_numpy("precursor_charges_in:0")

    ground_truth = np.array(
        [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    print(one_hot_charge[1])
    assert np.array_equal(one_hot_charge[0], ground_truth)
