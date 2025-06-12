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

    fragmentation = np.array([["HCD"], ["CID"]], dtype=np.object_)
    in_fragmentation = grpcclient.InferInput(
        "fragmentation_types", fragmentation.shape, "BYTES"
    )
    in_fragmentation.set_data_from_numpy(fragmentation)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_fragmentation],
        outputs=[
            grpcclient.InferRequestedOutput("fragmentation_types_encoding"),
        ],
    )

    one_fragmentation = result.as_numpy("fragmentation_types_encoding")

    ground_truth = np.array(
        [
            [
                0.0,
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
            ],
            [
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
                0.0,
            ],
        ]
    )

    print(one_fragmentation)
    print(ground_truth)

    assert np.array_equal(one_fragmentation, ground_truth)
