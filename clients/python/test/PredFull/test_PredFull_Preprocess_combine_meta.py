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

    charge = np.array(
        [
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
        ],
        dtype=np.float32,
    )

    fragmentation = np.array(
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
            ]
        ],
        dtype=np.float32,
    )

    mass = np.array([[0.042774]], dtype=np.float32)
    nce = np.array([[0.30]], dtype=np.float32)

    in_charge = grpcclient.InferInput("precursor_charges_in:0", [1, 30], "FP32")
    in_charge.set_data_from_numpy(charge)

    in_frag = grpcclient.InferInput("fragmentation_types_encoding", [1, 30], "FP32")
    in_frag.set_data_from_numpy(fragmentation)

    in_mass = grpcclient.InferInput("precursor_mass", [1, 1], "FP32")
    in_mass.set_data_from_numpy(mass)

    in_nce = grpcclient.InferInput("norm_collision_energy", [1, 1], "FP32")
    in_nce.set_data_from_numpy(nce)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_charge, in_frag, in_mass, in_nce],
        outputs=[
            grpcclient.InferRequestedOutput("meta_input"),
        ],
    )

    meta = result.as_numpy("meta_input")
    print(meta.shape)
    assert meta.shape == (1, 3, 30)
