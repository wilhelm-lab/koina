from server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
import requests

MODEL_NAME = "fragment_mz"


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    out_layer = "fragment_mz"
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("ProForma", [5, 1], "BYTES")
    in_pep_seq.set_data_from_numpy(
        np.array([["AAAAAA"] for _ in range(0, 5)], dtype=np.object_)
    )

    in_charge = grpcclient.InferInput("charges", [4], "INT32")
    in_charge.set_data_from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))

    in_ion_series = grpcclient.InferInput("ion_series", [3], "BYTES")
    in_ion_series.set_data_from_numpy(np.array(["b", "y", "a"], dtype=np.object_))

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge, in_ion_series],
        outputs=[grpcclient.InferRequestedOutput(out_layer)],
    )

    assert result.as_numpy(out_layer).shape == (5, 3, 4, 32)
