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
    seq = np.load("test/Prosit/arr_Prosit_2025_intensity_ptm1_seq.npy").reshape(
        [-1, 32]
    )

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("input_1", seq.shape, "INT64")
    in_pep_seq.set_data_from_numpy(seq.astype(np.int64))

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq],
        outputs=[
            grpcclient.InferRequestedOutput("output_1"),
        ],
    )

    irt = result.as_numpy("output_1")

    assert irt.shape == (5, 1)
    print(irt)
    assert np.allclose(
        irt,
        np.load("test/Prosit/arr_Prosit_2025_irt_ptm1_irt_raw.npy"),
        rtol=0,
        atol=1e-4,
    )
