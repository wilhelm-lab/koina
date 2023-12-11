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
    x = np.array(
        ["AAGPSLSHTSGGTQSK", "AAINQKLIETGER", "AANDAGYFNDEMAPIEVKTK", "ACDEFGHIKLMNPK"],
        dtype=object,
    ).reshape(-1, 1)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    inputs = [grpcclient.InferInput("peptide_sequences", x.shape, "BYTES")]
    inputs[0].set_data_from_numpy(x)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=inputs,
        outputs=[
            grpcclient.InferRequestedOutput("irt"),
        ],
    )

    preds = result.as_numpy("irt")

    assert preds.shape == (4, 1)

    assert np.allclose(
        preds,
        np.load("test/Deeplc/arr_preds.npy"),
        rtol=0,
        atol=1e-5,
    )
