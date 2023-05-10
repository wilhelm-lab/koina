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
    SEQUENCES = np.array(
        [["ACDEK/2"], ["ACDEFGK/2"], ["ACDEFGHIKLR/3"], ["ACDEFGHIKLMNPK/3"]],
        dtype=np.object_,
    )

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("proforma_ensemble", SEQUENCES.shape, "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq],
        outputs=[
            grpcclient.InferRequestedOutput("intensities"),
        ],
    )

    intensities = result.as_numpy("intensities")

    assert intensities.shape == (4, 58)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/ms2pip/ms2pip_server_int.npy"),
        rtol=0,
        atol=1e-4,
        equal_nan=True,
    )
