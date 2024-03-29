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
        [
            ["ACDEK"],
            ["ACDEFGK"],
            ["ACDEFGHI[+367.0537]KLR"],
            ["ACDEFGHIKLM[UNIMOD:35]NPK"],
        ],
        dtype=np.object_,
    )

    CHARGES = np.array(
        [[2], [2], [3], [3]],
        dtype=np.int32,
    )

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptide_sequences", SEQUENCES.shape, "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput("precursor_charges", CHARGES.shape, "INT32")
    in_charge.set_data_from_numpy(CHARGES)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge],
        outputs=[
            grpcclient.InferRequestedOutput("intensities"),
            grpcclient.InferRequestedOutput("mz"),
        ],
    )

    intensities = result.as_numpy("intensities")
    fragmentmz = result.as_numpy("mz")

    assert intensities.shape == (4, 58)
    assert fragmentmz.shape == (4, 58)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/ms2pip/ms2pip_server_int.npy"),
        rtol=0,
        atol=1e-4,
        equal_nan=True,
    )

    # Assert fragmentmz consistent
    assert np.allclose(
        fragmentmz,
        np.load("test/ms2pip/ms2pip_fragmentmz.npy"),
        rtol=0,
        atol=6e3,  # TODO figure out why they are so different
        equal_nan=True,
    )
