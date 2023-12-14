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
    SEQUENCES = np.array(
        [
            ["TQNVLGEK[UNIMOD:1881]GRR"],
            ["TQNVLGEK[UNIMOD:1881]GRR"],
            ["TQNVLGEK[UNIMOD:1881]GRR"],
            ["TQQASK[UNIMOD:1881]HIR"],
            ["TQQASK[UNIMOD:1881]HIR"],
        ],
        dtype=np.object_,
    )

    charge = np.array([[2], [2], [2], [2], [2]], dtype=np.int32)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptide_sequences", [5, 1], "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput("precursor_charges", [5, 1], "INT32")
    in_charge.set_data_from_numpy(charge)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge],
        outputs=[
            grpcclient.InferRequestedOutput("intensities"),
            grpcclient.InferRequestedOutput("mz"),
            grpcclient.InferRequestedOutput("annotation"),
        ],
    )

    intensities = result.as_numpy("intensities")
    fragmentmz = result.as_numpy("mz")
    annotation = result.as_numpy("annotation")

    assert intensities.shape == (5, 174)
    assert fragmentmz.shape == (5, 174)
    assert annotation.shape == (5, 174)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/Prosit/arr_Prosit_2023_intensity_XL_CMS3_int.npy"),
        rtol=0,
        atol=1e-2,
        equal_nan=True,
    )
