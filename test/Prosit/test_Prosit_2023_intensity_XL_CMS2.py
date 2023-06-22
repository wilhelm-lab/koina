from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
from pathlib import Path
import requests

# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 400


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    SEQUENCES_1 = np.array(
        [
            ["DIADAVTAAGVEVAK[UNIMOD:1896]SEVR"],
            ["AGDQIQSGVDAAIK[UNIMOD:1896]PGNTLPMR"],
            ["LIVVEK[UNIMOD:1896]FSVEAPK"],
            ["ANPWQQFAETHNK[UNIMOD:1896]GDRVEGK"],
            ["VLESAIANAEHNDGADIDDLK[UNIMOD:1896]VTK"],
        ],
        dtype=np.object_,
    )
    SEQUENCES_2 = np.array(
        [
            ["NFLVPQGK[UNIMOD:1896]AVPATK"],
            ["SANIALVLYK[UNIMOD:1896]DGER"],
            ["K[UNIMOD:1896]ELVLK"],
            ["AAGAELVGMEDLADQIK[UNIMOD:1896]K"],
            ["K[UNIMOD:1896]VSQALDILTYTKK"],
        ],
        dtype=np.object_,
    )

    charge = np.array([[3], [4], [3], [5], [6]], dtype=np.int32)
    ces = np.array([[28], [28], [28], [25], [25]], dtype=np.float32)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq_1 = grpcclient.InferInput("peptide_sequences_1", [5, 1], "BYTES")
    in_pep_seq_2 = grpcclient.InferInput("peptide_sequences_2", [5, 1], "BYTES")
    in_pep_seq_1.set_data_from_numpy(SEQUENCES_1)
    in_pep_seq_2.set_data_from_numpy(SEQUENCES_2)

    in_charge = grpcclient.InferInput("precursor_charges", [5, 1], "INT32")
    in_charge.set_data_from_numpy(charge)

    in_ces = grpcclient.InferInput("collision_energies", [5, 1], "FP32")
    in_ces.set_data_from_numpy(ces)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq_1, in_pep_seq_2, in_charge, in_ces],
        outputs=[
            grpcclient.InferRequestedOutput("intensities"),
            grpcclient.InferRequestedOutput("mz"),
            grpcclient.InferRequestedOutput("annotation"),
        ],
    )

    intensities = result.as_numpy("intensities")
    fragmentmz = result.as_numpy("mz")
    annotation = result.as_numpy("annotation")

    assert intensities.shape == (5, 174 * 2)
    assert fragmentmz.shape == (5, 174)
    assert annotation.shape == (5, 174 * 2)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/Prosit/arr_Prosit_2023_intensity_XL_CMS2_int.npy"),
        rtol=0,
        atol=1e-5,
        equal_nan=True,
    )
