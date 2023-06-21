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
    SEQUENCES_1 = np.array(
        [
            ["DIADAVTAAGVEVAKSEVR"],
            ["PEPTIDEK[UNIMOD:1896]PEPTIDEK"],
            ["K[UNIMOD:1884]PEPTIDEK"],
            ["PEPTIDEK[UNIMOD:1884]STNQC[UNIMOD:4]"],
            ["RHKC[UNIMOD:4]ESTK[UNIMOD:1896]"],
        ],
        dtype=np.object_,
    )
    SEQUENCES_2 = np.array(
        [
            ["PEPTIPEPK[UNIMOD:1896]TIPPT"],
            ["AAK[UNIMOD:1896]QGP"],
            ["QM[UNIMOD:35]GK[UNIMOD:1884]P"],
            ["AK[UNIMOD:1884]TNQ"],
            ["PEPKK[UNIMOD:1896]PEPK"],
        ],
        dtype=np.object_,
    )

    charge = np.array([[3] for _ in range(len(SEQUENCES_1))], dtype=np.int32)
    ces = np.array([[25] for _ in range(len(SEQUENCES_1))], dtype=np.float32)

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
    assert fragmentmz.shape == (5, 174 * 2)
    assert annotation.shape == (5, 174 * 2)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/Prosit/arr_Prosit_2023_intensity_XL_CMS2_int.npy"),
        rtol=0,
        atol=1e-5,
        equal_nan=True,
    )
