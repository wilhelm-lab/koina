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
    SEQUENCES = np.load(
        "test/Prosit/arr_Prosit_2023_intensity_TOF_seq.npy", allow_pickle=True
    )
    charge = np.load("test/Prosit/arr_Prosit_2023_intensity_TOF_charge.npy")
    ces = np.load("test/Prosit/arr_Prosit_2023_intensity_TOF_ce.npy")

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptide_sequences", SEQUENCES.shape, "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput("precursor_charges", charge.shape, "INT32")
    in_charge.set_data_from_numpy(charge)

    in_ces = grpcclient.InferInput("collision_energies", ces.shape, "FP32")
    in_ces.set_data_from_numpy(ces)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge, in_ces],
        outputs=[
            grpcclient.InferRequestedOutput("intensities"),
            grpcclient.InferRequestedOutput("mz"),
            grpcclient.InferRequestedOutput("annotation"),
        ],
    )

    intensities = result.as_numpy("intensities")
    fragmentmz = result.as_numpy("mz")
    annotation = result.as_numpy("annotation")

    assert intensities.shape == (SEQUENCES.shape[0], 174)
    assert fragmentmz.shape == (SEQUENCES.shape[0], 174)
    assert annotation.shape == (SEQUENCES.shape[0], 174)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/Prosit/arr_Prosit_2023_intensity_TOF_int.npy"),
        rtol=0,
        atol=1e-5,
        equal_nan=True,
    )
