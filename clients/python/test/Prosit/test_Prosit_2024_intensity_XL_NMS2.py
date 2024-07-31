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


def test_inference(capfd):
    SEQUENCES_1 = np.array(
        [
            ["AAAK[UNIMOD:1898]HLIER"],
            ["AAAK[UNIMOD:1898]HLIER"],
            ["AAAK[UNIMOD:1898]HLIER"],
            ["AAK[UNIMOD:1898]ALGLVIR"],
            ["AAK[UNIMOD:1898]ALGLVIR"],
        ],
        dtype=np.object_,
    )
    SEQUENCES_2 = np.array(
        [
            ["LYK[UNIMOD:1898]INAK"],
            ["LYK[UNIMOD:1898]INAK"],
            ["LYK[UNIMOD:1898]INAK"],
            ["LK[UNIMOD:1898]AADALGK"],
            ["LK[UNIMOD:1898]AADALGK"],
        ],
        dtype=np.object_,
    )

    charge = np.array([[4], [4], [5], [4], [3]], dtype=np.int32)
    ces = np.array([[32], [32], [31], [31], [31]], dtype=np.float32)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq_1 = grpcclient.InferInput("peptide_sequences_1", [5, 1], "BYTES")
    in_pep_seq_2 = grpcclient.InferInput("peptide_sequences_2", [5, 1], "BYTES")
    in_pep_seq_1.set_data_from_numpy(SEQUENCES_1.astype(np.bytes_))
    in_pep_seq_2.set_data_from_numpy(SEQUENCES_2.astype(np.bytes_))

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

    captured = capfd.readouterr()
    output_lines = captured.out.splitlines()

    assert intensities.shape == (5, 174)
    assert fragmentmz.shape == (5, 174)
    assert annotation.shape == (5, 174)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load(
            "/workspace/koina/clients/python/test/Prosit/arr_Prosit_2024_intensity_XL_NMS2_int.npy"
        ),
        rtol=0,
        atol=1e-3,
        equal_nan=True,
    )
