from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
import requests
from pathlib import Path


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
        [["TPVISGGPYEYR"], ["TPVITGAPYEYR"], ["GTFIIDPGGVIR"], ["GTFIIDPAAVIR"]],
        dtype=np.object_,
    )

    charge = np.array([[2] for _ in range(len(SEQUENCES))], dtype=np.int32)
    ces = np.array([[30] for _ in range(len(SEQUENCES))], dtype=np.float32)
    instr = np.array([["QE"] for _ in range(len(SEQUENCES))], dtype=np.object_)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptide_sequences", SEQUENCES.shape, "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput("precursor_charges", charge.shape, "INT32")
    in_charge.set_data_from_numpy(charge)

    in_ces = grpcclient.InferInput("collision_energies", ces.shape, "FP32")
    in_ces.set_data_from_numpy(ces)

    in_instr = grpcclient.InferInput("instrument_types", instr.shape, "BYTES")
    in_instr.set_data_from_numpy(instr)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge, in_ces, in_instr],
        outputs=[
            grpcclient.InferRequestedOutput("intensities"),
            grpcclient.InferRequestedOutput("mz"),
        ],
    )

    intensities = result.as_numpy("intensities")
    fragmentmz = result.as_numpy("mz")

    assert intensities.shape == fragmentmz.shape == (4, 44)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/AlphaPept/arr_AlphaPept_ms2_int_norm.npy"),
        rtol=0,
        atol=1e-5,
    )
