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

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptide_sequences", SEQUENCES.shape, "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput("precursor_charge", charge.shape, "INT32")
    in_charge.set_data_from_numpy(charge)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge],
        outputs=[grpcclient.InferRequestedOutput("ccs")],
    )

    ccs = result.as_numpy("ccs")

    # Assert intensities consistent
    assert np.allclose(
        ccs,
        np.load("test/AlphaPept/arr_AlphaPept_ccs_raw.npy"),
        rtol=0,
        atol=1e-4,
    )
