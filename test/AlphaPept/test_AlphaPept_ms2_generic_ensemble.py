from server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
import requests

MODEL_NAME = "AlphaPept_ms2_generic_ensemble"


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
    ces = np.array([[30] for _ in range(len(SEQUENCES))], dtype=np.int32)
    instr = np.array([[0] for _ in range(len(SEQUENCES))], dtype=np.int64)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptides_in_str:0", SEQUENCES.shape, "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput(
        "precursor_charge_in_int:0", charge.shape, "INT32"
    )
    in_charge.set_data_from_numpy(charge)

    in_ces = grpcclient.InferInput("collision_energy_in:0", ces.shape, "INT32")
    in_ces.set_data_from_numpy(ces)

    in_instr = grpcclient.InferInput("instrument_indices:0", instr.shape, "INT64")
    in_instr.set_data_from_numpy(instr)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge, in_ces, in_instr],
        outputs=[
            grpcclient.InferRequestedOutput("out/Reshape:0"),
        ],
    )

    intensities = result.as_numpy("out/Reshape:0")

    assert intensities.shape == (4, 11, 8)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/AlphaPept/arr_AlphaPept_ms2_raw.npy"),
        rtol=0,
        atol=1e-5,
    )
