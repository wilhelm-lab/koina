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
    """
    SEQUENCES = np.array(
        [
            ["LGGNEQVTR"],
            ["GAGSSEPVTGLDAK"],
            ["VEATFGVDESNAK"],
            ["YILAGVENSK"],
            ["TPVISGGPYEYR"],
            ["TPVITGAPYEYR"],
            ["DGLDAASYYAPVR"],
            ["ADVTPADFSEWSK"],
            ["GTFIIDPGGVIR"],
            ["GTFIIDPAAVIR"],
            ["LFLQFGAQGSPFLK"],
        ],
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
    """
    INPUT = np.load("test/UniSpec/test_input_tensor.npy")
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    in_INPUT = grpcclient.InferInput("input_tensor", INPUT.shape, "FP32")
    in_INPUT.set_data_from_numpy(INPUT)
    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_INPUT],
        outputs=[
            grpcclient.InferRequestedOutput("intensities"),
        ],
    )

    intensities = result.as_numpy("intensities")

    assert intensities.shape == (50, 7919)

    # Assert intensities consistent
    assert np.allclose(
        intensities,
        np.load("test/UniSpec/test_output_tensor.npy"),
        rtol=0,
        atol=1e-5,
    )
