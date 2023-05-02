from server_config import SERVER_GRPC, SERVER_HTTP
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
    seq = np.load("test/AlphaPept/arr_AlphaPept_rt_aa.npy")
    mod = np.load("test/AlphaPept/arr_AlphaPept_rt_mod.npy")
    charge = np.load("test/AlphaPept/arr_AlphaPept_ms2_charge.npy")
    nce = np.load("test/AlphaPept/arr_AlphaPept_ms2_nce.npy")
    instr = np.load("test/AlphaPept/arr_AlphaPept_ms2_instrument.npy").reshape([4, 1])

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("aa_indices__0", seq.shape, "INT64")
    in_pep_seq.set_data_from_numpy(seq)

    in_mod = grpcclient.InferInput("mod_x__1", mod.shape, "FP32")
    in_mod.set_data_from_numpy(mod)

    in_charge = grpcclient.InferInput("charges__2", charge.shape, "FP32")
    in_charge.set_data_from_numpy(charge)

    in_nce = grpcclient.InferInput("NCEs__3", nce.shape, "FP32")
    in_nce.set_data_from_numpy(nce)

    in_instr = grpcclient.InferInput("instrument_indices__4", instr.shape, "INT64")
    in_instr.set_data_from_numpy(instr)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_mod, in_charge, in_nce, in_instr],
        outputs=[
            grpcclient.InferRequestedOutput("output__0"),
        ],
    )

    intensities = result.as_numpy("output__0")

    assert intensities.shape == (4, 11, 8)

    assert np.allclose(
        intensities,
        np.load("test/AlphaPept/arr_AlphaPept_ms2_raw.npy"),
        rtol=0,
        atol=1e-5,
    )
