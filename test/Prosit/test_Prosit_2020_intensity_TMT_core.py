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
    seq = np.load("test/Prosit/arr_Prosit_2019_intensity_seq.npy")
    charge = np.load("test/Prosit/arr_Prosit_2019_intensity_charge.npy")
    ces = np.load("test/Prosit/arr_Prosit_2019_intensity_ces.npy")
    frag = np.load("test/Prosit/arr_Prosit_2020_intensityTMT_frag.npy").reshape([5, 1])

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptides_in:0", seq.shape, "INT32")
    in_pep_seq.set_data_from_numpy(seq)

    in_charge = grpcclient.InferInput("precursor_charge_in:0", charge.shape, "FP32")
    in_charge.set_data_from_numpy(charge)

    in_ces = grpcclient.InferInput("collision_energy_in:0", ces.shape, "FP32")
    in_ces.set_data_from_numpy(ces)

    in_frag = grpcclient.InferInput("fragmentation_type_in:0", frag.shape, "FP32")
    in_frag.set_data_from_numpy(frag)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge, in_ces, in_frag],
        outputs=[
            grpcclient.InferRequestedOutput("out/Reshape:0"),
        ],
    )

    intensities = result.as_numpy("out/Reshape:0")

    assert intensities.shape == (5, 174)

    assert np.allclose(
        intensities,
        np.load("test/Prosit/arr_Prosit_2020_intensityTMT_int_raw.npy"),
        rtol=0,
        atol=1e-4,
    )
