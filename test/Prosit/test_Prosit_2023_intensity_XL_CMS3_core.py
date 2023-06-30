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
    seq = np.load(
        "test/Prosit/arr_Prosit_2023_intensity_XL_CMS3_seq.npy", allow_pickle=True
    )
    charge = np.load(
        "test/Prosit/arr_Prosit_2023_intensity_XL_CMS3_charge.npy", allow_pickle=True
    )
    ces = np.load(
        "test/Prosit/arr_Prosit_2023_intensity_XL_CMS3_ces.npy", allow_pickle=True
    )

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptides_in", seq.shape, "INT32")
    in_pep_seq.set_data_from_numpy(seq)

    in_charge = grpcclient.InferInput("precursor_charge_in", charge.shape, "FP32")
    in_charge.set_data_from_numpy(charge)

    in_ces = grpcclient.InferInput("collision_energy_in", ces.shape, "FP32")
    in_ces.set_data_from_numpy(ces)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge, in_ces],
        outputs=[
            grpcclient.InferRequestedOutput("out"),
        ],
    )

    intensities = result.as_numpy("out")

    assert intensities.shape == (5, 174)

    assert np.allclose(
        intensities,
        np.load(
            "test/Prosit/arr_Prosit_2023_intensity_XL_CMS3_int_raw.npy",
            allow_pickle=True,
        ),
        rtol=0,
        atol=1e-2,
    )
