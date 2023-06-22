from test.server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
from pathlib import Path
import requests

# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")
print(MODEL_NAME)


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    seq_1 = np.load(
        "/workspace/koina/test/Prosit/arr_Prosit_2023_intensity_XL_CMS2_seq_1.npy"
    )
    seq_2 = np.load(
        "/workspace/koina/test/Prosit/arr_Prosit_2023_XL_CMS2_intensity_seq_2.npy"
    )
    charge = np.load(
        "/workspace/koina/test/Prosit/arr_Prosit_2023_XL_CMS2_intensity_charge.npy"
    )
    ces = np.load(
        "/workspace/koina/test/Prosit/arr_Prosit_2023_XL_CMS2_intensity_ces.npy"
    )

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq_1 = grpcclient.InferInput("peptides_in_1:0", seq_1.shape, "INT32")
    in_pep_seq_2 = grpcclient.InferInput("peptides_in_2:0", seq_2.shape, "INT32")
    in_pep_seq_1.set_data_from_numpy(seq_1)
    in_pep_seq_2.set_data_from_numpy(seq_2)

    in_charge = grpcclient.InferInput("precursor_charge_in:0", charge.shape, "FP32")
    in_charge.set_data_from_numpy(charge)

    in_ces = grpcclient.InferInput("collision_energy_in:0", ces.shape, "FP32")
    in_ces.set_data_from_numpy(ces)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq_1, in_pep_seq_2, in_charge, in_ces],
        outputs=[
            grpcclient.InferRequestedOutput("out/Reshape:0"),
        ],
    )

    intensities = result.as_numpy("out/Reshape:0")

    assert intensities.shape == (5, 174 * 2)

    assert np.allclose(
        intensities,
        np.load(
            "/workspace/koina/test/Prosit/arr_Prosit_2023_intensity_XL_CMS2_int_raw.npy"
        ),
        rtol=0,
        atol=1e-4,
    )
