from test.server_config import SERVER_GRPC, SERVER_HTTP
from pathlib import Path
import requests
import tritonclient.grpc as grpcclient
import numpy as np

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
        [
            ["AAAAAA"],
            ["PEPTIPEPTIPEPTIPEPTIPEPTIPEPT"],
            ["RHKDESTNQC[UNIMOD:4]GPAVILMFYW"],
            ["RHKDESTNQC[UNIMOD:4]GPAVILM[UNIMOD:35]FYW"],
        ],
        dtype=np.object_,
    )

    charge = np.array([[3] for _ in range(len(SEQUENCES))], dtype=np.int32)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptide_sequences", [4, 1], "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput("precursor_charges", [4, 1], "INT32")
    in_charge.set_data_from_numpy(charge)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge],
        outputs=[
            grpcclient.InferRequestedOutput("knots"),
            grpcclient.InferRequestedOutput("coefficients"),
            grpcclient.InferRequestedOutput("mz"),
            grpcclient.InferRequestedOutput("annotations"),
        ],
    )

    knots = result.as_numpy("knots")
    coefficients = result.as_numpy("coefficients")
    fragmentmz = result.as_numpy("mz")
    annotations = result.as_numpy("annotations")

    assert knots.shape == (8,)
    assert coefficients.shape == (4, 4, 200)
    assert fragmentmz.shape == (4, 200)
    assert annotations.shape == (4, 200)

    # Assert knots consistent
    assert np.allclose(
        knots,
        np.load("test/Altimeter/arr_Altimeter_2024_spline_filtered_knots.npy"),
        rtol=0,
        atol=1e-5,
        equal_nan=True,
    )

    # Assert coefficients consistent
    assert np.allclose(
        coefficients,
        np.load("test/Altimeter/arr_Altimeter_2024_spline_filtered_coefs.npy"),
        rtol=0,
        atol=1e-5,
        equal_nan=True,
    )

    # Assert mzs are consistent
    assert np.allclose(
        fragmentmz,
        np.load("test/Altimeter/arr_Altimeter_2024_spline_filtered_mz.npy"),
        rtol=0,
        atol=1e-5,
        equal_nan=True,
    )

    # Assert annotation names are consistent
    assert np.array_equal(
        annotations,
        np.load(
            "test/Altimeter/arr_Altimeter_2024_spline_filtered_annot.npy",
            allow_pickle=True,
        ),
        equal_nan=False,
    )
