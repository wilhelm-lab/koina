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
    SEQUENCES = np.array(
        [
            ["AA"],
            ["PEPTIPEPTIPEPTIPEPTIPEPTIPEPT"],
            ["RHKDESTNQCGPAVILMFYW"],
            ["RHKDESTNQC[UNIMOD:4]GPAVILMFYW"],
            ["RHKDESTNQCGPAVILM[UNIMOD:35]FYW"],
        ],
        dtype=np.object_,
    )

    charge = np.array([[3] for _ in range(len(SEQUENCES))], dtype=np.float32)

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptide_sequences", [5, 1], "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    in_charge = grpcclient.InferInput("precursor_charges", [5, 1], "INT32")
    in_charge.set_data_from_numpy(charge)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq, in_charge],
        outputs=[
            grpcclient.InferRequestedOutput("knots"),
            grpcclient.InferRequestedOutput("coefficients"),
            grpcclient.InferRequestedOutput("mz"),
            grpcclient.InferRequestedOutput("annotation"),
        ],
    )

    knots = result.as_numpy("knots")
    coefficients = result.as_numpy("coefficients")
    fragmentmz = result.as_numpy("mz")
    annotation = result.as_numpy("annotation")

    assert knots.shape == (8)
    assert coefficients.shape == (5, 4, 200)
    assert fragmentmz.shape == (5, 200)
    assert annotation.shape == (5, 200)
    
    print(knots)
    print(coefficients)
    print(fragmentmz)
    print(annotation)

    # Assert knots consistent
    assert np.allclose(knots, 
                       np.array([6, 13,21.7466, 24.7378, 32.5236, 39.1529, 48, 55]), # TODO Update  
                       rtol=0,
                       atol=1e-5,
                       equal_nan=True)
    
    # Assert coefficients consistent
    assert np.allclose(
        coefficients,
        np.load("test/Prosit/arr_Altimter_2024_splines_coef.npy"), #  TODO Update 
        rtol=0,
        atol=1e-5,
        equal_nan=True,
    )
    
    # Assert mzs are consistent
    assert np.allclose(
        fragmentmz,
        np.load("test/Prosit/arr_Altimter_2024_splines_mz.npy"), #  TODO Update 
        rtol=0,
        atol=1e-5,
        equal_nan=True,
    )
    
    # Assert annotation names are consistent
    assert np.array_equal(
        annotation,
        np.load("test/Prosit/arr_Altimter_2024_splines_annotation.npy"), #  TODO Update 
        equal_nan=True,
    )