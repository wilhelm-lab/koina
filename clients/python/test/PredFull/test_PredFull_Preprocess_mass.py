from test.server_config import SERVER_GRPC, SERVER_HTTP
from pathlib import Path
from test.lib import lib_test_available_grpc, lib_test_available_http
import numpy as np
import tritonclient.grpc as grpcclient

# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    lib_test_available_http(MODEL_NAME, SERVER_HTTP)


def test_available_grpc():
    lib_test_available_grpc(MODEL_NAME, SERVER_GRPC)


def test_inference():

    SEQUENCES = np.array(
        [["HIISVM[UNIMOD:35]R"]],
        dtype=np.object_,
    )

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("peptide_sequences", [1, 1], "BYTES")
    in_pep_seq.set_data_from_numpy(SEQUENCES)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq],
        outputs=[
            grpcclient.InferRequestedOutput("precursor_mass"),
        ],
    )

    mass = result.as_numpy("precursor_mass")
    ground_truth = 0.0427743459608515

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq],
        outputs=[
            grpcclient.InferRequestedOutput("precursor_mass_with_oxM"),
        ],
    )

    mass = result.as_numpy("precursor_mass_with_oxM")
    ground_truth = 871.481819
    print(mass)
    assert np.allclose(mass[0], ground_truth, atol=1e-4)
