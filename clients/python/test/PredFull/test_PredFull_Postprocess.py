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
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    spectrum = np.load("test/PredFull/numpy_arrays/HIISVMoxR2CID30_rawspectrum.npy")
    in_spectrum = grpcclient.InferInput("spectrum", spectrum.shape, "FP32")
    in_spectrum.set_data_from_numpy(spectrum)

    mass = np.array([[871.4]], dtype=np.float32)
    in_mass = grpcclient.InferInput("precursor_mass_with_oxM", [1, 1], "FP32")
    in_mass.set_data_from_numpy(mass)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_spectrum, in_mass],
        outputs=[
            grpcclient.InferRequestedOutput("mzs"),
            grpcclient.InferRequestedOutput("intensities"),
        ],
    )

    mzs = result.as_numpy("mzs")
    intensities = result.as_numpy("intensities")
    print(mzs)
    print(intensities)
    print(mzs.shape)

    assert intensities.shape == mzs.shape

    ground_truth_intensities = np.load(
        "test/PredFull/numpy_arrays/HIISVMoxR2CID30_intensities.npy"
    )
    ground_truth_mzs = np.load("test/PredFull/numpy_arrays/HIISVMoxR2CID30_mzs.npy")

    ground_truth_mzs = ground_truth_mzs[ground_truth_intensities > 1]
    ground_truth_intensities = ground_truth_intensities[ground_truth_intensities > 1]

    assert np.allclose(
        intensities,
        ground_truth_intensities,
        rtol=0,
        atol=1e-4,
    )

    assert np.allclose(
        mzs,
        ground_truth_mzs,
        rtol=0,
        atol=1e-4,
    )
