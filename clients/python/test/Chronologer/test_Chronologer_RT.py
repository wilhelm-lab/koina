import numpy as np
from test.server_config import SERVER_GRPC, SERVER_HTTP
from pathlib import Path
from test.lib import (
    lib_test_available_grpc,
    lib_test_available_http,
    lib_test_inference,
)
import tritonclient.grpc as grpcclient


# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    lib_test_available_http(MODEL_NAME, SERVER_HTTP)


def test_available_grpc():
    lib_test_available_grpc(MODEL_NAME, SERVER_GRPC)


def test_inference():
    lib_test_inference(MODEL_NAME, SERVER_GRPC, 1e-5)


# def test_inference():
#     x = np.array(
#         [
#             "AAGPSLSHTSGGTQSK",
#             "AAINQKLIETGER",
#             "AANDAGYFNDEMAPIEVKTK",
#             "ACDEFGHIKLMNPK",
#             "E[UNIMOD:27]GC[UNIMOD:4]HNY[UNIMOD:21]PPDK[UNIMOD:737]",
#             "[UNIMOD:739]-AAGS[UNIMOD:21]R[UNIMOD:36]NNHK[UNIMOD:739]",
#         ],
#         dtype=object,
#     ).reshape(-1, 1)

#     triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

#     inputs = [grpcclient.InferInput("peptide_sequences", x.shape, "BYTES")]
#     inputs[0].set_data_from_numpy(x)

#     result = triton_client.infer(
#         MODEL_NAME,
#         inputs=inputs,
#         outputs=[
#             grpcclient.InferRequestedOutput("rt"),
#         ],
#     )

#     preds = result.as_numpy("rt")

#     assert preds.shape == (6, 1)

#     assert np.allclose(
#         preds,
#         np.load("test/Chronologer/arr_preds.npy"),
#         rtol=0,
#         atol=1e-5,
#     )
