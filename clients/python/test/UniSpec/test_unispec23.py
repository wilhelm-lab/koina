from test.server_config import SERVER_GRPC, SERVER_HTTP
from pathlib import Path
from test.lib import (
    lib_test_available_grpc,
    lib_test_available_http,
    lib_test_inference,
)


# To ensure MODEL_NAME == test_<filename>.py
MODEL_NAME = Path(__file__).stem.replace("test_", "")


def test_available_http():
    lib_test_available_http(MODEL_NAME, SERVER_HTTP)


def test_available_grpc():
    lib_test_available_grpc(MODEL_NAME, SERVER_GRPC)


def test_inference():
    lib_test_inference(MODEL_NAME, SERVER_GRPC)

# from test.server_config import SERVER_GRPC, SERVER_HTTP
# import tritonclient.grpc as grpcclient
# import numpy as np
# import requests
# from pathlib import Path
# 
# 
# # To ensure MODEL_NAME == test_<filename>.py
# MODEL_NAME = Path(__file__).stem.replace("test_", "")
# 
# 
# def test_available_http():
#     req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
#     assert req.status_code == 200
# 
# 
# def test_available_grpc():
#     triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
#     assert triton_client.is_model_ready(MODEL_NAME)
# 
# 
# def test_inference():
#     INPUT = np.load("test/UniSpec/arr-UniSpec_unispec23_input_tensor.npy")
#     triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
#     in_INPUT = grpcclient.InferInput("input_tensor", INPUT.shape, "FP32")
#     in_INPUT.set_data_from_numpy(INPUT)
#     result = triton_client.infer(
#         MODEL_NAME,
#         inputs=[in_INPUT],
#         outputs=[
#             grpcclient.InferRequestedOutput("intensities"),
#         ],
#     )
# 
#     intensities = result.as_numpy("intensities")
# 
#     assert intensities.shape == (50, 7919)
# 
#     # Assert intensities consistent
#     assert np.allclose(
#         intensities,
#         np.load("test/UniSpec/arr-UniSpec_unispec23_output_tensor.npy"),
#         rtol=0,
#         atol=1e-5,
#     )
