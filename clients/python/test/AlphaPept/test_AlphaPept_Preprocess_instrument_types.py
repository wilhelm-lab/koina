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
    lib_test_inference(MODEL_NAME, SERVER_GRPC, atol=1e-4)

