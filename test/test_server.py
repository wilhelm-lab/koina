import requests
from server_config import SERVER_HTTP, SERVER_GRPC
import tritonclient.grpc as grpcclient


def test_server_ready_http():
    req = requests.get(f"{SERVER_HTTP}/v2/health/ready", timeout=1)
    assert req.status_code == 200


def test_server_ready_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_server_ready()


def test_server_live_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_server_live()
