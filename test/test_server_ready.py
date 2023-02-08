import requests
from test_config import SERVER_HTTP


def test_http_server_ready():
    req = requests.get(f"{SERVER_HTTP}/v2/health/ready", timeout=1)
    assert req.status_code == 200
