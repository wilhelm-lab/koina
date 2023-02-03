import requests

def test_http_server_ready():
    r = requests.get("http://dev:8501/v2/health/ready")
    assert r.status_code == 200
