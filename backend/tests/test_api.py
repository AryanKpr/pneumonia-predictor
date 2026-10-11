from fastapi.testclient import TestClient
from main import app

def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

def test_predict_valid_image():
    with TestClient(app) as client:
        with open("tests/sample_xray.jpeg", "rb") as f:
            r = client.post("/predict", files={"file": ("x.jpeg", f, "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["result"] in {"NORMAL", "PNEUMONIA"}

def test_rejects_non_image():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
        assert r.status_code == 400