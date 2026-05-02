from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_positive():
    response = client.post("/predict", json={"text": "I love this airline!"})
    assert response.status_code == 200
    assert "sentiment" in response.json()

def test_predict_empty():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in [200, 500]
