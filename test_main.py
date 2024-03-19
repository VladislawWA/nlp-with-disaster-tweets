from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/info")
    assert response.status_code == 200
    assert response.json() == {"info": "Naive Bayas model"}


def test_read_predict():
    response = client.post(url="/predict/",
                           json={"text": "Storm clouds are approaching London"})
    json_data = response.json()

    assert response.status_code == 200
    assert json_data == {"predict": "1"}

