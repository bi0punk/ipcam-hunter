from fastapi.testclient import TestClient

from webapp.main import app


def test_health_responde_ok():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "roi-dashboard"}
