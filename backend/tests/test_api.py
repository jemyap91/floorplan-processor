import os
import pytest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = ":memory:"
from backend.main import app

client = TestClient(app)

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "..", "input sample.pdf")

class TestHealthCheck:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

class TestProjectEndpoints:
    def test_list_projects_empty(self):
        response = client.get("/api/projects")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

class TestExportEndpoint:
    def test_export_no_project(self):
        response = client.get("/api/export/nonexistent?format=json")
        assert response.status_code == 404
