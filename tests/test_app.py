"""Basic smoke tests for the SDEWS Flask app."""
from app import create_app


def test_health_endpoint():
    app = create_app("testing")
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_index_page_renders():
    app = create_app("testing")
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"SDEWS" in resp.data


def test_dashboard_requires_login():
    app = create_app("testing")
    client = app.test_client()
    resp = client.get("/dashboard")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]