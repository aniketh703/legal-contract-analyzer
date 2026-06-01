"""
tests/test_app.py — Flask route smoke tests.
"""
import importlib

import pytest


@pytest.fixture
def flask_app():
    mod = importlib.import_module("app.main")
    mod.app.config["TESTING"] = True
    return mod.app


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


class TestFlaskRoutes:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "ok"}

    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"html" in resp.data.lower() or resp.mimetype == "text/html"
