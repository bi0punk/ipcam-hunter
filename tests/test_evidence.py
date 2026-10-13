import pytest
import requests

from roi_alert.evidence import fetch_isapi_snapshot


JPEG_HEAD = b"\xff\xd8\xff\xe0resto-del-jpeg"
NOT_JPEG = b"<html>error</html>"


def test_devuelve_bytes_jpeg(monkeypatch):
    calls = {}

    class FakeResp:
        content = JPEG_HEAD
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_get(url, auth=None, timeout=None):
        calls["url"] = url
        calls["auth"] = auth
        calls["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr(requests, "get", fake_get)
    data = fetch_isapi_snapshot("http://cam/picture", "u", "p", "digest", 3.0)
    assert data == JPEG_HEAD
    assert calls["url"] == "http://cam/picture"
    assert calls["timeout"] == 3.0
    # digest -> HTTPDigestAuth
    assert isinstance(calls["auth"], requests.auth.HTTPDigestAuth)
    assert calls["auth"].username == "u"


def test_auth_basic(monkeypatch):
    seen = {}

    class FakeResp:
        content = JPEG_HEAD

        def raise_for_status(self):
            return None

    def fake_get(url, auth=None, timeout=None):
        seen["auth"] = auth
        return FakeResp()

    monkeypatch.setattr(requests, "get", fake_get)
    fetch_isapi_snapshot("http://cam/picture", "u", "p", auth_mode="basic")
    assert isinstance(seen["auth"], requests.auth.HTTPBasicAuth)


def test_respuesta_no_jpeg_lanza_runtimeerror(monkeypatch):
    class FakeResp:
        content = NOT_JPEG

        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    with pytest.raises(RuntimeError, match="no es JPEG"):
        fetch_isapi_snapshot("http://cam/picture", "u", "p")


def test_error_http_se_propaga(monkeypatch):
    def raiser(*a, **k):
        raise requests.HTTPError("401 Unauthorized")

    monkeypatch.setattr(requests, "get", raiser)
    with pytest.raises(requests.HTTPError):
        fetch_isapi_snapshot("http://cam/picture", "u", "p")


def test_auth_mode_vacio_usa_digest(monkeypatch):
    seen = {}

    class FakeResp:
        content = JPEG_HEAD

        def raise_for_status(self):
            return None

    def fake_get(url, auth=None, timeout=None):
        seen["auth"] = auth
        return FakeResp()

    monkeypatch.setattr(requests, "get", fake_get)
    fetch_isapi_snapshot("http://cam/picture", "u", "p", auth_mode=None)
    assert isinstance(seen["auth"], requests.auth.HTTPDigestAuth)


def test_content_vacio_se_trata_como_no_jpeg(monkeypatch):
    class FakeResp:
        content = b""

        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    with pytest.raises(RuntimeError, match="no es JPEG"):
        fetch_isapi_snapshot("http://cam/picture", "u", "p")