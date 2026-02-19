import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth


def fetch_isapi_snapshot(
    url: str,
    username: str,
    password: str,
    auth_mode: str = "digest",
    timeout_s: float = 4.0
) -> bytes:
    auth_mode = (auth_mode or "digest").strip().lower()
    if auth_mode == "basic":
        auth = HTTPBasicAuth(username, password)
    else:
        auth = HTTPDigestAuth(username, password)

    r = requests.get(url, auth=auth, timeout=timeout_s)
    r.raise_for_status()
    data = r.content or b""

    if not data.startswith(b"\xff\xd8"):
        head = data[:200].decode("utf-8", errors="replace")
        raise RuntimeError(f"ISAPI snapshot no es JPEG (len={len(data)}). Head: {head}")

    return data