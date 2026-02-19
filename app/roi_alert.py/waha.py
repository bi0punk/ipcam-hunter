import base64
from pathlib import Path

import requests


class WahaClient:
    def __init__(self, base_url: str, session: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.api_key = api_key.strip()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-Api-Key"] = self.api_key
        return h

    def send_image_b64(self, chat_id: str, jpeg_path: Path, caption: str = "") -> dict:
        data_b64 = base64.b64encode(jpeg_path.read_bytes()).decode("ascii")
        payload = {
            "session": self.session,
            "chatId": chat_id,
            "file": {
                "mimetype": "image/jpeg",
                "filename": jpeg_path.name,
                "data": data_b64
            },
            "caption": caption
        }
        url = f"{self.base_url}/api/sendImage"
        r = requests.post(url, json=payload, headers=self._headers(), timeout=25)
        r.raise_for_status()
        return r.json() if r.content else {"status": "ok"}