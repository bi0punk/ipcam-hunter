from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    events_dir: Path = Path(os.getenv("EVENTS_DIR", "/data/events"))
    live_filename: str = os.getenv("LIVE_FILENAME", "live.jpg")
    status_filename: str = os.getenv("STATUS_FILENAME", "status.json")
    mjpeg_fps: float = float(os.getenv("MJPEG_FPS", "8"))
    mjpeg_boundary: str = "frame"

    @property
    def live_path(self) -> Path:
        return self.events_dir / self.live_filename

    @property
    def status_path(self) -> Path:
        return self.events_dir / self.status_filename