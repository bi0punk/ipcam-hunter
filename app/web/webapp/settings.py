import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    events_dir: Path = Path(os.getenv("EVENTS_DIR", "/data/events"))
    status_filename: str = os.getenv("STATUS_FILENAME", "status.json")

    @property
    def status_path(self) -> Path:
        return self.events_dir / self.status_filename
