from fastapi import APIRouter

from ..services.status_store import read_json
from ..settings import Settings

router = APIRouter()

@router.get("/status")
def status():
    s = Settings()
    return read_json(s.status_path)
