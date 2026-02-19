from fastapi import APIRouter
from ..settings import Settings
from ..services.status_store import read_json

router = APIRouter()

@router.get("/status")
def status():
    s = Settings()
    return read_json(s.status_path)