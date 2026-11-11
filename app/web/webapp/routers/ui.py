from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Starlette 1.x: el request va primero y se inyecta en el contexto.
    return templates.TemplateResponse(request, "index.html")
