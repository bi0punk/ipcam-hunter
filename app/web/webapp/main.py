from fastapi import FastAPI

from .routers.api import router as api_router
from .routers.stream import router as stream_router
from .routers.ui import router as ui_router


def create_app() -> FastAPI:
    app = FastAPI(title="ROI Dashboard", version="1.0.0")

    app.include_router(ui_router)
    app.include_router(api_router, prefix="/api", tags=["api"])
    app.include_router(stream_router, prefix="/api", tags=["stream"])

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness para Docker/orquestadores."""
        return {"status": "ok", "service": "roi-dashboard"}

    return app


app = create_app()
