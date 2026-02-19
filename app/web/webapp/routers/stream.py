from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from ..settings import Settings
from ..services.mjpeg import mjpeg_generator

router = APIRouter()

@router.get("/live.jpg")
def live_jpg():
    s = Settings()
    if not s.live_path.exists():
        raise HTTPException(status_code=404, detail="live.jpg not found yet")
    return FileResponse(str(s.live_path), media_type="image/jpeg")

@router.get("/live.mjpeg")
def live_mjpeg():
    s = Settings()
    return StreamingResponse(
        mjpeg_generator(s.live_path, fps=s.mjpeg_fps, boundary=s.mjpeg_boundary),
        media_type=f"multipart/x-mixed-replace; boundary={s.mjpeg_boundary}",
    )