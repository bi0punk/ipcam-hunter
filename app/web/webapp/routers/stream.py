import os
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, Response

from webapp.services.mjpeg import FrameHub

router = APIRouter()

def _get_hub(app) -> FrameHub:
    hub = getattr(app.state, "frame_hub", None)
    if hub is None:
        hub = FrameHub(
            rtsp_url=os.environ.get("CAM_RTSP_URL_PREVIEW", ""),
            preview_width=int(os.environ.get("PREVIEW_WIDTH", "640")),
            preview_fps=int(os.environ.get("PREVIEW_FPS", "12")),
            jpeg_quality=int(os.environ.get("JPEG_QUALITY", "80")),
            overlay_json=os.environ.get("OVERLAY_JSON", "/data/events/overlay.json"),
            status_json=os.environ.get("STATUS_JSON", "/data/events/status.json"),
        )
        hub.start()
        app.state.frame_hub = hub
    return hub

@router.get("/mjpeg")
def mjpeg(request: Request):
    hub = _get_hub(request.app)
    return StreamingResponse(
        hub.mjpeg_iter(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )

@router.get("/snapshot.jpg")
def snapshot(request: Request):
    hub = _get_hub(request.app)
    jpg, meta = hub.get()
    if not jpg:
        return Response(content=f"No frame yet. err={meta.err}".encode(), media_type="text/plain", status_code=503)
    return Response(content=jpg, media_type="image/jpeg")