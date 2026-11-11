"""Nombres usados por FastAPI, callbacks de OpenCV o desde los tests.

Vulture no sigue los decoradores de rutas ni la inyección de FastAPI, así que
estos símbolos se declaran aquí para poder correr el chequeo sin `|| true`.
"""

from app.roi_alert.cfg import resolve_events_dir
from app.roi_alert.events import EventContext, EventWorker
from app.roi_alert.rtsp import LatestFrameGrabber
from webapp.routers.api import status
from webapp.routers.stream import mjpeg, snapshot
from webapp.routers.ui import index
from webapp.services.mjpeg import FrameHub

EventContext.from_parts
EventWorker.stop
LatestFrameGrabber.stop
FrameHub.stop
resolve_events_dir
status
mjpeg
snapshot
index
