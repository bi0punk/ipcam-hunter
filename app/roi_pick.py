#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import yaml
import numpy as np

def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_cfg(path: Path, cfg: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def main():
    ap = argparse.ArgumentParser(description="Dibuja ROI poligonal sobre un frame RTSP y guarda en config.yaml")
    ap.add_argument("--config", default="app/config.yaml", help="Ruta a config.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)

    rtsp = cfg["camera"]["rtsp_url"]
    process_w = int(cfg["camera"]["process_width"])

    cap = cv2.VideoCapture(rtsp)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise SystemExit("No pude leer un frame desde RTSP. Revisa URL/credenciales/red.")

    h, w = frame.shape[:2]
    scale = process_w / float(w)
    frame_p = cv2.resize(frame, (process_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    points = []
    win = "ROI picker | click=agregar | Enter=guardar | Backspace=undo | Esc=salir"

    def redraw():
        img = frame_p.copy()
        if len(points) >= 1:
            for p in points:
                cv2.circle(img, p, 3, (0, 255, 0), -1)
            if len(points) >= 2:
                cv2.polylines(img, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 2)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC
            break
        if k in (10, 13):  # Enter
            if len(points) < 3:
                print("ROI requiere al menos 3 puntos.")
                continue
            cfg["roi"]["points"] = [[int(x), int(y)] for (x, y) in points]
            save_cfg(cfg_path, cfg)
            print("ROI guardado:", cfg["roi"]["points"])
            break
        if k == 8 and points:  # Backspace
            points.pop()
            redraw()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
