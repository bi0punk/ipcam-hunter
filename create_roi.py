#!/usr/bin/env python3
import argparse
import json
import os
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Ruta a la imagen")
    ap.add_argument("--out", default="roi.json", help="Archivo JSON de salida")
    ap.add_argument("--crop", default="roi_crop.png", help="Archivo del recorte")
    ap.add_argument("--resize", type=int, default=0,
                    help="Si >0, redimensiona el lado mayor a este valor SOLO para visualizar (coords se reescalan)")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"No pude leer la imagen: {args.image}")

    h0, w0 = img.shape[:2]
    scale = 1.0
    vis = img

    # Si la imagen es enorme, conviene visualizar con resize, pero mantener coords sobre la original.
    if args.resize and max(w0, h0) > args.resize:
        scale = args.resize / float(max(w0, h0))
        vis = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    # IMPORTANTE: necesita GUI (no funciona en servidor headless sin X)
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("ROI", vis, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        raise SystemExit("ROI vacío (cancelado o sin selección).")

    # Reescalar a coords de la imagen original
    x0 = int(round(x / scale))
    y0 = int(round(y / scale))
    w0_roi = int(round(w / scale))
    h0_roi = int(round(h / scale))

    # Clamp por seguridad
    x0 = max(0, min(x0, w0 - 1))
    y0 = max(0, min(y0, h0 - 1))
    w0_roi = max(1, min(w0_roi, w0 - x0))
    h0_roi = max(1, min(h0_roi, h0 - y0))

    crop = img[y0:y0 + h0_roi, x0:x0 + w0_roi]
    cv2.imwrite(args.crop, crop)

    data = {
        "image": os.path.abspath(args.image),
        "roi_xywh": {"x": x0, "y": y0, "w": w0_roi, "h": h0_roi},
        "roi_xyxy": {"x1": x0, "y1": y0, "x2": x0 + w0_roi, "y2": y0 + h0_roi},
        "image_shape_hw": [h0, w0],
        "display_scale": scale,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("OK")
    print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()