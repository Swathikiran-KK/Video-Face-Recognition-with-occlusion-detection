#!/usr/bin/env python3
import base64, io, os, tempfile
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = os.environ.get("YOLO_FACE_MODEL", "yolov8n-face.pt")
DEVICE = os.environ.get("DEVICE", "cpu")   # "cuda" if GPU available

app = Flask(__name__)
model = YOLO(MODEL_PATH)

# ----------------------------
# Helpers
# ----------------------------
def b64_to_temp_video(video_b64: str) -> str:
    video_bytes = base64.b64decode(video_b64)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_bytes)
    tmp.close()
    return tmp.name


def jpeg_b64_raw(bgr: np.ndarray, quality: int = 90) -> str:
    """Return RAW base64 of JPEG (no data:image/... prefix)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def clip_box(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_faces(candidates: List[Dict[str, Any]], iou_thr: float = 0.6) -> List[Dict[str, Any]]:
    """Keep highest score faces, suppress near-duplicates by IoU."""
    out = []
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        keep = True
        for k in out:
            if iou(c["bbox"], k["bbox"]) >= iou_thr and abs(c["frame_index"] - k["frame_index"]) <= 2:
                keep = False
                break
        if keep:
            out.append(c)
    return out


def increase_brightness(bgr: np.ndarray, factor: float = 1.25) -> np.ndarray:
    """
    Increase brightness using LAB color space.
    factor > 1.0 increases brightness.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = np.clip(l * factor, 0, 255).astype(np.uint8)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----------------------------
# API
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model": MODEL_PATH})


@app.route("/video_preprocess", methods=["POST"])
def video_preprocess():
    payload = request.get_json(force=True, silent=True) or {}
    video_b64 = payload.get("video_base64", "")
    if not video_b64:
        return jsonify({"error": "video_base64 missing"}), 400

    sample_fps = float(payload.get("sample_fps", 2))
    max_faces = int(payload.get("max_faces", 50))
    min_face_size = int(payload.get("min_face_size", 60))
    conf_thr = float(payload.get("conf_thr", 0.35))
    iou_dedupe = float(payload.get("iou_dedupe", 0.6))
    brightness_factor = float(payload.get("brightness_factor", 1.25))

    temp_path = None
    cap = None

    try:
        temp_path = b64_to_temp_video(video_b64)
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        step = max(1, int(round(fps / max(sample_fps, 0.1))))

        faces_found = []
        frame_index = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            if frame_index % step != 0:
                continue

            h, w = frame.shape[:2]

            results = model.predict(frame, conf=conf_thr, verbose=False, device=DEVICE)
            if not results or results[0].boxes is None:
                continue

            for b in results[0].boxes:
                conf = float(b.conf[0])
                x1, y1, x2, y2 = clip_box(*b.xyxy[0].tolist(), w, h)

                bw = x2 - x1
                bh = y2 - y1
                if bw < min_face_size or bh < min_face_size:
                    continue

                pad = int(0.12 * max(bw, bh))
                cx1, cy1, cx2, cy2 = clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)

                crop = frame[cy1:cy2, cx1:cx2].copy()
                if crop.size == 0:
                    continue

                # 🔆 Brightness enhancement
                crop = increase_brightness(crop, factor=brightness_factor)

                faces_found.append({
                    "frame_index": frame_index,
                    "score": conf,
                    "bbox": [cx1, cy1, cx2, cy2],
                    "crop": crop,
                })

            if len(faces_found) >= max_faces * 2:
                break

        faces_found = dedupe_faces(faces_found, iou_thr=iou_dedupe)
        faces_found = sorted(faces_found, key=lambda x: x["score"], reverse=True)[:max_faces]

        out_faces = []
        for f in faces_found:
            crop = f["crop"]
            preview = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)

            out_faces.append({
                "frame_index": int(f["frame_index"]),
                "face_crop_b64": jpeg_b64_raw(crop),
                "face_preview_b64": jpeg_b64_raw(preview),
                "score": float(round(f["score"], 5)),
            })

        return jsonify({"faces": out_faces})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9006, debug=False)
