# ============================================================
# Mask Detection API (FINAL – FACE_ID SAFE)
# ============================================================

import io
import numpy as np
import torch
from PIL import Image
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# App
# ============================================================

app = FastAPI(title="Mask Detection Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

torch.set_num_threads(1)

# ============================================================
# Load YOLOv5 MASK model
# ============================================================

print("Loading YOLOv5 face-mask model...")
mask_model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="mask_yolov5.pt",
    trust_repo=True,
    force_reload=False,
)
mask_model.eval()

MASK_CLASSES = mask_model.names
print("Mask classes:", MASK_CLASSES)

# ============================================================
# API
# ============================================================

@app.post("/detect-mask")
async def detect_mask(
    file: UploadFile = File(...),
    face_id: Optional[str] = Form(None),           
    x_face_id: Optional[str] = Header(None),       
):
    """
    Receives a face crop image and returns mask detection result.
    Behavior identical to old API, plus face_id passthrough.
    """

    # Prefer form face_id, fallback to header
    face_id = face_id or x_face_id

    # -------------------------
    # Load image (same as before)
    # -------------------------
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_np = np.array(img)

    has_mask = False
    confidence = 0.0

    # -------------------------
    # YOLOv5 MASK detection
    # -------------------------
    results = mask_model(img_np, size=224)
    dets = results.xyxy[0]

    if dets is not None and len(dets) > 0:
        det = dets[dets[:, 4].argmax()]
        conf = float(det[4])
        cls_name = MASK_CLASSES.get(int(det[5]), "unknown")

        if cls_name == "with_mask" and conf >= 0.5:
            has_mask = True
            confidence = round(conf, 4)

    # -------------------------
    # Response (FACE_ID SAFE)
    # -------------------------
    if has_mask:
        return {
            "face_id": face_id,                   
            "occlusion": "mask",
            "has_mask": True,
            "mask_type": "with_mask",
            "confidence": confidence,
        }

    return {
        "face_id": face_id,                       
        "occlusion": "none",
        "has_mask": False,
        "mask_type": "none",
        "confidence": 0.0,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
