from fastapi import FastAPI, UploadFile, File, Form, Header
import numpy as np
import cv2
from glasses_detector import GlassesClassifier
from typing import Optional

app = FastAPI(title="Glasses Detection Service")

# Explicit CPU-only classifier
classifier = GlassesClassifier(
    kind="sunglasses",
    size="small",
    device="cpu",
)

@app.post("/detect-glasses")
async def detect_glasses(
    file: UploadFile = File(...),
    face_id: Optional[str] = Form(None),         
    x_face_id: Optional[str] = Header(None),       
):
    # Prefer form face_id, fallback to header
    face_id = face_id or x_face_id

    image_bytes = await file.read()
    img_array = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        return {
            "face_id": face_id,
            "occlusion": "none",
            "has_sunglasses": False,
            "confidence": 0.0,
        }

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    result = classifier.predict(frame_rgb)

    has_sunglasses = (result == "present")

    return {
        "face_id": face_id,                        
        "occlusion": "sunglasses" if has_sunglasses else "none",
        "has_sunglasses": has_sunglasses,
        "confidence": 1.0 if has_sunglasses else 0.0,
    }

@app.get("/health")
def health():
    return {"status": "ok"}
