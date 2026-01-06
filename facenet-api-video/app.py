# ---- compatibility patch for old requests/urllib3 expecting collections.MutableMapping ----
import collections
try:
    from collections.abc import MutableMapping
    if not hasattr(collections, "MutableMapping"):
        collections.MutableMapping = MutableMapping
except Exception:
    pass
# ----------------------------------------------------------------------#



import io
import base64
import tempfile
import os
import traceback
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1


# Optional RetinaFace detector
try:
    from retinaface import RetinaFace
    HAVE_RETINA = True
except Exception:
    HAVE_RETINA = False

# Optional YOLOv8-face (Ultralytics)
try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except Exception:
    HAVE_YOLO = False

app = FastAPI()

# CORS for frontend/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MTCNN for face detection/cropping
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, keep_all=True,
              thresholds=[0.5, 0.6, 0.7], factor=0.709, min_face_size=20, device=device)

# FaceNet embedding model
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def pil_to_base64(pil_img, format="JPEG"):
    buf = BytesIO()
    pil_img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_tensor_for_facenet(pil_img, image_size=160):
    """
    Convert PIL image to a torch tensor suitable for InceptionResnetV1:
    - resize to image_size
    - convert to float32 [0..1]
    - normalize to [-1, 1]
    Returns: tensor shape (1, 3, H, W)
    """
    img = pil_img.resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    tensor = (tensor - 0.5) / 0.5
    return tensor


def tensor_from_mtcnn_item(item):
    """
    Accepts either a PIL.Image, a torch.Tensor (3,H,W) or (N,3,H,W) and returns
    a tensor of shape (1,3,H,W) normalized to [-1,1] on the model device.
    """
    if isinstance(item, Image.Image):
        return pil_to_tensor_for_facenet(item).to(device)
    if isinstance(item, torch.Tensor):
        t = item
        if t.dim() == 4:
            # If batch dim present, take first entry
            t = t[0]
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t.to(device)
    raise ValueError("Unsupported face item type for embedding")


def detect_faces_generic(pil_img):
    """Return (boxes, probs, landmarks, crops)
    - boxes: list of [x1,y1,x2,y2]
    - probs: list of scores or None
    - landmarks: list of landmarks arrays or None
    - crops: tensor (N,3,H,W) or None
    Detector selection via env var DETECTOR: 'retina' or 'mtcnn'
    """
    detector_choice = os.environ.get("DETECTOR", "mtcnn").lower()
    if detector_choice == "retina" and HAVE_RETINA:
        # RetinaFace expects a file path; save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            pil_img.save(tmp.name)
            tmp_path = tmp.name

        try:
            res = RetinaFace.detect_faces(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if res is None:
            return None, None, None, None

        boxes = []
        probs = []
        landmarks = []
        for k, v in res.items():
            # v usually contains 'facial_area' and 'score' and 'landmarks'
            area = v.get("facial_area") or v.get("bbox")
            if area is None:
                continue
            x1, y1, x2, y2 = area
            boxes.append([x1, y1, x2, y2])
            probs.append(float(v.get("score", 1.0)))
            lm = v.get("landmarks")
            if lm is not None:
                # landmarks may be dict with keys; convert to array of points
                if isinstance(lm, dict):
                    # try to map common keys
                    points = []
                    for name in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"):
                        if name in lm:
                            points.append(lm[name])
                    landmarks.append(np.array(points))
                else:
                    landmarks.append(np.array(lm))
            else:
                landmarks.append(None)

        return boxes, probs, landmarks, None

    if detector_choice == "yolov8" and HAVE_YOLO:
        # Run YOLO model to get bounding boxes for faces
        # Model path can be provided via YOLO_MODEL env var or default to 'yolov8n-face.pt' if available
        model_path = os.environ.get("YOLO_MODEL", None)
        if model_path is None:
            # try to use builtin 'yolov8n-face' if ultralytics has it; otherwise user must provide
            try:
                model = YOLO('yolov8n-face')
            except Exception:
                return None, None, None, None
        else:
            model = YOLO(model_path)

        # ultralytics accepts numpy array or PIL
        results = model(np.array(pil_img))
        boxes = []
        probs = []
        landmarks = []
        # ultralytics result: results[0].boxes.xyxy, results[0].boxes.conf
        try:
            r = results[0]
            xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.array([])
            conf = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.array([])
            for i, b in enumerate(xyxy):
                x1, y1, x2, y2 = map(float, b)
                boxes.append([x1, y1, x2, y2])
                probs.append(float(conf[i]) if len(conf) > i else 1.0)
                landmarks.append(None)
        except Exception:
            return None, None, None, None

        # Use MTCNN to create aligned crops and landmarks for each detected box
        crops_list = []
        lm_list = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = pil_img.crop((x1, y1, x2, y2)).convert('RGB')
            # ask mtcnn to align this crop
            try:
                aligned = mtcnn(crop)
                if aligned is None:
                    # fallback to resized crop
                    aligned_pil = crop.resize((160, 160))
                    crops_list.append(pil_to_tensor_for_facenet(aligned_pil)[0])
                    lm_list.append(None)
                else:
                    if isinstance(aligned, torch.Tensor) and aligned.dim() == 4:
                        aligned = aligned[0]
                    crops_list.append(aligned)
                    # try to get landmarks from mtcnn.detect on the original crop
                    try:
                        boxes_lm, probs_lm, lms = mtcnn.detect(crop, landmarks=True)
                        lm_list.append(lms[0] if lms is not None and len(lms) > 0 else None)
                    except Exception:
                        lm_list.append(None)
            except Exception:
                try:
                    aligned_pil = crop.resize((160, 160))
                    crops_list.append(pil_to_tensor_for_facenet(aligned_pil)[0])
                    lm_list.append(None)
                except Exception:
                    pass

        if crops_list:
            crops = torch.stack(crops_list, dim=0)
        else:
            crops = None

        return boxes, probs, lm_list, crops

    # Default: use MTCNN
    boxes, probs = mtcnn.detect(pil_img)
    # Also get aligned crops if possible
    crops, probs_crop = mtcnn(pil_img, return_prob=True)
    # landmarks from mtcnn.detect are not returned here except when requested; we can call detect(..., landmarks=True) separately
    try:
        boxes_lm, probs_lm, landmarks = mtcnn.detect(pil_img, landmarks=True)
        if boxes_lm is not None:
            boxes = boxes_lm
    except Exception:
        landmarks = None

    return boxes, probs if probs is not None else probs_crop, landmarks, crops


# -------------------------------
# ORIGINAL ENDPOINT (image → cropped face)
# -------------------------------
@app.post("/preprocess_mtcnn")
async def preprocess_mtcnn(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        rgb = image.convert("RGB")

        # Use MTCNN to produce aligned crops and probabilities
        crops, probs = mtcnn(rgb, return_prob=True)

        if crops is None:
            return {"error": "no_face_detected", "face_crop_b64": None, "face_preview_b64": None}

        # Choose best by probability (or first) and ensure we have a 1xC x H x W tensor
        if isinstance(crops, torch.Tensor):
            # crops is (N,3,H,W)
            best_idx = int(np.argmax(probs)) if probs is not None else 0
            face_tensor = crops[best_idx].unsqueeze(0)
        else:
            # crops can be list of tensors (3,H,W) or (1,3,H,W)
            best_idx = int(np.argmax(probs)) if probs is not None else 0
            face_item = crops[best_idx]
            if isinstance(face_item, torch.Tensor):
                face_tensor = face_item.unsqueeze(0) if face_item.dim() == 3 else face_item
            else:
                # fallback: convert PIL to tensor
                face_tensor = pil_to_tensor_for_facenet(face_item)

        # Create preview PIL (unnormalize)
        face_np = ((face_tensor[0].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        crop_pil = Image.fromarray(face_np)
        crop_b64 = pil_to_base64(crop_pil)

        # Compute embedding for this crop using same preprocessing used for video
        with torch.no_grad():
            face_tensor = face_tensor.to(device)
            emb = model(face_tensor)
            emb = F.normalize(emb, p=2, dim=1)[0].cpu().tolist()

        return {"error": None, "face_crop_b64": crop_b64, "face_preview_b64": crop_b64, "embedding": emb}

    except Exception:
        return {"error": traceback.format_exc()}


# -------------------------------
# ORIGINAL ENDPOINT (cropped face → embedding)
# -------------------------------
@app.post("/embed_cropped")
async def embed_cropped(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
        face_tensor = pil_to_tensor_for_facenet(img).to(device)  # shape (1,3,160,160)

        with torch.no_grad():
            emb_batch = model(face_tensor)
            emb_batch = F.normalize(emb_batch, p=2, dim=1)
            emb = emb_batch[0]

        return {"embedding": emb.cpu().tolist()}

    except Exception:
        return {"error": traceback.format_exc(), "embedding": None}


# ===================================================================================
# FAST VIDEO ENDPOINT (video → up to top_k faces, limited frames, downscaled)
# ===================================================================================
# ===================================================================================
# VIDEO ENDPOINT (UNIFORM FRAME SAMPLING ACROSS FULL VIDEO)
# - Samples up to max_samples frames evenly from start to end.
# - Processes only those frames (downscaled) for speed.
# - Collects faces from all sampled frames, then keeps top_k by score.
# ===================================================================================
# ===================================================================================
# VIDEO ENDPOINT (bounded frames + time-chunk diversity)
# ===================================================================================
@app.post("/preprocess_mtcnn_video")
async def preprocess_mtcnn_video(
    file: UploadFile = File(...),
    detector: str = Query("mtcnn", description="Detector to use: mtcnn|retina|yolov8"),
    top_k: int = Query(20, description="Max number of faces to return"),
    frame_skip: int = Query(3, description="Process every Nth frame"),
    max_seconds: float = Query(15.0, description="Max seconds from start to scan"),
    resize_width: int = Query(480, description="Resize frame width for speed (0 = no resize)"),
    debug: bool = Query(False, description="Return debug info")
):
    """
    Strategy:
    - Read ONLY the first max_seconds of the video (bounded max_frames).
    - For each processed frame (after frame_skip and downscale):
        * run detect_faces_generic (MTCNN/Retina/YOLO based on DETECTOR env).
        * compute a score using prob + sharpness.
        * assign the detection to a time chunk (e.g. every 3 seconds = 1 chunk).
        * keep the BEST face per chunk (so early, middle, late parts all contribute).
    - After the loop:
        * collect all chunk-best faces, sort by score, keep up to top_k.

    This gives you:
    - Speed: because we only scan a bounded part of the video, downscaled.
    - Coverage: because each time chunk has its own "winner" face, you don't
      get just the first person dominating all top_k slots.
    """
    try:
        os.environ["DETECTOR"] = detector.lower()

        # Save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "could_not_open_video", "faces": []}

        # FPS and frame limits
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0  # fallback
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(min(total_frames, fps * max_seconds)) if total_frames > 0 else int(fps * max_seconds)

        # chunk size: e.g. 3 seconds per chunk
        chunk_sec = 3.0
        chunk_len = max(int(fps * chunk_sec), 1)

        # probability threshold and frontalness parameters
        prob_threshold = 0.2
        max_eye_angle_deg = 25.0     # quite relaxed so we don't miss people
        max_nose_offset = 0.30       # fraction of face width

        # map: chunk_index -> best_face_dict
        best_faces_by_chunk = {}
        processed_frames = 0
        raw_detections = 0

        frame_idx = 0
        while True:
            if frame_idx >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % max(frame_skip, 1) != 0:
                frame_idx += 1
                continue

            processed_frames += 1

            # Downscale for speed
            h, w, _ = frame.shape
            if resize_width and w > resize_width:
                scale = resize_width / float(w)
                frame = cv2.resize(frame, (resize_width, int(h * scale)))
                h, w, _ = frame.shape

            # Sharpness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            norm_sharpness = float(min(sharpness / 1000.0, 1.0))

            # PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            boxes, probs_arr, landmarks, crops = detect_faces_generic(pil_img)

            if boxes is None or len(boxes) == 0:
                frame_idx += 1
                continue

            num_faces = len(boxes)
            raw_detections += num_faces

            for i in range(num_faces):
                # probability
                try:
                    prob = float(probs_arr[i]) if probs_arr is not None else 1.0
                except Exception:
                    prob = 1.0

                if prob < prob_threshold:
                    continue

                box = boxes[i]
                landmark = landmarks[i] if landmarks is not None and len(landmarks) > i else None

                # Light frontalness check
                frontal = True
                if landmark is not None and len(landmark) >= 3:
                    left_eye = np.array(landmark[0])
                    right_eye = np.array(landmark[1])
                    nose = np.array(landmark[2])

                    dx = right_eye[0] - left_eye[0]
                    dy = right_eye[1] - left_eye[1]
                    angle = abs(np.degrees(np.arctan2(dy, dx)))
                    if angle > max_eye_angle_deg:
                        frontal = False

                    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
                    face_w = max(box[2] - box[0], 1.0)
                    nose_offset = abs(nose[0] - eye_mid_x) / face_w
                    if nose_offset > max_nose_offset:
                        frontal = False

                if not frontal:
                    continue

                # Crop (aligned if crops tensor available)
                if isinstance(crops, torch.Tensor) and crops.dim() == 4 and crops.size(0) == num_faces:
                    face_tensor = crops[i]
                    face_np = ((face_tensor.cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255.0)
                    face_np = face_np.clip(0, 255).astype(np.uint8)
                    crop_pil = Image.fromarray(face_np)
                else:
                    x1, y1, x2, y2 = map(int, box)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop_pil = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))

                crop_b64 = pil_to_base64(crop_pil)
                score = float(prob) * 0.7 + float(norm_sharpness) * 0.3

                if score < 0.25:
                    continue

                chunk_index = frame_idx // chunk_len
                existing = best_faces_by_chunk.get(chunk_index)
                if existing is None or score > existing["score"]:
                    best_faces_by_chunk[chunk_index] = {
                        "face_crop_b64": crop_b64,
                        "face_preview_b64": crop_b64,
                        "frame_index": int(frame_idx),
                        "prob": float(prob),
                        "sharpness": float(sharpness),
                        "score": float(score),
                    }

            frame_idx += 1

        cap.release()
        try:
            os.remove(video_path)
        except Exception:
            pass

        faces = list(best_faces_by_chunk.values())
        if not faces:
            return {"error": "no_face_detected", "faces": []}

        faces.sort(key=lambda f: f.get("score", 0.0), reverse=True)
        faces = faces[:top_k]

        resp = {"error": None, "faces": faces}
        if debug:
            resp["debug"] = {
                "fps": fps,
                "total_frames": total_frames,
                "max_frames": max_frames,
                "processed_frames": processed_frames,
                "raw_detections": raw_detections,
                "chunks": len(best_faces_by_chunk),
                "frame_skip": frame_skip,
                "resize_width": resize_width,
            }
        return resp

    except Exception:
        return {"error": traceback.format_exc(), "faces": []}



