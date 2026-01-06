# Video Face Recognition with Occlusion Detection

## Overview
This repository contains a Proof of Concept (POC) for a **video-based face recognition system with occlusion awareness**, developed based on feedback from a previous internal review. The system is designed to recognize faces from video input while handling real-world challenges such as **masks and sunglasses**.

The solution uses a modular, service-based architecture orchestrated using **n8n**, enabling flexible integration of face detection, occlusion detection, embedding generation, and similarity search.

---

## Key Features
- Video-based face recognition
- Mask and sunglasses occlusion detection
- Occlusion-aware similarity thresholding
- End-to-end automated workflow using n8n
- Consistent embedding generation using FaceNet
- Vector similarity search using PostgreSQL (pgvector)

---

## System Architecture

### 1. Enrollment Workflow
Used to register known users into the system.

**Flow:**
- Images input (Google Drive)
- Face detection and alignment using **MTCNN**
- Face embedding generation using **FaceNet (InceptionResnetV1)**
- Embeddings stored in **PostgreSQL with pgvector**

---

### 2. Recognition Workflow
Used to recognize faces from video input.

**Flow:**
- Video input via webhook
- Frame sampling for efficiency
- Face detection using **YOLOv8n-face**
- Face cropping and preprocessing
- Occlusion detection:
  - Mask detection using pretrained **YOLOv5 face-mask model**
  - Sunglasses detection using pretrained **glasses-detector model**
- Occlusion results merged at face level
- Occlusion-aware branching logic:
  - No occlusion → strict similarity threshold
  - Mask / Sunglasses → relaxed similarity threshold
- Face embedding generation using FaceNet
- Similarity search using pgvector
- Recognition result returned via webhook

---

## Models Used
- **YOLOv8n-face** – Fast face detection from video frames
- **MTCNN** – Face detection and alignment for enrollment
- **FaceNet (InceptionResnetV1, VGGFace2)** – Face embedding generation
- **YOLOv5 Face Mask Model** – Mask detection  
  Source: https://github.com/spacewalk01/yolov5-face-mask-detection
- **Glasses Detector** – Sunglasses detection  
  Source: https://github.com/mantasu/glasses-detector

---

## Occlusion Handling Logic
- Mask and sunglasses detection results are combined into a single **occlusion label**:
  - `None`
  - `Mask`
  - `Sunglasses`
  - `Mask + Sunglasses`
- Recognition thresholds are adapted based on occlusion status to reduce false negatives.

---

## Technology Stack
- **Python**
- **FastAPI / Flask**
- **n8n** (workflow orchestration)
- **PostgreSQL + pgvector**
- **Docker / Podman**
- **PyTorch, OpenCV**


## License
This project is a Proof of Concept intended for internal evaluation and experimentation.
