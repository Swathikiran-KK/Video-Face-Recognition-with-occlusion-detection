FROM python:3.12-slim

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Python base deps
# -------------------------
RUN pip install --no-cache-dir --upgrade pip

# IMPORTANT: NumPy < 2
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    fastapi \
    uvicorn \
    pillow \
    opencv-python-headless \
    tqdm \
    python-multipart

# -------------------------
# PyTorch CPU (Python 3.12 compatible)
# -------------------------
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# -------------------------
# Glasses detector
# -------------------------
RUN pip install --no-cache-dir glasses-detector==1.0.4

# -------------------------
# App
# -------------------------
WORKDIR /app
COPY app.py /app/app.py

EXPOSE 9003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9003"]
