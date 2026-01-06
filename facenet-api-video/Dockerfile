# ===== Base image =====
FROM python:3.10-slim

# Avoid Python buffering logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ===== System deps (for OpenCV, etc.) =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ===== Python deps =====
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# Copy requirements and install. For optional extras (ultralytics, retinaface) pass
# build args: --build-arg INSTALL_ULTRALYTICS=1 --build-arg INSTALL_RETINAFACE=1
ARG INSTALL_ULTRALYTICS=0
ARG INSTALL_RETINAFACE=0
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Optional installs
RUN if [ "${INSTALL_ULTRALYTICS}" = "1" ] ; then pip install --no-cache-dir ultralytics ; fi
RUN if [ "${INSTALL_RETINAFACE}" = "1" ] ; then pip install --no-cache-dir retina-face ; fi

RUN pip install --upgrade requests urllib3



# Optional: if you want explicit torch/torchvision versions, use e.g.:
# RUN pip install --no-cache-dir \
#     torch==2.3.0+cpu \
#     torchvision==0.18.0+cpu \
#     torchaudio==2.3.0+cpu \
#     --index-url https://download.pytorch.org/whl/cpu

# ===== App code =====
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

EXPOSE 8000

# ===== Start FastAPI app =====
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
