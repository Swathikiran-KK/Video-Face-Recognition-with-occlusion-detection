FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheels
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Install all other deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App + model
COPY app.py .
COPY mask_yolov5.pt .

EXPOSE 9002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9002"]
