# Mask Service — Quick Start

1. Create a folder named `mask_service`.

2. In that folder add these files:
   - `app.py`
   - `Dockerfile`
   - `mask_yolov5.pt`
   - `requirements.txt`
   

3. Build the container:

```powershell
podman build -t mask-service .
```

4. Run the container:

```powershell
podman run -d --name mask-service -p 9002:9002 mask-service
```

5. If it crashed, check the logs:

```powershell
podman logs mask-service
```

6. Stop and remove (optional):

```powershell
podman stop mask-service
podman rm mask-service
```

That's it.
