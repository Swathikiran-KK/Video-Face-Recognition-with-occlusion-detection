# Glasses Service — Quick Start

1. Create a folder named `glass_service`.

2. In that folder add these files:
   - `app.py`
   - `Dockerfile`
   

3. Build the container:

    ```powershell
    podman build -t glasses-service .
    ```

4. Run the container:

    ```powershell
    podman run -d --name glasses-service -p 9003:9003 glasses-service
    ```

5. If it crashed, check the logs:

    ```powershell
    podman logs glasses-service
    ```

6. Stop and remove (optional):

    ```powershell
    podman stop glasses-service
    podman rm glasses-service
    ```

That's it.
