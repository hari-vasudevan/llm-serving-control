import subprocess
import time

import modal


app = modal.App("chapter-9-gpu-batch-plant")

SERVER_PORT = 8019
DEFAULT_DIM = 2048
DEFAULT_LAYERS = 8
DEFAULT_INITIAL_B = 16
DEFAULT_TICK_S = 0.5

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_file(
        "chapter_9/python/gpu_batch_server.py",
        remote_path="/root/gpu_batch_server.py",
    )
    .add_local_file(
        "chapter_9/python/workloads.py",
        remote_path="/root/workloads.py",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    scaledown_window=300,
)
@modal.web_server(port=SERVER_PORT, startup_timeout=180.0)
def serve():
    cmd = [
        "python",
        "/root/gpu_batch_server.py",
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
        "--device",
        "cuda",
        "--initial-B",
        str(DEFAULT_INITIAL_B),
        "--dim",
        str(DEFAULT_DIM),
        "--layers",
        str(DEFAULT_LAYERS),
        "--tick-s",
        str(DEFAULT_TICK_S),
        "--log-dir",
        "/tmp/ch9_logs",
    ]
    print("[modal] launching Chapter 9 GPU batch plant", flush=True)
    subprocess.Popen(cmd)
    time.sleep(10)
