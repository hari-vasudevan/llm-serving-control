import subprocess
import time

import modal


app = modal.App("chapter-7-vllm")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.8", "fastapi", "uvicorn", "requests")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 20,
    scaledown_window=60,
)
@modal.web_server(port=PORT, startup_timeout=240.0)
def serve():
    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--max-num-seqs",
            "8",
            "--max-model-len",
            "2048",
            "--attention-backend",
            "TRITON_ATTN",
            "--generation-config",
            "vllm",
        ],
    )
    # Give vLLM time to start before Modal routes traffic.
    time.sleep(20)
