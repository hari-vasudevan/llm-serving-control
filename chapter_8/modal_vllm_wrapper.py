import subprocess
import time

import modal


app = modal.App("chapter-8-vllm-wrapper")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_PORT = 8001
WRAPPER_PORT = 8000
MAX_NUM_SEQS = 64
MAX_MODEL_LEN = 4096
DEFAULT_B_MAX = 50
DEFAULT_DT = 1.0

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8",
        "fastapi",
        "uvicorn",
        "requests",
        "huggingface_hub[hf_xet]",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_file(
        "chapter_8/remote/vllm_modal_wrapper.py",
        remote_path="/root/vllm_modal_wrapper.py",
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    gpu="T4",
    timeout=60 * 60,
    scaledown_window=300,
)
@modal.web_server(port=WRAPPER_PORT, startup_timeout=480.0)
def serve():
    vllm_cmd = [
        "vllm",
        "serve",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--attention-backend",
        "TRITON_ATTN",
        "--generation-config",
        "vllm",
    ]
    wrapper_cmd = [
        "python",
        "/root/vllm_modal_wrapper.py",
        "--host",
        "0.0.0.0",
        "--port",
        str(WRAPPER_PORT),
        "--backend-url",
        f"http://127.0.0.1:{VLLM_PORT}",
        "--model",
        MODEL,
        "--B-init",
        "4",
        "--B-min",
        "1",
        "--B-max",
        str(DEFAULT_B_MAX),
        "--dt",
        str(DEFAULT_DT),
        "--max-tokens",
        "32",
        "--prompt-repeat",
        "192",
        "--trace-prefix",
        "CH8",
    ]

    print(f"[modal] launching vLLM on :{VLLM_PORT} model={MODEL}", flush=True)
    subprocess.Popen(vllm_cmd)
    time.sleep(35)

    print(f"[modal] launching wrapper on :{WRAPPER_PORT}", flush=True)
    subprocess.Popen(wrapper_cmd)
    time.sleep(10)
