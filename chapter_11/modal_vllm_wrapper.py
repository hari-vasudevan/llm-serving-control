import subprocess
import time

import modal


app = modal.App("chapter-11-token-budget")

MODEL = "Qwen/Qwen2.5-3B-Instruct"
VLLM_PORT = 8001
WRAPPER_PORT = 8000
MAX_NUM_SEQS = 64
MAX_MODEL_LEN = 2048
DEFAULT_B_MAX = 60
DEFAULT_DT = 1.0
DEFAULT_ADMISSION_FRACTION = 1.0

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.16,<0.17",
        "fastapi",
        "uvicorn",
        "requests",
        "nvidia-ml-py",
        "huggingface_hub[hf_xet]",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONPATH": "/root",
            "CH11_ADMISSION_FRACTION": str(DEFAULT_ADMISSION_FRACTION),
            "CH11_SCHEDULER_ENABLED": "1",
            "CH11_CONTROL_MODE": "open_loop",
            "CH11_CONTROL_FILE": "/tmp/ch11_scheduler_control.json",
        }
    )
    .add_local_file(
        "chapter_11/remote/vllm_modal_wrapper.py",
        remote_path="/root/vllm_modal_wrapper.py",
    )
    .add_local_dir(
        "chapter_11/remote/ch11_vllm",
        remote_path="/root/ch11_vllm",
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
    gpu="T4",
    timeout=60 * 60,
    scaledown_window=300,
    max_containers=1,
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
        "--scheduler-cls",
        "ch11_vllm.controlled_scheduler.ControlledScheduler",
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
        "16",
        "--B-min",
        "1",
        "--B-max",
        str(DEFAULT_B_MAX),
        "--dt",
        str(DEFAULT_DT),
        "--max-tokens",
        "64",
        "--prompt-repeat",
        "256",
        "--trace-prefix",
        "CH11",
    ]

    print(f"[modal] launching vLLM on :{VLLM_PORT} model={MODEL}", flush=True)
    subprocess.Popen(vllm_cmd)
    time.sleep(60)

    print(f"[modal] launching wrapper on :{WRAPPER_PORT}", flush=True)
    subprocess.Popen(wrapper_cmd)
    time.sleep(10)
