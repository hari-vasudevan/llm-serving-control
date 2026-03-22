#!/bin/bash
# start_vllm.sh  --  Chapter 5: start vllm-metal serving Qwen3-0.6B
#
# Usage:
#   ./start_vllm.sh              # foreground (Ctrl-C to stop)
#   ./start_vllm.sh --bg         # background, logs to /tmp/vllm_chapter5.log
#
# Port:          8001  (no conflict with Ollama on 11434)
# Model:         mlx-community/Qwen3-0.6B-4bit  (HuggingFace MLX model)
# max-num-seqs:  4    caps concurrent scheduled sequences so arrivals
#                     above 4/tick create real num_requests_waiting > 0
# max-model-len: 512  short context keeps KV cache small on 16 GB M2
#
# After starting, confirm the server is up:
#   curl http://localhost:8001/health
#   curl http://localhost:8001/metrics | grep num_requests

set -e

MODEL="mlx-community/Qwen3-0.6B-4bit"
PORT=8001
MAX_SEQS=4
MAX_LEN=512
LOG=/tmp/vllm_chapter5.log

VENV=~/.venv-vllm-metal
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[start_vllm] ERROR: vllm-metal venv not found at $VENV"
    echo "[start_vllm] Install with: curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash"
    exit 1
fi
source "$VENV/bin/activate"

CMD="vllm serve $MODEL \
    --port $PORT \
    --max-num-seqs $MAX_SEQS \
    --max-model-len $MAX_LEN"

if [[ "$1" == "--bg" ]]; then
    # Kill any existing vLLM on this port first
    if [ -f /tmp/vllm_chapter5.pid ]; then
        OLD_PID=$(cat /tmp/vllm_chapter5.pid)
        kill "$OLD_PID" 2>/dev/null && echo "[start_vllm] Killed old PID $OLD_PID" || true
        rm /tmp/vllm_chapter5.pid
    fi

    echo "[start_vllm] Starting in background, logging to $LOG"
    nohup $CMD > $LOG 2>&1 &
    VLLM_PID=$!
    echo $VLLM_PID > /tmp/vllm_chapter5.pid
    echo "[start_vllm] PID = $VLLM_PID"
    echo "[start_vllm] Model: $MODEL"
    echo "[start_vllm] Waiting for server on port $PORT (model download may take a few minutes on first run)..."

    # Wait up to 3 minutes for server to be ready
    for i in $(seq 1 180); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "[start_vllm] Server ready after ${i}s"
            curl -s http://localhost:$PORT/health
            echo ""
            exit 0
        fi
        sleep 1
        if [ $((i % 15)) -eq 0 ]; then
            echo "[start_vllm] Still waiting... (${i}s) last log line: $(tail -1 $LOG)"
        fi
    done
    echo "[start_vllm] WARNING: server did not respond after 180s"
    echo "[start_vllm] Check log: tail -50 $LOG"
else
    echo "[start_vllm] Starting in foreground on port $PORT"
    echo "[start_vllm] Model: $MODEL"
    echo "[start_vllm] max_num_seqs=$MAX_SEQS  max_model_len=$MAX_LEN"
    echo ""
    exec $CMD
fi
