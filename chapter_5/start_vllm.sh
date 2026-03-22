#!/bin/bash
# start_vllm.sh  --  Chapter 5: start vllm-metal serving Qwen3-0.6B
#
# Usage:
#   ./start_vllm.sh              # foreground (Ctrl-C to stop)
#   ./start_vllm.sh --bg         # background, logs to /tmp/vllm.log
#
# Port:       8001  (avoids conflict with Ollama on 11434 / default 8000)
# Model:      mlx-community/Qwen3-0.6B-4bit
# max_num_seqs: 4   -- caps concurrent scheduled sequences so arrivals
#                      above 4/tick create real num_requests_waiting > 0
# max_model_len: 512 -- short context keeps KV cache small on 16 GB M2
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

source ~/.venv-vllm-metal/bin/activate

CMD="vllm serve $MODEL \
    --port $PORT \
    --max-num-seqs $MAX_SEQS \
    --max-model-len $MAX_LEN \
    --dtype float16 \
    --disable-log-requests"

if [[ "$1" == "--bg" ]]; then
    echo "[start_vllm] Starting in background, logging to $LOG"
    nohup $CMD > $LOG 2>&1 &
    VLLM_PID=$!
    echo "[start_vllm] PID = $VLLM_PID"
    echo $VLLM_PID > /tmp/vllm_chapter5.pid

    # Wait for server to be ready (max 60s)
    echo "[start_vllm] Waiting for server on port $PORT..."
    for i in $(seq 1 60); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "[start_vllm] Server ready after ${i}s"
            break
        fi
        sleep 1
    done
    curl -s http://localhost:$PORT/health && echo ""
else
    echo "[start_vllm] Starting in foreground on port $PORT"
    echo "[start_vllm] Model: $MODEL"
    echo "[start_vllm] max_num_seqs=$MAX_SEQS  max_model_len=$MAX_LEN"
    echo ""
    exec $CMD
fi
