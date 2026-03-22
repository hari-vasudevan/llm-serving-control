#!/bin/bash
# start_vllm.sh  --  Chapter 5: start vllm-metal serving Qwen3-0.6B
#
# Usage:
#   ./start_vllm.sh              # foreground (Ctrl-C to stop)
#   ./start_vllm.sh --bg         # background, logs to /tmp/vllm_chapter5.log
#
# Always kills ALL existing vLLM and load_gen processes before starting,
# preventing stale queues from contaminating the new server.

set -e

MODEL="mlx-community/Qwen3-0.6B-4bit"
PORT=8001
MAX_SEQS=4
MAX_LEN=512
LOG=/tmp/vllm_chapter5.log

VENV=~/.venv-vllm-metal
source "$VENV/bin/activate"

CMD="vllm serve $MODEL --port $PORT --max-num-seqs $MAX_SEQS --max-model-len $MAX_LEN"

# Always kill ALL stale vLLM and load_gen processes first
echo "[start_vllm] Killing any stale vLLM / load_gen processes..."
pkill -9 -f "vllm serve"  2>/dev/null || true
pkill -9 -f "load_gen.py" 2>/dev/null || true
rm -f /tmp/vllm_chapter5.pid /tmp/load_gen.pid
sleep 2

if [[ "$1" == "--bg" ]]; then
    echo "[start_vllm] Starting in background, logging to $LOG"
    nohup $CMD > $LOG 2>&1 &
    VLLM_PID=$!
    echo $VLLM_PID > /tmp/vllm_chapter5.pid
    echo "[start_vllm] PID = $VLLM_PID"
    echo "[start_vllm] Model: $MODEL"
    echo "[start_vllm] Waiting for server on port $PORT..."

    for i in $(seq 1 180); do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "[start_vllm] Server ready after ${i}s"
            curl -s http://localhost:$PORT/health && echo ""
            # Confirm queue is clean
            Q=$(curl -s http://localhost:$PORT/metrics | grep "num_requests_waiting" | grep -v "#" | awk '{print $NF}')
            echo "[start_vllm] num_requests_waiting = $Q  (should be 0.0)"
            exit 0
        fi
        sleep 1
        if [ $((i % 15)) -eq 0 ]; then
            echo "[start_vllm] Still waiting... (${i}s)"
        fi
    done
    echo "[start_vllm] WARNING: server did not respond after 180s"
    echo "[start_vllm] Check: tail -50 $LOG"
else
    echo "[start_vllm] Starting in foreground on port $PORT"
    exec $CMD
fi
