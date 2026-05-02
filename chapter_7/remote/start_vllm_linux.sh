#!/usr/bin/env bash
set -euo pipefail

# Chapter 7: start a Linux/NVIDIA vLLM server plus the wrapper queue server.
#
# Intended for an Ubuntu-like GPU VM. You can also run the pieces manually
# inside a notebook, but a normal VM is a much better fit for the "local Mac
# drives a remote service" workflow than free Colab.

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8001}"
WRAPPER_PORT="${WRAPPER_PORT:-8002}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
API_KEY="${API_KEY:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/ch7-vllm}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/ch7-vllm}"

mkdir -p "$LOG_DIR"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install "vllm>=0.8" requests

VLLM_LOG="$LOG_DIR/vllm.log"
WRAPPER_LOG="$LOG_DIR/wrapper.log"

pkill -f "vllm serve .* --port $VLLM_PORT" 2>/dev/null || true
pkill -f "vllm_queue_server.py --port $WRAPPER_PORT" 2>/dev/null || true

VLLM_CMD=(
  vllm serve "$MODEL"
  --host 0.0.0.0
  --port "$VLLM_PORT"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-model-len "$MAX_MODEL_LEN"
  --generation-config vllm
)

if [ -n "$API_KEY" ]; then
  VLLM_CMD+=(--api-key "$API_KEY")
fi

nohup "${VLLM_CMD[@]}" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" > /tmp/ch7_vllm.pid

echo "[start] vLLM PID=$VLLM_PID"
echo "[start] Waiting for vLLM health on :$VLLM_PORT ..."
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null
echo "[start] vLLM healthy"

nohup python "$ROOT_DIR/vllm_queue_server.py" \
  --port "$WRAPPER_PORT" \
  --backend-url "http://127.0.0.1:${VLLM_PORT}" \
  --model "$MODEL" \
  --api-key "$API_KEY" \
  --B-init 2 \
  --B-min 1 \
  --B-max "$MAX_NUM_SEQS" \
  --dt 1.0 \
  >"$WRAPPER_LOG" 2>&1 &

WRAPPER_PID=$!
echo "$WRAPPER_PID" > /tmp/ch7_wrapper.pid

echo "[start] Wrapper PID=$WRAPPER_PID"
echo "[health] curl http://127.0.0.1:${WRAPPER_PORT}/health"
echo "[metrics] curl http://127.0.0.1:${WRAPPER_PORT}/metrics"
echo "[logs] tail -f $VLLM_LOG"
echo "[logs] tail -f $WRAPPER_LOG"
