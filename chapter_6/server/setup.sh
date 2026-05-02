#!/usr/bin/env bash
# =============================================================================
# setup.sh  --  Chapter 6: Intel Mac server setup
#
# IDEMPOTENT: always kills ALL previous instances then restarts from scratch.
# Safe to re-run at any time.
#
# Usage:
#   cd chapter_6/server
#   chmod +x setup.sh && ./setup.sh
#
# After setup:
#   Queue server : http://localhost:8002
#   Live logs    : tail -f /tmp/queue_server.log
#   Stop all     : kill $(cat /tmp/ollama.pid) && kill $(cat /tmp/queue_server.pid)
# =============================================================================

set -e

# ── Config ────────────────────────────────────────────────────────────────
MODEL="qwen2.5:0.5b"          # lightest Qwen in Ollama (~394MB, pure CPU fine)
QUEUE_SERVER_PORT=8002
OLLAMA_PORT=11434
LOG=/tmp/queue_server.log
OLLAMA_LOG=/tmp/ollama.log

# Allow Ollama to process B requests in parallel (not serial)
# This is the key env var that enables real concurrent inference
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1

# Intel Mac puts binaries here
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Chapter 6: Intel Mac LLM Queue Server Setup                 ║"
echo "║  (kills all previous instances, starts fresh)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── STEP 0: Nuclear kill of everything from previous runs ─────────────────
echo "[0/6] Killing all previous instances..."

# Kill queue server by PID file
if [ -f /tmp/queue_server.pid ]; then
    OLD=$(cat /tmp/queue_server.pid)
    kill "$OLD" 2>/dev/null && echo "  Killed queue_server PID $OLD" || true
    rm -f /tmp/queue_server.pid
fi
# Kill any stray queue_server.py processes
pkill -9 -f "queue_server.py"  2>/dev/null && echo "  Killed stray queue_server.py" || true

# Kill Ollama by PID file
if [ -f /tmp/ollama.pid ]; then
    OLD=$(cat /tmp/ollama.pid)
    kill "$OLD" 2>/dev/null && echo "  Killed ollama PID $OLD" || true
    rm -f /tmp/ollama.pid
fi
# Kill any stray ollama processes
pkill -9 -f "ollama serve"     2>/dev/null && echo "  Killed stray ollama serve" || true
pkill -9 -f "ollama runner"    2>/dev/null && echo "  Killed stray ollama runner" || true

# Kill anything on port 8002
lsof -ti :$QUEUE_SERVER_PORT | xargs kill -9 2>/dev/null && echo "  Killed port $QUEUE_SERVER_PORT" || true
# Kill anything on port 11434
lsof -ti :$OLLAMA_PORT        | xargs kill -9 2>/dev/null && echo "  Killed port $OLLAMA_PORT" || true

sleep 2
echo "  Clean slate confirmed."

# ── STEP 1: Homebrew ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    echo "  Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/usr/local/bin/brew shellenv)"
fi
export PATH="/usr/local/bin:$PATH"
echo "  $(brew --version | head -1)"

# ── STEP 2: Python ────────────────────────────────────────────────────────
echo ""
echo "[2/6] Checking Python..."
if ! command -v python3 &>/dev/null; then
    brew install python@3.11
    export PATH="/usr/local/opt/python@3.11/bin:$PATH"
fi
echo "  $(python3 --version)"

# ── STEP 3: Python dependencies ───────────────────────────────────────────
echo ""
echo "[3/6] Python dependencies..."
pip3 install requests numpy --quiet
echo "  requests, numpy OK"

# ── STEP 4: Ollama ────────────────────────────────────────────────────────
echo ""
echo "[4/6] Ollama..."
OLLAMA_BIN="/usr/local/bin/ollama"

if [ ! -x "$OLLAMA_BIN" ]; then
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "  $($OLLAMA_BIN --version 2>/dev/null || echo 'installed')"

# Start Ollama with OLLAMA_NUM_PARALLEL so B requests run concurrently
echo "  Starting Ollama (OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL)..."
OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL \
OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS \
nohup $OLLAMA_BIN serve > $OLLAMA_LOG 2>&1 &
OLLAMA_PID=$!
echo $OLLAMA_PID > /tmp/ollama.pid
echo "  PID=$OLLAMA_PID  log=$OLLAMA_LOG"

for i in $(seq 1 30); do
    if curl -s http://localhost:$OLLAMA_PORT/api/tags &>/dev/null; then
        echo "  Ollama ready after ${i}s"
        break
    fi
    sleep 1
    [ $i -eq 30 ] && echo "  WARNING: Ollama slow, check $OLLAMA_LOG"
done

# ── STEP 5: Pull model ────────────────────────────────────────────────────
echo ""
echo "[5/6] Model: $MODEL"
$OLLAMA_BIN pull $MODEL
echo "  Pull complete."

# Smoke test -- time it so we know baseline latency
echo "  Timing baseline request..."
T0=$(python3 -c "import time; print(int(time.time()*1000))")
RESP=$(curl -s http://localhost:$OLLAMA_PORT/api/generate \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"2+2\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
    --max-time 60 \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','ERR')[:20].strip())" 2>/dev/null \
    || echo "TIMEOUT")
T1=$(python3 -c "import time; print(int(time.time()*1000))")
echo "  Response: '$RESP'  (${T1-T0}ms cold start)"

# ── STEP 6: Queue server ──────────────────────────────────────────────────
echo ""
echo "[6/6] Starting queue server..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nohup python3 -u "$SCRIPT_DIR/queue_server.py" \
    --port $QUEUE_SERVER_PORT \
    --ollama_port $OLLAMA_PORT \
    --model "$MODEL" \
    --B_init 2 \
    --B_max 8 \
    --dt 1.0 \
    > $LOG 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /tmp/queue_server.pid
echo "  PID=$SERVER_PID  log=$LOG"

READY=0
for i in $(seq 1 30); do
    if curl -s http://localhost:$QUEUE_SERVER_PORT/health &>/dev/null; then
        echo "  Queue server ready after ${i}s"
        READY=1; break
    fi
    sleep 1
done
if [ $READY -eq 0 ]; then
    echo "  ERROR: queue server failed. Check:"; tail -20 $LOG; exit 1
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  READY                                                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Model         : %-43s ║\n" "$MODEL"
printf "║  Parallelism   : OLLAMA_NUM_PARALLEL=%-24s ║\n" "$OLLAMA_NUM_PARALLEL"
printf "║  Ollama        : http://localhost:%-27s ║\n" "$OLLAMA_PORT"
printf "║  Queue server  : http://localhost:%-27s ║\n" "$QUEUE_SERVER_PORT"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  curl http://localhost:8002/health   -- health                ║"
echo "║  curl http://localhost:8002/metrics  -- latency + queue       ║"
echo "║  tail -f /tmp/queue_server.log       -- live dispatcher log   ║"
echo "║  tail -f /tmp/ollama.log             -- ollama log            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Your IP (share with controller Mac):                         ║"
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || \
     ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
printf "║    %-58s ║\n" "$IP"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Health:"
curl -s http://localhost:$QUEUE_SERVER_PORT/health | python3 -m json.tool
echo ""
