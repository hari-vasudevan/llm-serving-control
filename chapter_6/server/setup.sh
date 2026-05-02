#!/usr/bin/env bash
# =============================================================================
# setup.sh  --  Chapter 6: Intel Mac server setup
#
# Run this once on the Intel Mac after cloning the repo.
# Sets up Ollama, pulls the model, installs Python deps, and
# starts the queue server in the background.
#
# Usage:
#   cd chapter_6/server
#   chmod +x setup.sh
#   ./setup.sh
#
# After setup, the queue server runs on port 8002.
# Check status:  curl http://localhost:8002/health
# View metrics:  curl http://localhost:8002/metrics
# Stop server:   kill $(cat /tmp/queue_server.pid)
# =============================================================================

set -e

MODEL="qwen2.5:0.5b"          # ~400MB, fast on old Intel CPU
QUEUE_SERVER_PORT=8002
OLLAMA_PORT=11434
LOG=/tmp/queue_server.log

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Chapter 6: Intel Mac LLM Queue Server Setup                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# 1. Homebrew
# ---------------------------------------------------------------------------
echo "[1/6] Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    echo "  Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for Intel Mac
    eval "$(/usr/local/bin/brew shellenv)"
else
    echo "  Homebrew found: $(brew --version | head -1)"
fi

# ---------------------------------------------------------------------------
# 2. Python 3.10+
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "  Installing Python via Homebrew..."
    brew install python@3.11
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "  $PYTHON_VERSION"

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing Python dependencies..."
pip3 install --upgrade pip --quiet
pip3 install flask requests numpy --quiet
echo "  flask, requests, numpy installed"

# ---------------------------------------------------------------------------
# 4. Ollama
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "  Downloading Ollama for macOS (Intel)..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "  Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
fi

# Start Ollama in background if not already running
if ! curl -s http://localhost:$OLLAMA_PORT/api/tags &>/dev/null; then
    echo "  Starting Ollama service..."
    ollama serve > /tmp/ollama.log 2>&1 &
    echo $! > /tmp/ollama.pid
    sleep 3
    echo "  Ollama started (PID=$(cat /tmp/ollama.pid))"
else
    echo "  Ollama already running on port $OLLAMA_PORT"
fi

# ---------------------------------------------------------------------------
# 5. Pull model
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Pulling model: $MODEL"
echo "  (This may take a few minutes on first run...)"
ollama pull $MODEL
echo "  Model ready: $MODEL"

# Quick smoke test
echo "  Smoke test..."
RESP=$(curl -s http://localhost:$OLLAMA_PORT/api/generate \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"2+2\",\"stream\":false}" \
    --max-time 30 | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','ERROR')[:20])" 2>/dev/null || echo "TIMEOUT")
echo "  Model response: '$RESP'"

# ---------------------------------------------------------------------------
# 6. Start queue server
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Starting queue server on port $QUEUE_SERVER_PORT..."

# Kill any existing instance
if [ -f /tmp/queue_server.pid ]; then
    kill $(cat /tmp/queue_server.pid) 2>/dev/null || true
    rm /tmp/queue_server.pid
    sleep 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -u "$SCRIPT_DIR/queue_server.py" \
    --port $QUEUE_SERVER_PORT \
    --ollama_port $OLLAMA_PORT \
    --model "$MODEL" \
    > $LOG 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /tmp/queue_server.pid
echo "  PID=$SERVER_PID  log=$LOG"

# Wait for it to be ready
echo "  Waiting for queue server..."
for i in $(seq 1 30); do
    if curl -s http://localhost:$QUEUE_SERVER_PORT/health &>/dev/null; then
        echo "  Ready after ${i}s"
        break
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SETUP COMPLETE                                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Model:        %-46s ║\n" "$MODEL"
printf "║  Ollama:       http://localhost:%-30s ║\n" "$OLLAMA_PORT"
printf "║  Queue server: http://localhost:%-30s ║\n" "$QUEUE_SERVER_PORT"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  curl http://localhost:8002/health    -- health check         ║"
echo "║  curl http://localhost:8002/metrics   -- queue metrics        ║"
echo "║  curl http://localhost:8002/status    -- full status          ║"
echo "║  kill \$(cat /tmp/queue_server.pid)    -- stop server          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Show current metrics
echo "Current metrics:"
curl -s http://localhost:$QUEUE_SERVER_PORT/metrics | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for k, v in d.items():
        print(f'  {k:30s} = {v}')
except:
    print('  (server starting up)')
"
echo ""
