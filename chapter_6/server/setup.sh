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

# Intel Mac Ollama installs to /usr/local/bin -- ensure it's on PATH
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"

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
    eval "$(/usr/local/bin/brew shellenv)"
else
    echo "  Homebrew found: $(brew --version | head -1)"
fi
# Intel Mac brew is at /usr/local
export PATH="/usr/local/bin:$PATH"

# ---------------------------------------------------------------------------
# 2. Python 3.10+
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "  Installing Python via Homebrew..."
    brew install python@3.11
    export PATH="/usr/local/opt/python@3.11/bin:$PATH"
fi
PYTHON_VERSION=$(python3 --version 2>&1)
echo "  $PYTHON_VERSION"

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing Python dependencies..."
pip3 install --upgrade pip --quiet 2>/dev/null || pip3 install --upgrade pip
pip3 install requests numpy --quiet
echo "  requests, numpy installed"

# ---------------------------------------------------------------------------
# 4. Ollama
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Checking Ollama..."

# Ollama on Intel Mac installs the CLI to /usr/local/bin/ollama
# It does NOT install a macOS .app bundle -- it's a CLI-only tool on Linux/Mac
OLLAMA_BIN="/usr/local/bin/ollama"

if [ ! -f "$OLLAMA_BIN" ]; then
    echo "  Downloading and installing Ollama CLI..."
    # The official install script puts the binary at /usr/local/bin/ollama
    curl -fsSL https://ollama.com/install.sh | sh
    echo "  Ollama installed at $OLLAMA_BIN"
else
    echo "  Ollama found at $OLLAMA_BIN"
fi

# Verify the binary is executable
if [ ! -x "$OLLAMA_BIN" ]; then
    echo "  ERROR: $OLLAMA_BIN exists but is not executable"
    echo "  Try: chmod +x $OLLAMA_BIN"
    exit 1
fi

echo "  Version: $($OLLAMA_BIN --version 2>/dev/null || echo 'ok')"

# Start Ollama serve in background if not already running
if curl -s http://localhost:$OLLAMA_PORT/api/tags &>/dev/null; then
    echo "  Ollama already running on port $OLLAMA_PORT"
else
    echo "  Starting Ollama server..."
    # Kill any stale ollama processes first
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 1
    nohup $OLLAMA_BIN serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo $OLLAMA_PID > /tmp/ollama.pid
    echo "  Waiting for Ollama to be ready (PID=$OLLAMA_PID)..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:$OLLAMA_PORT/api/tags &>/dev/null; then
            echo "  Ollama ready after ${i}s"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo "  WARNING: Ollama did not respond after 30s"
            echo "  Check: tail -20 /tmp/ollama.log"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 5. Pull model
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Pulling model: $MODEL"
echo "  (This may take several minutes on first run -- ~400MB download)"
$OLLAMA_BIN pull $MODEL
echo "  Model ready: $MODEL"

# Quick smoke test
echo "  Smoke test..."
RESP=$(curl -s http://localhost:$OLLAMA_PORT/api/generate \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"2+2\",\"stream\":false}" \
    --max-time 60 \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','ERROR')[:30].strip())" 2>/dev/null \
    || echo "TIMEOUT/ERROR")
echo "  Response preview: '$RESP'"
if [ "$RESP" = "TIMEOUT/ERROR" ]; then
    echo "  WARNING: smoke test failed -- model may be slow on first load, continuing anyway"
fi

# ---------------------------------------------------------------------------
# 6. Start queue server
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Starting queue server on port $QUEUE_SERVER_PORT..."

# Kill any existing instance
if [ -f /tmp/queue_server.pid ]; then
    OLD_PID=$(cat /tmp/queue_server.pid)
    kill "$OLD_PID" 2>/dev/null || true
    rm -f /tmp/queue_server.pid
    sleep 1
fi
pkill -f "queue_server.py" 2>/dev/null || true
sleep 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nohup python3 -u "$SCRIPT_DIR/queue_server.py" \
    --port $QUEUE_SERVER_PORT \
    --ollama_port $OLLAMA_PORT \
    --model "$MODEL" \
    > $LOG 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > /tmp/queue_server.pid
echo "  PID=$SERVER_PID  log=$LOG"

# Wait for it to be ready
echo "  Waiting for queue server..."
READY=0
for i in $(seq 1 30); do
    if curl -s http://localhost:$QUEUE_SERVER_PORT/health &>/dev/null; then
        echo "  Ready after ${i}s"
        READY=1
        break
    fi
    sleep 1
done

if [ $READY -eq 0 ]; then
    echo "  ERROR: queue server did not start. Check log:"
    tail -20 $LOG
    exit 1
fi

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
echo "║  Useful commands:                                             ║"
echo "║  curl http://localhost:8002/health   -- health check          ║"
echo "║  curl http://localhost:8002/metrics  -- queue depth + latency ║"
echo "║  curl http://localhost:8002/status   -- full state            ║"
echo "║  tail -f /tmp/queue_server.log       -- live dispatcher log   ║"
echo "║  tail -f /tmp/ollama.log             -- ollama log            ║"
echo "║  kill \$(cat /tmp/queue_server.pid)   -- stop queue server     ║"
echo "║  kill \$(cat /tmp/ollama.pid)         -- stop ollama           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  To find your local IP for the controller Mac:                ║"
echo "║  ipconfig getifaddr en0                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Show current health + metrics
echo "Health check:"
curl -s http://localhost:$QUEUE_SERVER_PORT/health | python3 -m json.tool 2>/dev/null || \
    curl -s http://localhost:$QUEUE_SERVER_PORT/health
echo ""
echo "Your local IP address (share this with the controller Mac):"
ipconfig getifaddr en0 2>/dev/null || \
    ipconfig getifaddr en1 2>/dev/null || \
    ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1
echo ""
