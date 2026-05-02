# Chapter 6 — Real Queue Server on Intel Mac

## Architecture

```
Intel Mac (server)                    M-series Mac (controller)
─────────────────────                 ──────────────────────────
  ollama serve (:11434)
       ↑
  queue_server.py (:8002)  ←──────── controller/run_controller.py
    - /enqueue                          reads /metrics
    - /metrics                          sends POST /control {"B": N}
    - /control
```

The key difference from Chapter 5: requests **genuinely wait** in the queue
until the dispatcher picks them up. `l_total = t_first_token - t_enqueue`
includes real queue wait time, making the cascade plant model correct:

    l_total(B, q) = alpha*B + gamma*B^2 + (q/B)*dt*1000

## Intel Mac Setup (run once after cloning)

```bash
cd chapter_6/server
chmod +x setup.sh
./setup.sh
```

This installs Homebrew, Python, Ollama, pulls `qwen2.5:0.5b`, and starts
the queue server on port 8002.

## Manual start/stop

```bash
# Start
python3 queue_server.py --port 8002 --model qwen2.5:0.5b --B_init 3

# Check health
curl http://localhost:8002/health

# Check metrics
curl http://localhost:8002/metrics | python3 -m json.tool

# Set batch size
curl -X POST http://localhost:8002/control -d '{"B": 5}'

# Reset metrics between experiments
curl -X POST http://localhost:8002/reset

# Stop
kill $(cat /tmp/queue_server.pid)
```

## Test the queue server locally

```bash
# Fire 5 req/s for 30s and watch queue build up
python3 load_gen.py --rate 5 --duration 30
```

## From the M-series Mac (controller side)

```bash
# Replace with your Intel Mac's local IP
SERVER=http://192.168.x.x:8002

# Health check
curl $SERVER/health

# Metrics
curl $SERVER/metrics

# Enqueue a test prompt
curl -X POST $SERVER/enqueue -d '{"prompt": "What is 2+2?"}'

# Set B (the controller does this each tick)
curl -X POST $SERVER/control -d '{"B": 5}'
```

## Queue server endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /metrics | JSON metrics: q_sw, B, l_total stats |
| GET | /prom_metrics | Prometheus text format |
| GET | /status | Full server state |
| POST | /enqueue | Enqueue a prompt (async, returns immediately) |
| POST | /enqueue_sync | Enqueue and block until complete (returns l_total) |
| POST | /control | Set batch size: `{"B": N}` |
| POST | /reset | Clear queue and reset metrics |

## Key metrics

| Metric | Meaning |
|--------|---------|
| `q_sw` | Current queue depth (requests waiting to be dispatched) |
| `B_current` | Current dispatch batch size |
| `l_total_mean` | Mean l_total over last 200 requests [ms] |
| `l_total_p95` | p95 l_total over last 200 requests [ms] |
| `completed` | Total completed requests since last reset |

## Model choice

`qwen2.5:0.5b` is recommended for the Intel Mac — it's ~400MB and
runs comfortably on CPU. Alternatives if it's too slow:
- `tinyllama:latest` (~600MB, similar speed)
- `qwen2.5:0.5b-instruct` (same size, instruction-tuned)

If CPU is very slow (>5s per request at B=1), use:
- Reduce `num_predict` in `queue_server.py` to 1 (already set)
- Or use a quantised smaller model

## Next steps (controller side)

The controller (`controller/` directory) will:
1. Run `characterise.py` against the Intel Mac server to identify alpha, gamma
2. Run `design_controller.py` to compute cascade gains using beta_q = dt*1000/B0
3. Run `run_controller.py` which reads `/metrics` and sends `/control` each tick
