# LLM Serving Control: Project State (updated 2026-05-17)

## Repository

```text
path: /Users/hvasudevan/Library/Mobile Documents/com~apple~CloudDocs/git_src_personal/llm-serving-control
branch: chapter-10-experimental-not-for-merge-yet
```

## Chapter Progression

| Chapter | Topic | Status |
|---------|-------|--------|
| 2b-9 | Queue-based PI cascade controller | Complete (clean CUDA batch plant) |
| 10 | vLLM token-budget characterization | Complete, committed, post-mortem written |
| 11 | Phase 1: open-loop budget sweep | Complete (commit `b56e2b4`) |
| 11 | Phase 2: closed-loop TTFT PI controller | Complete (see below) |
| 11 | Phase 3: power controller + disturbance rejection | Not started |

## Chapter 11 Phase 2 Status (2026-05-17)

All code is committed. Phase 2 demonstrated a working closed-loop TTFT PI
controller that:
- Regulates TTFT by throttling `max_num_scheduled_tokens` per schedule() call
- Uses a rolling TTFT window fed back from the wrapper to the scheduler
- Correctly identifies the plant as negative-gain: fraction↑ → TTFT↓

Key implementation in:
- `chapter_11/remote/ch11_vllm/controlled_scheduler.py` — PI + actuator
- `chapter_11/remote/vllm_modal_wrapper.py` — feedback loop + sweep endpoint
- `chapter_11/python/run_ttft_sweep.py` — local runner
- `chapter_11/python/plot_ttft_sweep.py` — SVG plots

## Critical Lessons Learned

### 1. vLLM preemption crash in ttft mode
Never cap `max_num_running_reqs` dynamically in ttft mode. Only cap
`max_num_scheduled_tokens`. Capping running_reqs mid-execution forces
preemption and crashes vLLM at high load.

### 2. Plant gain sign
Token-budget plant has NEGATIVE gain: fraction↑ → TTFT↓.
e_norm = (measured - target) / target (NOT target - measured).
This is opposite to typical control where higher actuator = higher output.

### 3. Actuator authority is load-dependent
At qps=8 with 64-token prompts:
- Natural TTFT ≈ 130ms
- Max achievable TTFT (fraction=0.01 floor) ≈ 270ms
- Targets >280ms are UNACHIEVABLE at this load

At qps=2 with 64-token prompts:
- fraction=0.10 → TTFT=310ms
- fraction=0.05 → TTFT=617ms
(Much more leverage — low arrival rate allows queue buildup)

### 4. Modal container cold-start
Health check must use per-request timeout of `min(remaining, 300s)` to
survive the ~4 minute Qwen2.5-3B cold-start. A 5s or 30s per-request
timeout will drop before Modal proxies the cold-started container.

### 5. Gains at qps=8
kp=0.15 drives fraction from 1.0 to floor in ~3 seconds. This is too
aggressive when the plant response time is also ~3-5 seconds. Use kp=0.05
for operating points where the plant has limited authority.

## Modal Endpoint

```text
URL: https://hvasudevan--chapter-11-token-budget-serve.modal.run
App: chapter-11-token-budget
Model: Qwen/Qwen2.5-3B-Instruct
GPU: T4
Deploy cmd: /Users/hvasudevan/.venvs/modal/bin/modal deploy chapter_11/modal_vllm_wrapper.py
Stop cmd: /Users/hvasudevan/.venvs/modal/bin/modal app stop chapter-11-token-budget --yes
```

## Experimental Results

### Phase 1 (open-loop budget sweep, qps=2, committed)
```
fraction=1.0  → TTFT=103ms, power=64W
fraction=0.10 → TTFT=310ms, power=65W
fraction=0.05 → TTFT=617ms, power=66W
```

### Phase 2 (closed-loop TTFT PI, qps=8)
```
target=300ms: TTFT_mean=285ms, error_rate=0, throughput=8.1 req/s
target=500ms: TTFT_mean=271ms, error_rate=0, actuator saturated at floor
```
