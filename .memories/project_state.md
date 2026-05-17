# LLM Serving Control: Project State (updated 2026-05-17)

## Repository

```text
path: /Users/hvasudevan/Library/Mobile Documents/com~apple~CloudDocs/git_src_personal/llm-serving-control
branch: master  (merged from chapter-10-experimental-not-for-merge-yet 2026-05-17)
```

## Chapter Progression

| Chapter | Topic | Status |
|---------|-------|--------|
| 1–4 | Simulation + Ollama single/cascade loops | Complete |
| 5 | vLLM Apple Silicon (broke — bad metrics) | Complete |
| 6–9 | Queue cascade: Intel Mac, Modal GPU (partial/worked) | Complete |
| 10 | vLLM token-budget characterization | Complete — post-mortem README written |
| 11 Ph1 | Open-loop budget sweep | Complete (`b56e2b4`) |
| 11 Ph2 | Closed-loop TTFT PI, dispatch-delay | Complete (`1279536`, kp=0.05) |
| 11 Ph3 | Load-step rejection, window=10 | Complete (`ef8534c`, kp=0.03 ki=0.002) |
| 11 Ph3b | Load-step rejection, window=3 | Complete (`c1f6320`) |
| 11 Ph4 | Power controller (constant GPU power) | Not started |

## READMEs (2026-05-17)

All chapters now have READMEs with step-by-step run instructions:
- Ch 1–5: created from scratch
- Ch 6, 8, 9: expanded with prerequisites + run sections
- Ch 10: rewritten as post-mortem with "Reproducing" section
- Ch 11: Quickstart added (5 commands: venv → deploy → run → video → MATLAB)
- Root: "Getting Started" section with prerequisites table + Modal venv setup

## Modal Endpoint (Ch 11)

```text
URL:    https://hvasudevan--chapter-11-token-budget-serve.modal.run
App:    chapter-11-token-budget
Model:  Qwen/Qwen2.5-3B-Instruct on NVIDIA T4
Deploy: source .modal-venv/bin/activate && modal deploy chapter_11/modal_vllm_wrapper.py
Stop:   modal app stop chapter-11-token-budget --yes
```

## Critical Lessons Learned

1. **Plant gain sign**: Token-budget: NEGATIVE (fraction↑ → TTFT↓). Dispatch-delay: POSITIVE.
2. **vLLM preemption crash**: Never cap `max_num_running_reqs` in ttft mode — crashes vLLM.
3. **Token-budget authority**: At qps=8, fraction floor → max TTFT ≈ 170ms. Use dispatch-delay for wide range.
4. **Dispatch-delay t_send**: Must be captured BEFORE `time.sleep(delay_s)` or delay is invisible.
5. **Effective feedback lag**: τ_total = window/(2×qps) + (window-1)/2×(1/qps). At qps=4, window=10: 2.375s; window=3: 0.625s.
6. **Phase margin**: PM = 180° - 90° - kp × feedback_rate × τ_total × (180/π). kp=0.03, window=3: PM=80°.
7. **Modal proxy timeout**: ~15 min. Chained 3-target run ≈ 9.7 min — send all targets in one HTTP POST.
8. **GPU utilization floor**: T4 always ~95-100% at qps=4–8 with Qwen2.5-3B (6GB weights, HBM-bound).
9. **Video uid**: Use enumeration index, not `sent_at_s` — ms collisions at qps=8.
10. **Answer timing in video**: Fire at `sent_at_s + ttft_ms/1000` (first token), NOT `recv_at_s` (full completion).
11. **Window=3 vs window=10**: window=3 drops τ_total to 0.625s, making QPS-step transients visible in TTFT (std 39→99ms at 200ms/qps=8).

## Experimental Results

### Phase 1 (open-loop budget sweep, qps=2)
```
fraction=1.0  → TTFT=103ms, power=64W
fraction=0.10 → TTFT=310ms (transient artifact, not steady-state), power=65W
fraction=0.05 → TTFT=617ms, power=66W
GPU power flat across all fractions — memory-bandwidth floor
```

### Phase 2 (dispatch-delay PI, qps=8, kp=0.05, ki=0.005, window=10)
```
target=200ms: mean=203ms, p95=242ms
target=350ms: mean=349ms, p95=393ms
target=500ms: mean=498ms, p95=545ms
```

### Phase 3 (load-step, kp=0.03, ki=0.002, window=10)
Run: `python/results/load_step_20260517_212627`

| Target | QPS | Mean    | p95     | Std    |
|--------|-----|---------|---------|--------|
| 200ms  | 4   | 198.4ms | 268.5ms | 41.4ms |
| 200ms  | 8   | 200.6ms | 220.2ms | 12.0ms |
| 200ms  | 4   | 200.9ms | 230.8ms | 16.9ms |
| 350ms  | 4   | 351.8ms | 437.3ms | 49.6ms |
| 350ms  | 8   | 348.9ms | 364.8ms | 10.7ms |
| 500ms  | 8   | 499.9ms | 520.4ms | 13.3ms |

### Phase 3b (load-step, kp=0.03, ki=0.002, window=3)
Run: `python/results/load_step_20260517_223115`

| Target | QPS | Mean    | p95     | Std     | Note |
|--------|-----|---------|---------|---------|------|
| 200ms  | 4   | 198.6ms | 247.4ms | 39.2ms  | |
| 200ms  | 8   | 210.6ms | 235.4ms | 99.7ms  | QPS transient visible |
| 200ms  | 4   | 202.7ms | 276.6ms | 41.6ms  | Re-settled |
| 350ms  | 8   | 351.3ms | 392.7ms | 28.7ms  | |
| 500ms  | 8   | 503.2ms | 592.9ms | 43.6ms  | |

All means within 5ms of setpoint.
