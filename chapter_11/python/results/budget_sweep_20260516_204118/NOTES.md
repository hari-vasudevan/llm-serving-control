# Chapter 11 Phase 1 Run Notes

Run timestamp: 2026-05-16 20:41:18 IST

Endpoint:

```text
https://hvasudevan--chapter-11-token-budget-serve.modal.run
```

Configuration:

```text
admission_fractions = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
offered_rate_qps = 2
duration_s = 45
warmup_s = 10
max_tokens = 32
prompt_repeat = 64
metric_period_s = 0.5
```

Why qps=2:

An initial qps=8 attempt overloaded the plant even at full fraction, producing
large vLLM waiting backlog. This run uses lower offered load so the sweep
captures the static token-budget plant curve instead of runaway queue buildup.

Headline observations:

- Fractions 1.0 through 0.25 had similar TTFT means, roughly 103-122 ms.
- TTFT degraded sharply at 0.1 and 0.05.
- GPU mean power stayed near 64-66 W across the sweep.
- Energy/request worsened at 0.05 because throughput dropped while power did
  not fall.
- vLLM queue wait remained near zero down to 0.25, then rose at 0.1 and 0.05.

Key outputs:

```text
sweep_summary.csv
sweep_summary.json
sweep_response.json
logs/run_budget_sweep.log
plots/phase1_dashboard.svg
plots/*.svg
```
