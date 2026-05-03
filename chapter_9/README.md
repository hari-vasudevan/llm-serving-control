# Chapter 9 -- Lower-Level GPU Batching Plant

Chapter 9 is the direct follow-up to the Chapter 8 failure mode.

Chapter 8 showed that a top-level Modal + vLLM serving experiment, even with
a wrapper FIFO queue, does not cleanly expose the Chapter 2 cascade plant.  It
produced a weak positive `B -> q` relation but an unphysical negative
`q -> L_mean` fit.  Chapter 9 therefore moves one level down: away from HTTP
LLM serving and toward a minimal GPU scheduling/batching plant.

The goal is to test the Chapter 2 cascade theory on the simplest real plant
where its assumptions are physically meaningful.

## Chapter 2 Terminology

The experiment keeps the Chapter 2 controller vocabulary:

- inner loop: `B -> q`
- outer loop: `q_ref -> L_mean`
- actuator: `B[k]`, the commanded batch size
- plant state: `q[k]`, the FIFO queue depth
- outer-loop output: `L_mean[k]`, measured request latency
- optional output: `L_p95[k]`, rolling measured p95 latency
- arrivals: `lambda[k]`, sampled arrivals per control tick

The Chapter 2 plant equations were:

```text
q[k+1] = max(0, q[k] + lambda[k] - B[k])

L_mean[k] = alpha*B[k] + gamma*B[k]^2 + beta*q[k]

L_p95[k] = L_mean[k] + 1.645*delta/sqrt(B[k])
```

Chapter 9 identifies the corresponding quantities from a real GPU batch
workload instead of assuming them.

## Architecture

```text
MATLAB cascade controller
    -> Modal web endpoint
        -> Python plant server on Modal T4
        -> FIFO queue q[k]
        -> exact batch-size actuator B[k]
        -> fixed GPU batch workload
        -> measured batch service time
        -> request/batch/tick logs
```

The Python plant server is intentionally lower-level than vLLM.  A request is
a fixed tensor job.  A batch is exactly `B[k]` queued tensor jobs stacked
together and sent through a fixed PyTorch GPU workload.

The scheduler is deliberately ticked, not eager:

```text
once per DT:
    pop up to B[k] jobs from FIFO
    run one GPU batch
    leave the remainder in q[k]
```

This is important.  If the worker drains continuously, `B` becomes only a
micro-batch cap and the Chapter 2 queue state never appears.

This removes:

- network latency inside the measured service time,
- streaming effects,
- vLLM scheduler opacity,
- serverless/runtime interference,
- prompt and generation-length variability.

## Files

- `python/gpu_batch_server.py`
  HTTP plant server with FIFO queue, exact `B`, measured GPU batch service,
  and CSV logs.
- `python/workloads.py`
  Fixed PyTorch batched matrix workload.
- `modal_gpu_batch_server.py`
  Modal deployment entrypoint for the real GPU experiment.
- `python/requirements.txt`
  Minimal dependencies for optional local CPU smoke tests.
- `matlab/characterise_plant.m`
  Asks Modal to run in-container open-loop sweeps to identify `B -> q`,
  service time, and `q -> L_mean`.
- `matlab/design_controller.m`
  Chapter 2-style cascade controller design.
- `matlab/run_cascade_controller.m`
  Closed-loop test with steady load and arrival spikes.

## Run Flow

The real Chapter 9 GPU experiment runs on Modal, not on the local Mac.

From the repository root:

```bash
python3 -m venv .modal-venv
source .modal-venv/bin/activate
pip install modal
modal setup
modal deploy chapter_9/modal_gpu_batch_server.py
```

Modal prints a web URL.  Put that URL in `CH9_SERVER` before running MATLAB.
For example:

```matlab
setenv('CH9_SERVER', 'https://YOUR-MODAL-URL.modal.run')
```

Then in MATLAB:

```matlab
cd chapter_9/matlab
characterise_plant
design_controller
run_cascade_controller
```

`characterise_plant` calls:

```text
POST /characterise
```

so arrival generation runs inside the Modal GPU container.  MATLAB receives
the summarized results and does the fitting/plotting locally.  This avoids
using MATLAB-to-Modal round trips as the load generator.

`design_controller` writes `controller_config.xml` locally and uploads it to
the Modal plant via:

```text
POST /controller_config
```

The Modal container therefore picks up the locally computed Chapter 2
coefficients without needing access to the local filesystem.

For a local CPU smoke test only:

```bash
cd chapter_9/python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python gpu_batch_server.py --device cpu --port 8019 --initial-B 8 --dim 256 --layers 2
```

Then explicitly point MATLAB at the local smoke-test server:

```matlab
setenv('CH9_SERVER', 'http://127.0.0.1:8019')
```

## Is Python Fast Enough?

Yes, for this experiment.  Python runs inside the Modal GPU container as the
discrete scheduler and logging wrapper; the GPU compute kernel is PyTorch on
CUDA.

The measured service interval is:

```python
workload.synchronize()
t0 = time.perf_counter()
workload.run(B_actual)
workload.synchronize()
t1 = time.perf_counter()
service_time_ms = 1000 * (t1 - t0)
```

That makes the measurement include GPU work and exclude queued asynchronous
launches from previous batches.

Python is appropriate for controller/sample periods around `50--500 ms`.  It
is not intended for microsecond kernel scheduling.  Chapter 2 is a discrete
scheduling controller, so this is the right timescale for the theory.

## Success Criteria

Chapter 9 succeeds if the identified plant has physically credible signs:

```text
B increase -> queue drain improves near overload
q increase -> L_mean increases
service_time changes smoothly with B
closed-loop B moves q toward q_ref
outer loop moves q_ref to regulate L_mean
```

The most important check is:

```text
q -> L_mean must be positive
```

If that relation is not positive in this lower-level setup, the Chapter 2
cascade theory is not matching the real scheduling plant.  If it is positive
here but not in Chapter 8, then Chapter 8 failed because the top-level serving
stack hid the plant, not because the cascade idea was wrong.

## Operating Point Selection

The useful actuator range stops when larger `B` no longer increases
completions per tick.  The MATLAB characterisation therefore computes:

```text
B_max_effective = first B with completions >= 98% of max(completions)
```

For the current Modal T4 workload this has been around `B = 2400`; `B = 3200`
does not materially improve drain rate and should not be used as the controller
upper limit.  The controller still uses the Chapter 2 sign convention:

```text
dq[k+1] = dq[k] - beta_q*dB[k], beta_q > 0
```
