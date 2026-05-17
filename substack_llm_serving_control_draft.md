# Finding the plant in LLM serving control

Subtitle:
The project started as a neat cascade-control idea. It became a longer lesson in measurement, wrong abstractions, and the suspiciously slippery meaning of a queue.

This project began with a very optimistic sentence:

An LLM serving system is a queueing system.

That sentence is not wrong. It is also not nearly specific enough to design a controller.

In a queueing sketch, requests arrive, wait, get batched, run on compute, and eventually produce tokens. So the first control idea was natural:

If the queue grows, increase batch size.

If latency grows, adjust the queue target.

That gives a cascade controller. The inner loop uses batch size to regulate queue depth. The outer loop moves the queue reference to regulate latency. Very tidy. The kind of thing that looks good on a whiteboard, which is also where many dangerous ideas dress well.

The real project was not "we wrote a controller and it worked." That would be a much shorter post, and also a less honest one.

The real project was a sequence of attempts to find the layer in the LLM serving stack where the controller variables meant something physical.

The surprising part was not that the first model was simple. Simple models are useful. The surprising part was how many times a variable called `q` was not the queue state the controller needed, or a variable called `B` was not the actuator the controller thought it was.

This post is a summary of that investigation. It is intentionally more of a learning log than a tutorial. If there is a theme, it is this:

Characterization first. Control second. Do not skip the awkward middle where your model quietly laughs at you.

Repo: https://github.com/hari-vasudevan/llm-serving-control

Chapter map: https://github.com/hari-vasudevan/llm-serving-control/blob/main/README.md

## Chapter 1 and Chapter 2: the clean theory

The first two chapters were the clean control-theory base.

Chapter 1:
https://github.com/hari-vasudevan/llm-serving-control/tree/main/chapter_1

Chapter 2:
https://github.com/hari-vasudevan/llm-serving-control/tree/main/chapter_2

I started with a discrete-time plant:

```text
q[k+1] = max(0, q[k] + lambda[k] - B[k])
```

Here:

- `lambda[k]` is the number of arrivals in a scheduling tick
- `B[k]` is the commanded batch size or service capacity
- `q[k]` is the queue depth

The first latency model was:

```text
L_mean[k] = alpha*B[k] + gamma*B[k]^2 + beta*q[k]
```

This says two things. First, larger batches may cost more service time. Second, larger queues add waiting time. It is a compact model, not a universal law of nature. Nobody should carve `gamma*B^2` into stone tablets. I only have a MacBook, not a mountain.

In Chapter 1, the controller was an integral state-feedback regulator on this simulated plant. In Chapter 2, I split the same idea into a cascade:

- inner loop: `B -> q`
- outer loop: `q_ref -> L_mean` / `L_p95`

The simulation behaved well. When arrivals stepped up, the controller moved `B`. Queue depth came back. Latency followed the modeled plant. This was useful because it proved the basic control structure was internally consistent.

[Insert Figure 1: Chapter 2 simulated cascade result]

Caption: The Chapter 1/2 simulation was the clean starting point. When `B` is a real actuator, `q` is a real state, and latency follows the assumed plant, the cascade behaves as expected.

But simulation also gave me a false sense of vocabulary. I had names for the variables. I did not yet know where those variables lived in a real serving stack.

That difference became the entire project.

## Chapter 3: trying the cascade on Ollama, and discovering an empty queue

Chapter 3:
https://github.com/hari-vasudevan/llm-serving-control/tree/main/chapter_3

Chapter 3 moved from MATLAB simulation to real Ollama on an M-series Mac using `qwen2.5:3b`.

The plan was still close to the Chapter 2 model:

1. Sweep `B` at `q = 0` to identify the service-time terms.
2. Run sustained load at different arrival rates.
3. Let the queue settle.
4. Fit `beta`, the queue-to-latency gain.

The script even says this explicitly:

```text
l_mean = alpha*B + gamma*B^2 + beta*q
```

This is a very reasonable thing to try. It also did not give the plant I wanted.

The issue was that the experiment did not create a meaningful persistent queue. The drain rule and sequential request pattern kept `q` close to zero. The supposed queue term was not really identifiable. If the state almost never moves, you can give it a Greek letter, but the regression still has no magic powers. MATLAB is good, but it is not a therapist for underexcited systems.

This was the first important negative result:

The top-level request path looked like a serving system, but the Chapter 2 queue state was not present in the way the model required.

That does not mean Ollama had no internal scheduling, or that requests did no waiting anywhere. It means the variable available to my experiment was not the clean state in the Chapter 2 equations.

This is where the project started to become less about controller design and more about plant discovery.

## Chapter 4: when the queue is missing, control latency directly

Chapter 4:
https://github.com/hari-vasudevan/llm-serving-control/tree/main/chapter_4

Chapter 4 is a good example of not forcing the original architecture when the plant refuses to cooperate.

The Chapter 3 lesson was that the queue term was not identifiable in that setup. So Chapter 4 stopped pretending the cascade was the right abstraction and identified a direct map:

```text
TTFT = f(B)
```

The important experimental correction was concurrency. Sequential requests do not create the same contention as concurrent requests. So the Chapter 4 identification fired `B` requests concurrently using MATLAB `parfeval` and measured first-token latency.

That produced a more honest plant for that setup:

- `B` was effectively a concurrency input
- TTFT changed with `B`
- the queue was not the controlled state
- a single-loop integral controller made more sense than a cascade

This chapter did not validate the original cascade. It did something better: it admitted the cascade was the wrong tool for that layer.

That sounds obvious in hindsight, but at the time it was an important correction. The controller was not sacred. The plant was.

Also, this is where I started becoming suspicious of any variable named `queue` unless I had personally seen it wait. Trust, but verify. Especially if the metric comes from three abstraction layers away and is wearing a nice dashboard.

## Chapter 5: vLLM on Apple Silicon, and more ways for a queue to not be a queue

Chapter 5:
https://github.com/hari-vasudevan/llm-serving-control/tree/main/chapter_5

Chapter 5 moved to vLLM on Apple Silicon with the Metal backend.

The hope was that a real serving stack would expose better scheduler metrics and make the cascade story more physical. Instead, Chapter 5 became a very useful pile of "almost, but not quite."

There were two separate problems.

The first was load generation. MATLAB `parfeval` had too much dispatch overhead for the experiment. The Chapter 5 characterization notes that dispatching 8 calls could take around 600 ms because of worker overhead. If the requests themselves complete in roughly the same timescale, the load generator is no longer a sharp input. By the time the last request arrives, the first request may already be gone. That is not a burst. That is a polite queue of requests forming an orderly line before the server even gets to be stressed.

So the experiment separated load generation from measurement:

- Python threads generated concurrent HTTP load.
- MATLAB observed `/metrics`.
- MATLAB controlled experiment timing.

That was a good architectural correction.

The second problem was worse for the cascade: the vLLM Metal waiting metric was not usable. The README notes that `vllm:num_requests_waiting` accumulated monotonically due to a Prometheus multiprocessing gauge issue. A queue metric that only goes up is a mood, not a state variable.

Chapter 5 then used a software inflight counter as a proxy. This helped move the work forward, but it was still not the same as exposing the real scheduler backlog. Software proxies can be useful, but they need to be treated with suspicion. A proxy queue can tell you what your wrapper thinks is happening. It may not tell you what the GPU scheduler is doing.

The lesson from Chapter 5 was not "vLLM is bad" or "Metal is bad." The lesson was more specific:

If the metric you need is broken, or if your load generator is slower than the effect you want to observe, the control experiment is already compromised.

That sounds like a lab safety poster for control engineers. Maybe it should be.

## Chapter 6: the Intel Mac queue server, the first real waiting room

Chapter 6:
https://github.com/hari-vasudevan/llm-serving-control/tree/main/chapter_6

Chapter 6 was the pivot point.

The architecture moved to a real queue server running on an Intel Mac:

```text
M-series Mac controller -> Intel Mac queue_server.py -> Ollama -> CPU
```

The server exposed:

- `/enqueue`
- `/metrics`
- `/control`
- `/reset`

Most importantly, requests genuinely waited in a FIFO queue until the dispatcher picked them up. This was the first setup where `l_total` included real queue wait:

```text
l_total = queue_wait + TTFT
```

This felt like progress because the queue finally behaved like a queue. A tiny victory, but after a few chapters of queue-adjacent fog, I was willing to celebrate. Quietly. With maybe one extra cup of coffee.

Chapter 6 clarified a few important things.

First, total latency is not the same as model service time. If the controller uses `l_total` without separating queue wait from TTFT, the sign of the response can get confusing. Reducing `B` may reduce service cost, but it can also increase queue wait faster. At high queue depth, that can become positive feedback.

Second, a FIFO queue alone is not enough. The compute layer also needs to behave like a batching plant. On the Intel CPU, the machine time-sliced work rather than giving a clean GPU-style batch service relationship. That means the queue was real, but the independent `B -> q` drain-rate handle was still not as clean as the Chapter 2 model wanted.

This was the point where the next direction became clear:

I needed a real GPU experiment, but with more control over the layer being measured.

Chapter 6 did not finish the cascade story. It made the story sharper. It showed that real queue wait matters, that latency must be decomposed, and that CPU time-slicing was not the right place to validate a batch scheduler controller.

That is what pushed the project toward Modal.

## Chapter 7: Modal and native vLLM, useful but still too high level

Chapter 7:
https://github.com/hari-vasudevan/llm-serving-control/tree/main/chapter_7

Chapter 7 moved to Modal with native vLLM on an NVIDIA T4.

This chapter mattered because it proved the remote GPU path worked:

- deploy a cheap GPU endpoint
- drive it from the local controller
- read health and metrics
- characterize latency under concurrency
- run a closed-loop experiment

The actuator in this chapter was client-side concurrency `C`, not batch size `B`. The characterization found a latency-vs-concurrency relationship. The single-loop controller could move concurrency in response to latency.

But the native vLLM waiting-queue metric stayed near zero in the main run.

So again, the system was controllable, but it was not the Chapter 2 cascade plant.

This is an important distinction. A working controller is not automatically a validation of the model you originally wanted to test. Chapter 7 validated that a remote endpoint could be controlled from the outside. It did not validate the queue-plus-batch abstraction.

The danger here would have been to declare success because a plot moved in the right direction. Plots are persuasive. Some plots are also very good at lying politely.

## Chapter 8: wrapper queue on Modal, and the wrong layer problem

Chapter 8:
https://github.com/hari-vasudevan/llm-serving-control/tree/main/chapter_8

Chapter 8 tried to recreate the Chapter 2 cascade more directly on Modal:

```text
MATLAB controller -> Modal wrapper queue -> vLLM -> GPU
```

The wrapper accepted requests, stored them in a software FIFO, dispatched exactly `B` requests per tick into vLLM, and returned queue and latency summaries back to MATLAB.

This was the closest top-level LLM serving experiment to the original cascade design. It had:

- a wrapper FIFO
- per-tick dispatch
- MATLAB characterization
- MATLAB cascade design
- remote GPU execution
- load segments and closed-loop traces

And still, the outer plant did not look physical.

The key identification result was:

```text
l_mean(q_mean) = -4.9228*q + 648.7647
```

A negative queue-to-latency slope is not a credible queueing law for the plant we were trying to identify. If a larger queue appears to reduce latency, something else is dominating the measurement.

This was the big conceptual correction:

The control law was not necessarily wrong. It was being applied at the wrong layer.

At the top-level LLM API, latency includes too many things:

- wrapper queueing
- vLLM scheduler timing
- internal batching
- model execution
- streaming behavior
- network timing
- request variability

The wrapper queue was real. But the measured top-level latency was still too aggregated to expose a clean `q -> L_mean` relation.

The Chapter 2 variables existed in my wrapper code. They still did not exist cleanly in the plant being measured.

This was annoying, but also useful. The wrong sign is not a small problem. It is the experiment waving both arms and saying, "please stop fitting this line."

## Chapter 9: moving down until the variables became physical

Chapter 9:
https://github.com/hari-vasudevan/llm-serving-control/tree/main/chapter_9

Chapter 9 moved one level lower.

Instead of treating the whole LLM API as the plant, the experiment became a lower-level GPU batching plant:

```text
MATLAB cascade controller
  -> Modal endpoint
  -> FIFO queue
  -> exact batch-size actuator B[k]
  -> fixed PyTorch GPU batch workload
  -> measured batch service time
```

This removed many of the confounding effects:

- no token streaming inside the measured service time
- no opaque vLLM scheduler inside the control loop
- no prompt-length variability
- no hidden request batching policy
- no MATLAB-to-Modal round trip inside the control clock

The scheduler was deliberately ticked:

```text
once per tick:
    inject arrivals
    pop up to B[k] jobs from FIFO
    run one GPU batch
    leave the rest as carry-over backlog
```

The definition of `q` mattered a lot. The working version used `q_after`, the carry-over backlog after dispatch. The pre-dispatch queue includes the current tick's arrivals, so it has a moving arrival floor. That made `q_ref` physically untrackable. Once `q` meant carry-over backlog, the inner loop finally had a state it could regulate.

[Insert Figure 2: Chapter 9 plant characterization]

Caption: In Chapter 9 the signs finally matched the theory: increasing `B` drained carry-over backlog, `q -> L_mean` was positive, and service time was measured per GPU batch.

This is where the Chapter 2 cascade made physical sense again.

`B` was no longer client concurrency. It was no longer a wrapper-level hint sent into an opaque scheduler. It was the actual batch size of a fixed GPU workload.

`q` was no longer a remote metric or an inferred top-level queue. It was the remaining FIFO backlog after the dispatch decision.

Service time was measured around GPU work:

```text
synchronize
start timer
run batch
synchronize
stop timer
```

That does not mean Chapter 9 is a production LLM serving system. It is deliberately lower-level than that. But that was the point. It was the simplest real plant where the Chapter 2 variables became real enough to test.

[Insert Figure 3: Chapter 9 closed-loop run]

Caption: The final Chapter 9 run used the Chapter 2 cascade shape without arrival feedforward. The inner loop moved `B` to track carry-over backlog, while the outer loop moved `q_ref` to regulate `L_mean` across load and reference changes.

## The model changed, and that is allowed

One thing I want to be honest about is that the model evolved.

The early latency model had a quadratic service term:

```text
alpha*B + gamma*B^2
```

That was a reasonable starting assumption. Larger batches can have nonlinear service cost. But in the small-scale Chapter 9 experiment, the measured GPU batch service did not show a useful quadratic coefficient in the way the early model suggested.

That does not make the early model useless. It made the early model a scaffold. It gave me a thing to measure, and then the measurement told me which parts were still worth keeping.

This is the healthiest role for simple models. They are not contracts. They are instruments. Sometimes the instrument points at the thing you wanted. Sometimes it points at the floor and you realize you installed the sensor upside down. Both are data, though one is less dignified.

The final Chapter 9 lesson is also sharper than the original plan:

```text
L_mean ~= queue_wait(q) + service(B)
```

Total latency is not only a function of backlog. It also contains a service-time term that changes with batch size and operating regime. The outer loop should account for that instead of treating latency as a pure queue-depth output.

That is future work, not a footnote.

## What I would take forward

The most important output of this project is not a tuned controller. It is a better sense of where a controller should sit.

The top-level LLM serving API is useful for product metrics:

- user-visible latency
- throughput
- errors
- request success
- cost

But it may be too high in the stack for a clean batching controller. By the time a request becomes an API latency sample, it has passed through many layers.

The control layer wants variables that are physically close to the scheduler:

- a real backlog state
- a direct batch-size or service-rate actuator
- measured service time per batch
- separate queue-wait and compute-service signals
- a sampling tick owned by the scheduler, not by a remote client

The project moved through several abstractions:

Chapter 1/2: clean simulation, where the controller and plant agreed.

Chapter 3: real Ollama experiment, where the queue term was not identifiable.

Chapter 4: direct TTFT-vs-concurrency control, useful because the queue was not the right state.

Chapter 5: vLLM on Apple Silicon, where load generation and metrics both created problems.

Chapter 6: Intel Mac queue server, where requests finally waited for real, but CPU behavior still did not expose a clean batching plant.

Chapter 7: native vLLM on Modal, where remote GPU control worked but the native queue stayed near zero.

Chapter 8: Modal wrapper queue, where the software queue was real but top-level LLM latency still hid the plant.

Chapter 9: lower-level GPU batching, where `B`, `q`, and service time finally became physical.

Looking back, the movement downward was not a retreat from the original idea. It was the process of finding where the original idea actually applied.

That is the main thing I learned.

The controller was simple. The hard part was earning the right to use it.

Or, to put it in the most engineering way possible:

Before tuning the controller, make sure the plant is not imaginary.

This advice has unfortunately high reuse potential.

