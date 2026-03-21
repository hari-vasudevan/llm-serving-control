function [q_next, L_mean, L_p95] = llm_plant(q, B, lambda, perturbed)
%LLM_PLANT  Statistical LLM inference serving plant — Chapter 2.
%
%   Identical to Chapter 1 except:
%     - Arrivals are true Poisson: a[k] ~ Poisson(lambda)
%     - Latency has additive per-tick noise
%     - L_p95 is empirical 95th pct of a 60-sample rolling buffer
%       (warm-started at L_p95_target to avoid cold-start integrator wind-up)
%
%   Signature matches Chapter 1 standalone plant for drop-in use.

persistent L_buf buf_idx buf_full
N = 60;
if isempty(L_buf)
    L_buf    = perturbed.L_p95_target * ones(1, N);
    buf_idx  = 0;
    buf_full = true;
end

alpha = perturbed.alpha;
gam   = perturbed.gamma;
beta  = perturbed.beta;
delta = perturbed.delta;

% True Poisson arrivals
a_k = poissrnd(max(0, lambda));

% Queue state equation
q_next = max(0, q + a_k - B);

% Latency sample
B_safe = max(1, B);
l_det  = alpha*B + gam*B^2 + beta*q;
l_samp = l_det + (delta / sqrt(B_safe)) * randn();

% Circular buffer update
buf_idx = mod(buf_idx, N) + 1;
L_buf(buf_idx) = l_samp;

% Rolling statistics
if buf_full
    data = L_buf;
else
    data = L_buf(1 : buf_idx);
end

L_mean = mean(data);
sorted = sort(data);
idx95  = max(1, ceil(0.95 * length(sorted)));
L_p95  = sorted(idx95);

end
