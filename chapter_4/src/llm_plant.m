function [q_next, L_mean, L_p95] = llm_plant(q, B, lambda, perturbed)
%LLM_PLANT Nonlinear LLM inference serving plant (pure MATLAB function).
%
% Inputs
%   q          current queue depth              [requests]
%   B          batch size (control input)       [requests]
%   lambda     arrivals during this tick        [requests/tick]
%   perturbed  plant parameter struct
%
% Outputs
%   q_next     queue depth at next tick         [requests]
%   L_mean     mean end-to-end latency          [ms]
%   L_p95      surrogate p95 latency            [ms]
%
% Plant equations
%   State:
%     q[k+1] = max(0, q[k] + lambda[k] - B[k])
%
%   Deterministic mean-latency output:
%     L_mean[k] = alpha*B[k] + gamma*B[k]^2 + beta*q[k]
%
%   Optional surrogate p95 output:
%     L_p95[k]  = L_mean[k] + 1.645*delta/sqrt(max(B[k],1))
%
% Notes
%   1) The equilibrium for current controller design is computed from
%      L_mean, not L_p95.
%   2) L_p95 here is only a deterministic approximation/surrogate, not a
%      true rolling statistical percentile estimator.
%   3) Queue balance at equilibrium requires B0 = lambda_mean.

alpha = perturbed.alpha;
gam   = perturbed.gamma;
beta  = perturbed.beta;
delta = perturbed.delta;

% Enforce physically meaningful values
q      = max(0, q);
B      = max(perturbed.B_min, min(perturbed.B_max, B));
lambda = max(0, lambda);

% Queue update
q_next = max(0, min(perturbed.q_max, q + lambda - B));

% Mean latency model
L_mean = alpha*B + gam*B^2 + beta*q;

% Surrogate p95 latency model
B_safe = max(1, B);
L_p95  = L_mean + 1.645 * delta / sqrt(B_safe);

end