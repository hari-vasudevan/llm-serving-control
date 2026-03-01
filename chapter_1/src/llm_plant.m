function [q_next, L_mean, L_p95] = llm_plant(q, B, lambda, perturbed)
%LLM_PLANT  Nonlinear LLM inference serving plant (pure MATLAB function).
%
%   Inputs
%     q          current queue depth  [requests]
%     B          batch size           [requests]  — control input
%     lambda     arrival rate         [requests/tick]  — disturbance
%     perturbed  plant parameter structure
%
%   Outputs
%     q_next     queue depth at next tick
%     L_mean     mean end-to-end latency  [ms]
%     L_p95      p95  end-to-end latency  [ms]
%
%   Plant equations
%   ───────────────
%   State:   q[k+1] = max(0,  q[k] + lambda[k] - B[k])
%
%   Output:  L_mean[k] = alpha*B[k] + gamma*B[k]^2 + beta*q[k]
%              ↑ service time (nonlinear in B)   ↑ queuing time
%
%            L_p95[k]  = L_mean[k] + 1.645 * delta / sqrt(B[k])
%              ↑ p95 spread shrinks as batch grows (averaging effect)

alpha = perturbed.alpha;
gam   = perturbed.gamma;
beta  = perturbed.beta;
delta = perturbed.delta;

% State equation — queue is non-negative
q_next = max(0,  q + lambda - B);

% Output equations
B_safe = max(1, B);                          % guard sqrt against B→0
L_mean = alpha*B + gam*B^2 + beta*q;
L_p95  = L_mean + 1.645 * delta / sqrt(B_safe);

end
