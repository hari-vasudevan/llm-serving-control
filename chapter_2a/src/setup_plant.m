%% setup_plant.m  — chapter 2
%
%  entry point.  defines plant parameters, calls design_controller.m to
%  produce the 'controller' struct used by simulink.
%
%  controller architecture:
%    direct single loop:  b[k] -> l_p95[k]
%    augmented state:     x = [dq ; xi]
%    integrator:          xi accumulates measured latency error directly
%                         xi[k+1] = xi[k] + (l_target - l_p95_measured[k])
%
%  the normal (measurement-based) integrator is used in the simulink block
%  rather than the model-based update xi[k+1] = xi[k] - beta*dq - c1*db.
%  the lqr/pole-placement design is still done on the linearised augmented
%  model — in the linear regime both are equivalent.  the measurement-based
%  integrator is more robust to model mismatch but introduces the rolling
%  buffer lag (~6s) into the integral path.  lqr_r and pp_tau2 must be
%  set large enough that k_i is slow relative to this lag.
%
%  change 'method' to switch between 'lqr' and 'pole_placement'.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% ── plant parameters ───────────────────────────────────────────────────────
%
%   latency model (output equation h(q,b)):
%     l_p95 = alpha*b + gamma*b^2 + beta*q + 1.645*delta/sqrt(b)
%
%   alpha  : linear service-time coefficient    [ms / req-in-batch]
%   gamma  : quadratic service-time coefficient [ms / req^2]
%   beta   : queue-to-latency sensitivity       [ms / waiting-req]
%   delta  : p95 spread coefficient             [ms * sqrt(req)]
%            per-tick noise is (delta/sqrt(b))*randn(), so the expected
%            p95 offset above the mean is 1.645*delta/sqrt(b).

perturbed.alpha  = 10;    % ms
perturbed.gamma  = 0.8;   % ms
perturbed.beta   = 2;     % ms/req
perturbed.delta  = 15;    % ms*sqrt(req)

perturbed.dt     = 0.1;   % s        — scheduling tick
perturbed.B_min  = 1;     % req/tick — minimum batch size
perturbed.B_max  = 32;    % req/tick — maximum batch size (vram limit)
perturbed.q_max  = 30;    % req      — dq saturation bound in controller

% ── operating conditions ───────────────────────────────────────────────────
perturbed.lambda_mean  = 8;    % req/tick — mean arrival rate at equilibrium
perturbed.L_p95_target = 150;  % ms       — sla target

% ── lqr weights ────────────────────────────────────────────────────────────
%
%   lqr_q  : cost on dq  (queue deviation from q_eq)
%              larger => faster reaction to queue changes, larger db swings
%
%   lqr_xi : cost on xi  (latency integrator state)
%              larger => faster unwinding of accumulated error
%              keep small — xi is slow due to rolling buffer lag
%
%   lqr_r  : cost on db  (batch size increment)
%              larger => smoother b, slower closed-loop response
%
%   bandwidth constraints for this plant:
%     1. nmp zero at tau_nmp ~ 2.3s  (see design_controller output)
%        => closed-loop tau should be >= tau_nmp to limit undershoot
%     2. rolling buffer lag ~ 6s (60 samples * dt=0.1s)
%        => integral path tau should be >= 6s to avoid integrator overshoot
%     effective limit: tau_cl >= max(tau_nmp, buffer_lag) ~ 6s
%
%   to achieve tau_cl ~ 6-10s with lqr: increase lqr_r.
%   rule of thumb: if poles are too fast, increase lqr_r by 10x and recheck.

perturbed.lqr_q  = 1;      % cost on dq
perturbed.lqr_xi = 0.1;    % cost on xi  (keep small: integrator is slow)
perturbed.lqr_r  = 500;    % cost on db  (large: respects buffer lag + nmp)

% ── pole placement parameters ─────────────────────────────────────────────
%
%   pp_tau1 : dominant closed-loop time constant [s]
%              should be >= max(tau_nmp, buffer_lag) ~ 6s
%   pp_tau2 : second pole time constant [s]
%              set slower than tau1 to keep the integrator from fighting
%              the proportional term during transients

perturbed.pp_tau1 = 8.0;    % s — dominant pole
perturbed.pp_tau2 = 16.0;   % s — integral pole

% ── method selection ───────────────────────────────────────────────────────
method = 'lqr';
% method = 'pole_placement';

% ── design controller ──────────────────────────────────────────────────────
controller = design_controller(perturbed, method);
