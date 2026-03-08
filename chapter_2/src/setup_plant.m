%% setup_plant.m  — chapter 2
%
%  entry point.  defines plant parameters in 'perturbed', then calls
%  design_controller.m to produce the 'controller' struct used by simulink.
%
%  controller architecture:
%    direct loop  b[k] -> l_p95[k]
%    augmented state: x = [dq ; xi]  where xi integrates latency error
%
%  change 'method' to switch between 'lqr' and 'pole_placement'.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% ── plant parameters ───────────────────────────────────────────────────────
%
%   latency model:  l_p95 = alpha*b + gamma*b^2 + beta*q + 1.645*delta/sqrt(b)
%
%   alpha   : linear service-time coefficient         [ms per req in batch]
%   gamma   : quadratic service-time coefficient      [ms per req^2]
%   beta    : queue-to-latency sensitivity            [ms per waiting req]
%   delta   : p95 spread coefficient                  [ms * sqrt(req)]
%             (noise term is (delta/sqrt(b))*randn per tick,
%              so p95 offset above mean = 1.645*delta/sqrt(b))

perturbed.alpha  = 10;    % ms
perturbed.gamma  = 0.8;   % ms
perturbed.beta   = 2;     % ms/req
perturbed.delta  = 15;    % ms*sqrt(req)

perturbed.dt     = 0.1;   % s  — scheduling tick
perturbed.B_min  = 1;     % req/tick — minimum allowed batch size
perturbed.B_max  = 32;    % req/tick — maximum allowed batch size (vram limit)
perturbed.q_max  = 30;    % req      — integrator saturation bound

% ── operating conditions ───────────────────────────────────────────────────
perturbed.lambda_mean  = 8;    % req/tick — mean arrival rate
perturbed.L_p95_target = 150;  % ms       — sla target for l_p95

% ── lqr weights ────────────────────────────────────────────────────────────
%
%   q penalises state cost, r penalises control effort.
%
%   lqr_q  : cost on dq  (queue deviation)
%              larger => controller reacts faster to queue changes,
%              more aggressive b swings
%
%   lqr_xi : cost on xi  (latency integrator state)
%              larger => faster integral wind-up removal
%
%   lqr_r  : cost on db  (batch size change)
%              larger => smoother b, slower latency correction
%              increasing r is the practical way to detune away from
%              the nmp zero without explicit bandwidth constraints

perturbed.lqr_q  = 1;     % cost on dq
perturbed.lqr_xi = 0.1;   % cost on xi
perturbed.lqr_r  = 50;    % cost on db  (high => slow, safe w.r.t. nmp)

% ── pole placement parameters ─────────────────────────────────────────────
%
%   nmp zero at z = 1.088  =>  tau_nmp ~ 1.2s
%   practical guideline: tau >= 1.0s to keep undershoot below ~2ms.
%   poles faster than tau_nmp are stable but produce larger undershoot.

perturbed.pp_tau1 = 2.0;   % s — dominant closed-loop time constant
perturbed.pp_tau2 = 4.0;   % s — second pole (slower, supports integrator)

% ── method selection ───────────────────────────────────────────────────────
method = 'lqr';
% method = 'pole_placement';

% ── design controller ──────────────────────────────────────────────────────
controller = design_controller(perturbed, method);
