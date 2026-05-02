%% design_controller.m  --  Chapter 6: Single-loop integral controller
%
% The Intel Mac runs inference sequentially with time-sliced concurrency.
% The cascade (inner queue loop + outer latency loop) requires independent
% control of queue depth and latency -- not possible here because:
%   - Service rate barely changes with B (CPU time-slices, doesn't batch)
%   - The software FIFO queue is controlled by our arrival rate, not by B
%   - Once queue builds, it cannot drain unless lambda < B
%
% The correct model for this machine is the direct TTFT(B) relationship
% identified in Stage 2:
%   TTFT(B) ≈ alpha*B + gamma*B^2    (R^2 ≈ 0.89)
%
% Single integral controller (same as Chapter 4):
%   e[k]       = L_target - l_meas[k]        error
%   xi[k+1]    = clip(xi[k] + e[k], ...)     integrator with anti-windup
%   B[k]       = clamp(B0 + K_il * xi[k], B_min, B_max)
%
% Gain design (linearised plant at B0):
%   beta_eff = d(TTFT)/dB|_B0 = alpha + 2*gamma*B0
%   Desired CL pole: z_cl = exp(-dt/tau_cl)
%   K_il = (1 - z_cl) / beta_eff   [POSITIVE: dl/dB > 0 so K > 0]
%
% Sign check (K_il > 0):
%   l > L_target (e < 0) -> xi decreases -> B = B0 + K*xi decreases
%                        -> TTFT decreases -> l decreases  CORRECT
%   l < L_target (e > 0) -> xi increases -> B increases
%                        -> TTFT increases -> l increases  CORRECT

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% ── Tuning ────────────────────────────────────────────────────────────────
TAU_CL  = 20.0;   % closed-loop time constant [s]
                  % slower = more robust to measurement noise
                  % 20s means ~20 ticks to correct a step disturbance
B_MIN   = 1;
B_MAX   = 4;      % OLLAMA_NUM_PARALLEL ceiling

% ── Load identified params ────────────────────────────────────────────────
out_dir = fileparts(mfilename('fullpath'));
load(fullfile(out_dir, 'identified_params.mat'));

% Use identified lambda_ss as B0 if available
if exist('lambda_ss','var') && lambda_ss >= B_MIN && lambda_ss <= B_MAX
    B0 = lambda_ss;
else
    B0 = 2;
end
DT = 1.0;

ttft_B0  = alpha*B0 + gamma*B0^2;
beta_eff = alpha + 2*gamma*B0;

% ── Design ────────────────────────────────────────────────────────────────
z_cl = exp(-DT / TAU_CL);
K_il = (1 - z_cl) / beta_eff;     % positive

% Anti-windup: B = B0 + K_il * xi in [B_min, B_max]
% xi range: [(B_min - B0) / K_il, (B_max - B0) / K_il]
xi_min = (B_MIN - B0) / K_il;
xi_max = (B_MAX - B0) / K_il;

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 6: Single-Loop Integral Controller                  ║\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Plant:  TTFT(B) = %.4f*B + (%.4f)*B^2\n', alpha, gamma);
fprintf('║          beta_eff = %.4f ms/req @ B0=%d\n', beta_eff, B0);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  K_il  = %.8f  (>0: %s)\n', K_il, mat2str(K_il > 0));
fprintf('║  z_cl  = %.6f   tau_cl = %.0fs\n', z_cl, TAU_CL);
fprintf('║  xi range: [%.2f, %.2f]\n', xi_min, xi_max);
fprintf('║  B0=%d  B=[%d,%d]\n', B0, B_MIN, B_MAX);
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

assert(K_il > 0, sprintf('K_il should be positive, got %.6f', K_il));
assert(abs(z_cl) < 1, 'CL pole unstable');

save(fullfile(out_dir, 'controller_params.mat'), ...
    'alpha','gamma','beta_eff','B0','DT','B_MIN','B_MAX', ...
    'K_il','z_cl','TAU_CL','xi_min','xi_max','SERVER');
fprintf('[save] controller_params.mat\n');
