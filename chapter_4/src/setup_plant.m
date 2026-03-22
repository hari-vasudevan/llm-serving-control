%% setup_plant.m  --  Chapter 4: Single-loop integral controller on TTFT
%
% Chapter 4 architecture:
%   Real Ollama plant (qwen2.5:3b), identified TTFT(B) curve, single-loop
%   integral controller — B directly regulates TTFT via GPU concurrency.
%
% Why single-loop (not cascade):
%   Chapter 3 showed q≈0 in the real system — the drain rule always
%   matches arrivals so no persistent queue builds.  The cascade inner loop
%   was regulating a state that doesn't exist.  The real dynamics are:
%     l_meas[k] = f(B[k])   -- static nonlinear map, identified from data
%   Linearised: Δl = beta_eff * ΔB   where beta_eff = dl/dB|_{B0}
%   A single integral controller is sufficient.
%
% Controller law:
%   e[k]       = l_target - l_meas[k]
%   xi[k+1]    = xi[k] + e[k]
%   dB[k]      = -K_il * xi[k]
%   B[k]       = clamp(B0 + dB[k], B_min, B_max)
%
%   K_il = (exp(-dt/tau_cl) - 1) / beta_eff
%
% Closed-loop pole:  z_cl = 1 + beta_eff * K_il = exp(-dt/tau_cl)
%
% IMPORTANT: run setup_plant.m BEFORE pressing Run in Simulink.
%            run identify_plant.m first if identified_params.mat does not exist.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% -------------------------------------------------------------------------
% 1. Load identified parameters
% -------------------------------------------------------------------------
id_path = fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'identification', 'identified_params.mat');

if ~exist(id_path, 'file')
    error(['identified_params.mat not found.\n' ...
           'Run chapter_4/identification/identify_plant.m first.\n' ...
           'Expected at: %s'], id_path);
end

load(id_path, 'identified');
fprintf('=== Loaded identified parameters (%s) ===\n', identified.model);
fprintf('  l(B) = %.4f + %.4f*B + %.4f*B^2   (R^2 = %.4f)\n', ...
    identified.c0, identified.c1, identified.c2, identified.r2);
fprintf('  B0        = %d\n',       identified.B0);
fprintf('  l(B0)     = %.2f ms\n',  identified.l0);
fprintf('  beta_eff  = %.4f ms/req\n\n', identified.beta_eff);

% -------------------------------------------------------------------------
% 2. Plant parameters (physical bounds only — latency model is identified)
% -------------------------------------------------------------------------
perturbed.dt      = 1.0;    % s   -- scheduling tick
perturbed.B_min   = 1;      %     -- batch size lower bound
perturbed.B_max   = 12;     %     -- safe ceiling (wall-clock < 800ms at B=12)
perturbed.q_max   = 1;      %     -- q≈0 always; kept for interface compatibility

% Equilibrium from identification
perturbed.B0      = identified.B0;
perturbed.q0      = 1;      % nominal; not used by single-loop controller
perturbed.beta_eff = identified.beta_eff;

% Operating conditions
perturbed.lambda_mean    = identified.B0;  % arrivals/tick ≈ B0 at equilibrium
perturbed.L_mean_target  = identified.l0 * 0.85;   % target = 85% of eq TTFT
perturbed.L_p95_target   = identified.l0 * 1.5;    % p95 headroom

fprintf('=== Operating point ===\n');
fprintf('  lambda_mean    = %g req/tick\n',  perturbed.lambda_mean);
fprintf('  B0             = %d req\n',        perturbed.B0);
fprintf('  L_mean_target  = %.1f ms\n',       perturbed.L_mean_target);
fprintf('  L_p95_target   = %.1f ms\n\n',     perturbed.L_p95_target);

% -------------------------------------------------------------------------
% 3. Rolling window
% -------------------------------------------------------------------------
perturbed.N_win = 20;   % samples -- 20 s rolling window at dt=1s

% -------------------------------------------------------------------------
% 4. Controller design -- single-loop integral
% -------------------------------------------------------------------------
%
% Plant (linearised at B0):
%   Δl[k] = beta_eff * ΔB[k]       (static gain, no plant dynamics)
%
% Augmented with integrator:
%   xi[k+1] = xi[k] + e[k]  where  e[k] = l_target - l_meas[k] = -Δl[k]
%
% Closed-loop pole with gain K_il:
%   z_cl = 1 + beta_eff * K_il
%
% Gain from desired closed-loop time constant tau_cl:
%   z_cl = exp(-dt / tau_cl)
%   K_il = (z_cl - 1) / beta_eff
%
% Sign check:
%   If beta_eff > 0 (TTFT increases with B): K_il < 0 (reduce B when l > target)
%   If beta_eff < 0 (TTFT decreases with B): K_il > 0 (increase B when l > target)

dt      = perturbed.dt;
tau_cl  = 20.0;   % s -- closed-loop time constant (>> N_win*dt = 20s)
z_cl    = exp(-dt / tau_cl);
K_il    = (z_cl - 1) / perturbed.beta_eff;

% Stability check
z_cl_actual = 1 + perturbed.beta_eff * K_il;
stable      = abs(z_cl_actual) < 1;

% Anti-windup limits for xi (in xi domain)
%   B = B0 - K_il * xi  in [B_min, B_max]
%   => xi in [(B0 - B_max)/K_il,  (B0 - B_min)/K_il]  if K_il > 0
%      xi in [(B0 - B_min)/K_il,  (B0 - B_max)/K_il]  if K_il < 0
if K_il ~= 0
    bounds  = [(perturbed.B0 - perturbed.B_max) / K_il, ...
               (perturbed.B0 - perturbed.B_min) / K_il];
    xi_min  = min(bounds);
    xi_max  = max(bounds);
else
    xi_min = -1e6;  xi_max = 1e6;
end

fprintf('=== Controller design (single-loop integral) ===\n');
fprintf('  tau_cl    = %.1f s\n',   tau_cl);
fprintf('  z_cl      = %.4f\n',     z_cl);
fprintf('  K_il      = %.6f\n',     K_il);
fprintf('  z_cl_act  = %.4f\n',     z_cl_actual);
fprintf('  Stable:   %s\n',         mat2str(stable));
fprintf('  xi range: [%.2f, %.2f]\n\n', xi_min, xi_max);

% Pack controller struct (all-numeric for Simulink compatibility)
controller.K_il      = K_il;
controller.z_cl      = z_cl;
controller.tau_cl    = tau_cl;
controller.B0        = perturbed.B0;
controller.B_min     = perturbed.B_min;
controller.B_max     = perturbed.B_max;
controller.xi_min    = xi_min;
controller.xi_max    = xi_max;
controller.L_target  = perturbed.L_mean_target;
controller.beta_eff  = perturbed.beta_eff;
% q0 kept for inner_c compatibility with existing Simulink wiring
controller.inner_c.q0    = perturbed.q0;
controller.inner_c.B0    = perturbed.B0;
controller.inner_c.q_max = perturbed.q_max;

% -------------------------------------------------------------------------
% 5. Ollama plant object
% -------------------------------------------------------------------------
ollama = ollama_plant();
ollama.ollama_url    = 'http://localhost:11434/api/generate';
ollama.model_name    = identified.model;
ollama.num_predict   = 1;
ollama.n_win         = perturbed.N_win;
ollama.http_timeout  = 15;
ollama.n_warmup      = 4;
ollama.q_max         = 20;
ollama.b_min         = perturbed.B_min;
ollama.b_max         = perturbed.B_max;
ollama.prompts_path  = fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'llm_requirements', 'prompts.txt');

load_prompts(ollama);

if isempty(gcp('nocreate'))
    parpool('local', min(ollama.b_max, feature('numcores')));
end
src_path = fileparts(mfilename('fullpath'));
addAttachedFiles(gcp, src_path);

fprintf('ollama_plant ready: %d prompts. model=%s. B0=%d. B_max=%d.\n', ...
    numel(ollama.prompt_list), ollama.model_name, perturbed.B0, perturbed.B_max);
fprintf('z^-1 IC = perturbed.B0 = %d\n', perturbed.B0);
