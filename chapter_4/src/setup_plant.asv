%% setup_plant.m  --  Chapter 3: Cascade Control on Real Ollama Inference
% Entry point. Defines plant parameters, designs the cascade controller,
% and instantiates the ollama_plant System object for Simulink.
%
% IMPORTANT: run this script BEFORE pressing Run in Simulink.
% The Simulink model reads perturbed, controller, and ollama from the
% base workspace at simulation start.
%
% Plant switching (inside Plant1 subsystem in Simulink):
%   Stochastic model  ->  fcn MATLAB Function block connected
%   Real Ollama       ->  ollama_plant MATLAB System block connected

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% -- 1. Plant parameters -------------------------------------------------------
% alpha/gamma/delta are used by the stochastic plant only.
% The real Ollama plant measures latency directly from wall-clock time.
perturbed.alpha = 0.1;   % ms -- stochastic plant: linear service-time coeff
perturbed.gamma = 0.8;   % ms -- stochastic plant: quadratic coeff
perturbed.beta  = 2;     % ms/req -- queuing latency (used for outer loop gain)
perturbed.delta = 15;    % ms -- stochastic plant: p95 spread coeff

perturbed.dt    = 1.0;   % s   -- scheduling tick
perturbed.B_min = 1;     %     -- batch size lower bound
perturbed.B_max = 12;    %     -- 2.4x lambda_mean; wall~490ms, safe for dt=1s
perturbed.q_max = 12;    %     -- 2.4x lambda_mean; realistic cap

% -- 2. Operating conditions ---------------------------------------------------
% Tuned for qwen2.5:7b on M2, num_predict=1:
%   Measured single-request warm latency:  ~175-240 ms
%   4 concurrent requests wall-clock time: ~200 ms
perturbed.lambda_mean   = 5;    % req/tick -- 5 arrivals/s: B0=5, wall~250ms, TTFT~206ms
perturbed.L_p95_target  = 400;  % ms -- realistic p95 for qwen2.5:3b
perturbed.L_mean_target = 220;  % ms -- between min 120ms (B=2) and eq 206ms (B=5)

% -- 3. Rolling window ----------------------------------------------------------
perturbed.N_win = 2;    % samples -- rolling window (20 s at dt=1 s)

% -- 4. Equilibrium -------------------------------------------------------------
%   Queue balance: B0 = lambda_mean
%   q0: initial queue setpoint. Set to lambda_mean so the system starts
%       near natural balance. The outer loop drives latency from here.
perturbed.B0 = perturbed.lambda_mean;   % = 3
perturbed.q0 = perturbed.lambda_mean;   % = 3 requests (initial setpoint)

% Evaluate stochastic plant at equilibrium (sanity check only)
[~, L_mean_eq, L_p95_eq] = llm_plant(perturbed.q0, perturbed.B0, ...
    perturbed.lambda_mean, perturbed);
fprintf('=== Equilibrium ===\n');
fprintf('  B0  = %.2f  req/tick\n', perturbed.B0);
fprintf('  q0  = %.2f  requests\n', perturbed.q0);
fprintf('  Stochastic plant at eq: L_mean=%.1f ms, L_p95=%.1f ms\n', L_mean_eq, L_p95_eq);
fprintf('  Real Ollama target:     L_mean=%.0f ms, L_p95=%.0f ms\n\n', ...
    perturbed.L_mean_target, perturbed.L_p95_target);

% -- 5. Inner loop pole placement params ----------------------------------------
%   Time constants must be >> dt=1s and << tau_out=30s
perturbed.pp_tau1 = 5.0;    % s -- inner pole 1
perturbed.pp_tau2 = 8.0;    % s -- inner pole 2
perturbed.pp_tau  = 3.0;    % s -- oscillatory envelope (if pp_f>0)
perturbed.pp_f    = 0;      % Hz -- 0 = non-oscillatory

% -- 6. Outer loop time constant ------------------------------------------------
%   tau_out >> N_win*dt = 20s measurement lag
perturbed.tau_out = 30;     % s -- outer CL time constant

% -- 7. Design cascade controller -----------------------------------------------
% method = 'lqr';
method = 'pole_placement';

controller = design_controller(perturbed, method);

% -- 8. Ollama plant object (Chapter 3) ----------------------------------------
%   Prompts loaded HERE (not inside setupImpl) so Simulink propagation
%   pass never touches fopen/regexprep.
%   Parallel pool also opened here.
ollama = ollama_plant();
ollama.ollama_url    = 'http://localhost:11434/api/generate';
ollama.model_name    = 'qwen2.5:3b';
ollama.num_predict   = 1;     % 1 token: ~175-200 ms/req on M2 (fastest)
ollama.n_win         = perturbed.N_win;
ollama.http_timeout  = 10;    % s -- abort after 10s
ollama.n_warmup      = 4;
ollama.q_max         = perturbed.q_max;
ollama.b_min         = perturbed.B_min;
ollama.b_max                = perturbed.B_max;   % = 12
ollama.arrival_noise_factor = 0.3;  % 0=deterministic, 1=full Poisson, 0.3=reduced noise
ollama.prompts_path  = fullfile(fileparts(mfilename('fullpath')), ...
    '..', 'llm_requirements', 'prompts.txt');

load_prompts(ollama);

if isempty(gcp('nocreate'))
    parpool('local', min(ollama.b_max, feature('numcores')));
end
% Attach src directory to pool workers so ollama_ttft.m is findable
src_path = fileparts(mfilename('fullpath'));
addAttachedFiles(gcp, src_path);

fprintf('ollama_plant ready: %d prompts. num_predict=%d. B_max=%d.\n', ...
    numel(ollama.prompt_list), ollama.num_predict, ollama.b_max);
fprintf('z^-1 IC = perturbed.q0 = %.2f  (set in Simulink z^-1 block)\n', perturbed.q0);
