%% id_stage2.m  --  Plant identification Stage 2: beta from sustained load
%
% Standalone. No dependency on setup_plant.m.
% Requires identified_stage1.mat from id_stage1.m.
%
% Identifies beta from:  l_ss - (alpha*B_ss + gamma*B_ss^2) = beta * q_ss
%
% Runs ONE lambda value at a time to stay within execution timeout.
% Change lambda_to_run below and re-run for each value.
% After all lambda values are done, run id_stage2_fit.m to fit beta.

clear; clc;
src_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'src');
addpath(src_dir);
out_dir = fileparts(mfilename('fullpath'));

% ── Load Stage 1 results ──────────────────────────────────────────────────
s1_path = fullfile(out_dir, 'identified_stage1.mat');
if ~exist(s1_path, 'file')
    error('Run id_stage1.m first.');
end
load(s1_path, 's1');
alpha_id = s1.alpha;
gamma_id = s1.gamma;
fprintf('Loaded stage 1: alpha=%.4f  gamma=%.4f\n\n', alpha_id, gamma_id);

% ── Config ────────────────────────────────────────────────────────────────
url         = 'http://localhost:11434/api/generate';
model       = 'qwen2.5:3b';
num_predict = 1;
timeout     = 15;
b_max       = 12;
b_min       = 1;
q_max       = 20;
settle_ticks  = 20;    % ticks to reach steady state
measure_ticks = 15;    % ticks to average over

% ── Choose lambda for this run ────────────────────────────────────────────
% Run this script separately for each value: 2, 3, 5, 7, 8
lambda_to_run = 5;    % <-- CHANGE THIS each run

fprintf('Stage 2: lambda = %d req/tick\n', lambda_to_run);
fprintf('Model: %s   settle=%d ticks   measure=%d ticks\n\n', ...
    model, settle_ticks, measure_ticks);

% ── Parallel pool ─────────────────────────────────────────────────────────
if isempty(gcp('nocreate'))
    parpool('local', min(16, feature('numcores')));
end
addAttachedFiles(gcp, src_dir);

% ── Warm-up ───────────────────────────────────────────────────────────────
fprintf('Warming up...\n');
wf = cell(4,1);
for i=1:4
    wf{i} = parfeval(@ollama_ttft,1,url,model,'Hello',1,timeout);
end
for i=1:4; try; fetchOutputs(wf{i}); catch; end; end
fprintf('Warm.\n\n');

% ── Sustained load run ────────────────────────────────────────────────────
% Simple drain rule: B[k] = clamp(q[k] + a[k], b_min, b_max)
% Drives system to a steady-state (B_ss, q_ss, l_ss) for identification.

total_ticks = settle_ticks + measure_ticks;
q_k         = lambda_to_run;
lat_buf     = 200 * ones(1, 20);
buf_idx     = 0;

all_q = zeros(total_ticks,1);
all_b = zeros(total_ticks,1);
all_l = zeros(total_ticks,1);

for tick = 1:total_ticks
    a_k = poissrnd(lambda_to_run);
    b_k = min(b_max, max(b_min, round(q_k + a_k)));

    futures = cell(b_k,1);
    for i=1:b_k
        futures{i} = parfeval(@ollama_ttft,1,url,model,'What is 2+2?',num_predict,timeout);
    end
    lats = zeros(1,b_k);
    for i=1:b_k
        try; lats(i) = fetchOutputs(futures{i}); catch; lats(i) = NaN; end
    end

    valid_lats = lats(~isnan(lats));
    for i=1:numel(valid_lats)
        buf_idx = mod(buf_idx, 20) + 1;
        lat_buf(buf_idx) = valid_lats(i);
    end
    l_meas = mean(lat_buf);
    q_k    = max(0, min(q_max, q_k + a_k - b_k));

    all_q(tick) = q_k;
    all_b(tick) = b_k;
    all_l(tick) = l_meas;

    phase = 'SETTLE';
    if tick > settle_ticks; phase = 'MEASURE'; end
    fprintf('  [%s] tick %2d:  a=%d  b=%d  q=%.1f  l=%.0f ms\n', ...
        phase, tick, a_k, b_k, q_k, l_meas);
end

% ── Steady-state averages ─────────────────────────────────────────────────
meas_idx = (settle_ticks+1):total_ticks;
q_ss = mean(all_q(meas_idx));
b_ss = mean(all_b(meas_idx));
l_ss = mean(all_l(meas_idx));
service = alpha_id * b_ss + gamma_id * b_ss^2;
beta_est = (q_ss > 0.1) * (l_ss - service) / max(q_ss, 0.01);

fprintf('\n  q_ss=%.2f  b_ss=%.2f  l_ss=%.1f ms\n', q_ss, b_ss, l_ss);
fprintf('  service term (alpha*B+gamma*B^2) = %.1f ms\n', service);
fprintf('  residual = %.1f ms  =>  beta_est = %.4f ms/req\n\n', l_ss-service, beta_est);

% ── Append to stage2 results file ─────────────────────────────────────────
s2_path = fullfile(out_dir, 'identified_stage2.mat');
if exist(s2_path, 'file')
    load(s2_path, 's2');
else
    s2.lambda = []; s2.q_ss = []; s2.b_ss = []; s2.l_ss = [];
    s2.beta_est = []; s2.model = model;
end
s2.lambda  = [s2.lambda;  lambda_to_run];
s2.q_ss    = [s2.q_ss;    q_ss];
s2.b_ss    = [s2.b_ss;    b_ss];
s2.l_ss    = [s2.l_ss;    l_ss];
s2.beta_est= [s2.beta_est; beta_est];
save(s2_path, 's2');
fprintf('Appended lambda=%d to identified_stage2.mat\n', lambda_to_run);
fprintf('Points collected so far: %d / 5\n', numel(s2.lambda));
fprintf('\n');
if numel(s2.lambda) >= 3
    fprintf('Enough points collected. Run id_stage2_fit.m to fit beta.\n');
end
