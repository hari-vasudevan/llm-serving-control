%% characterise_plant.m  --  Chapter 5: vLLM plant characterisation
%
% Standalone -- no dependency on setup_plant.m.
%
% ROOT CAUSE OF PREVIOUS q=0 RESULTS
% ------------------------------------
% Blocking tick:  fire B requests → fetchOutputs (waits ALL ~2400ms) → read q
%   By the time the next tick starts, all requests have completed.
%   q is always 0 at tick boundaries. Beta is unidentifiable.
%
% Non-blocking tick (this script):
%   fire B requests → pause(dt=1s) → read q → fire next batch
%   After 1s, the first batch is still in flight (completes at ~2400ms).
%   vLLM has running=3, waiting=4 at the 1s tick boundary.
%   q is non-zero. Beta is identifiable.
%
% TICK STRUCTURE (Stages 3 and 4):
%   1. Read q[k] from /metrics  <- queue from previous tick's overflow
%   2. Fire b_k requests concurrently (non-blocking: do NOT fetchOutputs yet)
%   3. pause(dt) to tick the clock
%   4. Repeat
%   After all ticks: collect all futures, compute latency statistics.
%
% Stage 2 keeps the blocking structure (we WANT q=0 for alpha/gamma fit).
%
% HOW TO RUN
%   1. ./start_vllm.sh --bg
%   2. curl http://localhost:8001/health
%   3. Run this script from MATLAB

clear; clc;

id_dir  = fileparts(mfilename('fullpath'));
src_dir = fullfile(id_dir, '..', 'src');
addpath(src_dir);
fprintf('[init] id_dir  = %s\n', id_dir);
fprintf('[init] src_dir = %s\n\n', src_dir);

% -------------------------------------------------------------------------
% Configuration
% -------------------------------------------------------------------------
cfg.base_url    = 'http://localhost:8001';
cfg.model       = 'mlx-community/Qwen3-0.6B-4bit';
cfg.metrics_url = 'http://localhost:8001/metrics';
cfg.gen_url     = 'http://localhost:8001/v1/completions';
cfg.timeout     = 30;
cfg.dt          = 1.0;      % s -- tick clock for Stages 3/4

cfg.num_predict = 1;        % Stage 2: TTFT, q=0
cfg.e2e_tokens  = 20;       % Stages 3/4: ~250ms/req -> queue persists across ticks

cfg.b_sweep     = [1 2 3 4 5 6 8];
cfg.n_reps      = 5;

cfg.lambda_build = 10;      % req/tick for Stage 3 (> max_num_seqs=4)
cfg.b_max        = 8;       % worker pool cap
cfg.build_ticks  = 15;
cfg.meas_ticks3  = 10;

cfg.lambda_sweep = [1 2 3 4 6 8];
cfg.settle_ticks = 15;
cfg.meas_ticks4  = 10;

cfg.prompts = {'What is 2+2?'; 'Name a colour.'; 'Capital of France?'; 'Days in a week?'; 'Name a planet.'};

fprintf('[config] model        = %s\n',   cfg.model);
fprintf('[config] dt           = %.1fs\n',cfg.dt);
fprintf('[config] e2e_tokens   = %d\n',   cfg.e2e_tokens);
fprintf('[config] lambda_build = %d\n',   cfg.lambda_build);
fprintf('[config] lambda_sweep = %s\n',   num2str(cfg.lambda_sweep));
fprintf('[config] b_max        = %d\n\n', cfg.b_max);

% =========================================================================
% STAGE 1 -- Smoke test
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 1: Smoke test\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

try
    webread([cfg.base_url '/health'], weboptions('Timeout',5));
    fprintf('[stage1] Health: OK\n');
catch e
    error('vLLM not reachable: %s', e.message);
end

try
    m = read_metrics(cfg.metrics_url);
    fprintf('[stage1] num_requests_running = %g\n', safe_get(m,'vllm_num_requests_running'));
    fprintf('[stage1] num_requests_waiting = %g\n', safe_get(m,'vllm_num_requests_waiting'));
    fprintf('[stage1] Metrics OK.\n\n');
catch e
    fprintf('[stage1] Metrics warning: %s\n\n', e.message);
end

% =========================================================================
% STAGE 2 -- B sweep at q=0  (BLOCKING, TTFT)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 2: B sweep at q=0  [TTFT, blocking]\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', min(cfg.b_max+2, feature('numcores')));
end
addAttachedFiles(gcp(), src_dir);
fprintf('[stage2] Pool: %d workers\n\n', gcp().NumWorkers);

fprintf('[stage2] Warming up...\n');
for w = 1:3
    f = parfeval(@vllm_ttft, 1, cfg.gen_url, cfg.model, 'Hello', cfg.num_predict, cfg.timeout);
    try; lat = fetchOutputs(f); fprintf('  warmup %d: %.0fms\n',w,lat);
    catch e; fprintf('  warmup %d: error\n',w); end
end
fprintf('\n');

n_b = numel(cfg.b_sweep);
l_mean_b = zeros(n_b,1);
l_std_b  = zeros(n_b,1);

for bi = 1:n_b
    b = cfg.b_sweep(bi);
    fprintf('[stage2] B=%d (%d reps)...\n', b, cfg.n_reps);
    reps = zeros(cfg.n_reps,1);
    for r = 1:cfg.n_reps
        fut = cell(b,1);
        for i = 1:b
            fut{i} = parfeval(@vllm_ttft, 1, cfg.gen_url, cfg.model, ...
                cfg.prompts{mod(i-1,numel(cfg.prompts))+1}, cfg.num_predict, cfg.timeout);
        end
        lats = zeros(1,b);
        for i = 1:b
            try; lats(i) = fetchOutputs(fut{i}); catch; lats(i) = NaN; end
        end
        reps(r) = mean(lats,'omitnan');
        fprintf('  rep %d: %.1fms  [%s]\n', r, reps(r), num2str(round(lats),'%d '));
    end
    l_mean_b(bi) = mean(reps,'omitnan');
    l_std_b(bi)  = std(reps,'omitnan');
    fprintf('  --> %.1f ± %.1f ms\n\n', l_mean_b(bi), l_std_b(bi));
end

v = isfinite(l_mean_b) & l_mean_b > 0;
p2 = [cfg.b_sweep(v)', cfg.b_sweep(v)'.^2] \ l_mean_b(v);
alpha_id = p2(1); gamma_id = p2(2);
l_fit = [cfg.b_sweep(v)', cfg.b_sweep(v)'.^2] * p2;
r2_s2 = 1 - sum((l_mean_b(v)-l_fit).^2)/sum((l_mean_b(v)-mean(l_mean_b(v))).^2);

fprintf('[stage2] alpha = %.4f ms/req\n',   alpha_id);
fprintf('[stage2] gamma = %.4f ms/req^2\n', gamma_id);
fprintf('[stage2] R^2   = %.4f\n\n',        r2_s2);

fig2 = figure('Visible','off');
errorbar(cfg.b_sweep, l_mean_b, l_std_b, 'bo','MarkerFaceColor','b','LineWidth',1.2); hold on;
bf = linspace(0.5, max(cfg.b_sweep)+0.5, 200);
plot(bf, alpha_id*bf+gamma_id*bf.^2, 'r-','LineWidth',2);
xlabel('B [req]'); ylabel('TTFT [ms]');
title(sprintf('Stage 2 (q=0)\nalpha=%.3f  gamma=%.4f  R^2=%.3f',alpha_id,gamma_id,r2_s2));
grid on; saveas(fig2, fullfile(id_dir,'ch5_stage2_b_sweep.png'));
fprintf('[stage2] Plot saved.\n\n');

% =========================================================================
% STAGE 3 -- Queue buildup  (NON-BLOCKING tick, e2e)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 3: Queue buildup  [e2e %dtok, NON-BLOCKING dt=%.1fs]\n', cfg.e2e_tokens, cfg.dt);
fprintf('═══════════════════════════════════════════════════════════════\n\n');
fprintf('Tick structure: read q → fire b_k requests → pause(dt) → repeat\n');
fprintf('Requests fire faster than they complete -> queue builds in vLLM.\n\n');

n3    = cfg.build_ticks + cfg.meas_ticks3;
b_k3  = min(cfg.lambda_build, cfg.b_max);
q_log3 = zeros(n3,1);
all_fut3 = {};   % accumulate all futures -- collect at end

for tick = 1:n3
    phase = 'build';
    if tick > cfg.build_ticks; phase = 'MEAS '; end

    t_tick = tic;

    % 1. Read q[k] -- leftover from previous tick's requests still in vLLM
    q_k = 0;
    try
        m   = read_metrics(cfg.metrics_url);
        q_k = safe_get(m,'vllm_num_requests_waiting');
    catch; end

    % 2. Fire b_k3 requests (non-blocking)
    for i = 1:b_k3
        all_fut3{end+1} = parfeval(@vllm_e2e, 1, cfg.gen_url, cfg.model, ...
            cfg.prompts{mod(i-1,numel(cfg.prompts))+1}, cfg.e2e_tokens, cfg.timeout);
    end

    % 3. Tick clock: wait remaining dt
    elapsed = toc(t_tick);
    if elapsed < cfg.dt
        pause(cfg.dt - elapsed);
    end

    q_log3(tick) = q_k;
    fprintf('  tick %3d [%s]  q=%5.1f  b=%d  (wall=%.0fms)\n', ...
        tick, phase, q_k, b_k3, toc(t_tick)*1000);
end

% Collect all futures (blocks here, after ticks are done)
fprintf('\n[stage3] Collecting %d futures...\n', numel(all_fut3));
lats3 = zeros(1,numel(all_fut3));
for i = 1:numel(all_fut3)
    try; lats3(i) = fetchOutputs(all_fut3{i}); catch; lats3(i) = NaN; end
end

% Latency for measurement window = futures from last meas_ticks3 * b_k3 requests
n_meas_fut3 = cfg.meas_ticks3 * b_k3;
l_ss3 = mean(lats3(end-n_meas_fut3+1:end), 'omitnan');
q_ss3 = mean(q_log3(cfg.build_ticks+1:end));
b_ss3 = b_k3;
svc3  = alpha_id*b_ss3 + gamma_id*b_ss3^2;

fprintf('[stage3] q_ss=%.2f  b_ss=%d  l_ss=%.0fms  svc=%.0fms\n', q_ss3, b_ss3, l_ss3, svc3);

if q_ss3 > 0.1
    beta_id = (l_ss3 - svc3) / q_ss3;
    fprintf('[stage3] beta = %.4f ms/req\n\n', beta_id);
else
    beta_id = NaN;
    fprintf('[stage3] WARNING: q_ss still 0. Try larger lambda_build or smaller dt.\n\n');
end

fig3 = figure('Visible','off');
plot(1:n3, q_log3, 'b-o','LineWidth',1.5); hold on;
xline(cfg.build_ticks+0.5,'r--','Measurement window');
xlabel('Tick'); ylabel('num\_requests\_waiting');
title('Stage 3 — Queue buildup (non-blocking ticks)'); grid on;
saveas(fig3, fullfile(id_dir,'ch5_stage3_queue.png'));
fprintf('[stage3] Plot saved.\n\n');

% =========================================================================
% STAGE 4 -- Operating envelope  (NON-BLOCKING tick, e2e)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 4: Operating envelope  [e2e, NON-BLOCKING dt=%.1fs]\n', cfg.dt);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

n_lam    = numel(cfg.lambda_sweep);
env_q    = zeros(n_lam,1);
env_b    = zeros(n_lam,1);
env_l    = zeros(n_lam,1);

for li = 1:n_lam
    lam = cfg.lambda_sweep(li);
    b_k = min(lam, cfg.b_max);
    n4  = cfg.settle_ticks + cfg.meas_ticks4;
    fprintf('[stage4] lambda=%d  b=%d  (%d settle + %d meas)...\n', ...
        lam, b_k, cfg.settle_ticks, cfg.meas_ticks4);

    q_all4  = zeros(n4,1);
    all_fut4 = {};

    for tick = 1:n4
        phase = 'settle'; if tick > cfg.settle_ticks; phase = 'MEAS  '; end
        t_tick = tic;

        % 1. Read q[k]
        q_k4 = 0;
        try; m = read_metrics(cfg.metrics_url); q_k4 = safe_get(m,'vllm_num_requests_waiting'); catch; end

        % 2. Fire non-blocking
        for i = 1:b_k
            all_fut4{end+1} = parfeval(@vllm_e2e, 1, cfg.gen_url, cfg.model, ...
                cfg.prompts{mod(i-1,numel(cfg.prompts))+1}, cfg.e2e_tokens, cfg.timeout);
        end

        % 3. Tick clock
        elapsed = toc(t_tick);
        if elapsed < cfg.dt; pause(cfg.dt - elapsed); end

        q_all4(tick) = q_k4;
        fprintf('  tick %2d [%s]  q=%5.1f  b=%d\n', tick, phase, q_k4, b_k);
    end

    % Collect futures
    fprintf('  Collecting %d futures...\n', numel(all_fut4));
    lats4 = zeros(1,numel(all_fut4));
    for i = 1:numel(all_fut4)
        try; lats4(i) = fetchOutputs(all_fut4{i}); catch; lats4(i) = NaN; end
    end

    n_meas_fut4 = cfg.meas_ticks4 * b_k;
    meas4 = (cfg.settle_ticks+1):n4;
    env_q(li) = mean(q_all4(meas4));
    env_b(li) = b_k;
    env_l(li) = mean(lats4(end-n_meas_fut4+1:end), 'omitnan');
    fprintf('  --> q_ss=%.2f  l_ss=%.0fms\n\n', env_q(li), env_l(li));

    % Drain queue before next lambda: wait for all futures to complete
    fprintf('  Draining queue...\n');
    pause(5);  % let vLLM finish outstanding requests
end

fig4 = figure('Visible','off','Position',[100 100 900 350]);
subplot(1,2,1);
plot(cfg.lambda_sweep, env_l, 'b-o','LineWidth',1.5,'MarkerFaceColor','b');
xlabel('\lambda [req/tick]'); ylabel('l_{ss} [ms]'); title('Latency vs lambda'); grid on;
subplot(1,2,2);
plot(cfg.lambda_sweep, env_q, 'r-o','LineWidth',1.5,'MarkerFaceColor','r');
xlabel('\lambda [req/tick]'); ylabel('q_{ss} [req]'); title('Queue depth vs lambda'); grid on;
saveas(fig4, fullfile(id_dir,'ch5_stage4_envelope.png'));
fprintf('[stage4] Plot saved.\n\n');

% =========================================================================
% SUMMARY
% =========================================================================
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  IDENTIFIED PARAMETERS  (%s)\n', cfg.model);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  alpha = %8.4f  ms/req      R^2 = %.4f\n', alpha_id, r2_s2);
fprintf('║  gamma = %8.4f  ms/req^2\n', gamma_id);
if ~isnan(beta_id)
    fprintf('║  beta  = %8.4f  ms/req\n', beta_id);
else
    fprintf('║  beta  =      NaN\n');
end
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Operating envelope:\n');
for li = 1:n_lam
    fprintf('║    lambda=%2d  q=%5.1f  l=%6.0f ms\n', cfg.lambda_sweep(li), env_q(li), env_l(li));
end
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% =========================================================================
% SAVE
% =========================================================================
identified.alpha          = alpha_id;
identified.gamma          = gamma_id;
identified.beta           = beta_id;
identified.r2_stage2      = r2_s2;
identified.model          = cfg.model;
identified.timestamp      = char(datetime('now'));
identified.envelope.lambda = cfg.lambda_sweep;
identified.envelope.q_ss   = env_q';
identified.envelope.l_ss   = env_l';
identified.envelope.b_ss   = env_b';
identified.raw.b_sweep     = cfg.b_sweep;
identified.raw.l_mean_b    = l_mean_b;
identified.raw.l_std_b     = l_std_b;

out_path = fullfile(id_dir, 'identified_params.mat');
save(out_path, 'identified');
fprintf('[save] %s\n\n', out_path);
fprintf('Paste into setup_plant.m:\n');
fprintf('  perturbed.alpha = %.4f;\n', alpha_id);
fprintf('  perturbed.gamma = %.4f;\n', gamma_id);
if ~isnan(beta_id)
    fprintf('  perturbed.beta  = %.4f;\n', beta_id);
end

% =========================================================================
% HELPERS
% =========================================================================
function m = read_metrics(url)
    raw = webread(url, weboptions('Timeout',3,'ContentType','text'));
    m   = struct();
    for line = strsplit(raw, newline)
        s = strtrim(line{1});
        if isempty(s) || s(1)=='#'; continue; end
        clean = strtrim(regexprep(s,'\{[^}]*\}',''));
        p = regexp(clean,'\s+','split');
        if numel(p)<2; continue; end
        v = str2double(p{2});
        if isnan(v); continue; end
        f = matlab.lang.makeValidName(p{1});
        if isfield(m,f); m.(f)=m.(f)+v; else; m.(f)=v; end
    end
end

function v = safe_get(m, field)
    if isfield(m, field); v = m.(field); else; v = 0; end
end
