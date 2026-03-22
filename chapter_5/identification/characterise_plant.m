%% characterise_plant.m  --  Chapter 5: vLLM plant characterisation
%
% Standalone -- no dependency on setup_plant.m or any workspace variable.
%
% PURPOSE
% -------
% Chapter 3 taught us that Ollama has no real request queue: q≈0 always.
% Chapter 5 uses vLLM, which has a proper continuous-batching scheduler
% with an observable queue exposed at /metrics.  Before designing the
% cascade controller we need to confirm the plant model holds:
%
%   l[k] = alpha*B[k] + gamma*B[k]^2 + beta*q[k]
%
% and identify alpha, gamma, beta from real hardware.
%
% TWO LATENCY METRICS -- used deliberately in different stages:
%
%   Stage 2 uses TTFT (vllm_ttft, num_predict=1):
%     Measures pure GPU concurrency cost with no queueing.
%     Requests complete in ~20-100ms so q stays 0.  Correct for
%     fitting alpha and gamma cleanly at q=0.
%
%   Stages 3/4 use end-to-end (vllm_e2e, num_predict=e2e_tokens):
%     Each request occupies a vLLM scheduler slot for ~400ms (20 tokens
%     x ~20ms/token on 0.6B).  With 8 concurrent requests and
%     max_num_seqs=4, the 4 excess requests genuinely queue in vLLM's
%     scheduler.  End-to-end is also the correct production metric.
%
% TICK STRUCTURE (Stages 3 and 4):
%   At the START of each tick:  read q[k] from /metrics.
%   Then fire this tick's requests concurrently.
%   Then collect latencies (fetchOutputs blocks until all complete).
%   This mirrors exactly how the cascade controller will operate.
%   No pause() calls anywhere -- q[k] is the leftover from the
%   previous tick's requests, which have already completed.
%
% HOW TO RUN
% ----------
%   1. Start vLLM:  cd chapter_5 && ./start_vllm.sh --bg
%   2. Confirm up:  curl http://localhost:8001/health
%   3. Run this script from MATLAB
%
% STAGES
%   1 -- /metrics smoke test
%   2 -- B sweep at q=0          -> fits alpha, gamma
%   3 -- Sustained lambda > 4    -> fits beta from real queue
%   4 -- Operating envelope      -> (lambda, q_ss, l_ss, wall-clock)
%
% OUTPUTS
%   identified_params.mat        -- alpha, gamma, beta, R2
%   ch5_stage2_b_sweep.png
%   ch5_stage3_queue.png
%   ch5_stage4_envelope.png

clear; clc;

% -------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
id_dir  = fileparts(mfilename('fullpath'));
src_dir = fullfile(id_dir, '..', 'src');
addpath(src_dir);

fprintf('[init] id_dir  = %s\n', id_dir);
fprintf('[init] src_dir = %s\n', src_dir);

% -------------------------------------------------------------------------
% Configuration
% -------------------------------------------------------------------------
cfg.base_url    = 'http://localhost:8001';
cfg.model       = 'mlx-community/Qwen3-0.6B-4bit';
cfg.metrics_url = 'http://localhost:8001/metrics';
cfg.gen_url     = 'http://localhost:8001/v1/completions';
cfg.timeout     = 30;        % s per request

cfg.num_predict = 1;         % Stage 2: TTFT only
cfg.e2e_tokens  = 20;        % Stages 3/4: ~400ms per request -> queue builds

% Stage 2
cfg.b_sweep     = [1 2 3 4 5 6 8];
cfg.n_reps      = 5;

% Stage 3
cfg.lambda_build = 10;       % req/tick, must be > max_num_seqs (=4)
cfg.build_ticks  = 15;
cfg.meas_ticks3  = 10;

% Stage 4
cfg.lambda_sweep = [1 2 3 4 5 6 8 10];
cfg.settle_ticks = 15;
cfg.meas_ticks   = 10;
cfg.b_max_env    = 8;

cfg.prompts = {
    'What is 2+2?'
    'Name a colour.'
    'What is the capital of France?'
    'How many days in a week?'
    'Name a planet.'
};

fprintf('[config] model        = %s\n',   cfg.model);
fprintf('[config] b_sweep      = %s\n',   num2str(cfg.b_sweep));
fprintf('[config] n_reps       = %d\n',   cfg.n_reps);
fprintf('[config] e2e_tokens   = %d\n',   cfg.e2e_tokens);
fprintf('[config] lambda_build = %d\n',   cfg.lambda_build);
fprintf('[config] lambda_sweep = %s\n\n', num2str(cfg.lambda_sweep));

% =========================================================================
% STAGE 1 -- Metrics smoke test
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 1: vLLM metrics smoke test\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

try
    webread([cfg.base_url '/health'], weboptions('Timeout', 5));
    fprintf('[stage1] Health: OK\n');
catch e
    error('vLLM not reachable at %s -- run ./start_vllm.sh --bg\n%s', cfg.base_url, e.message);
end

try
    raw = webread(cfg.metrics_url, weboptions('Timeout', 5, 'ContentType', 'text'));
    m   = parse_vllm_metrics(raw);
    show = {'vllm:num_requests_running', 'vllm:num_requests_waiting'};
    for i = 1:numel(show)
        fn = matlab.lang.makeValidName(show{i});
        if isfield(m, fn)
            fprintf('[stage1]  %-38s = %g\n', show{i}, m.(fn));
        else
            fprintf('[stage1]  %-38s = (not found)\n', show{i});
        end
    end
    fprintf('\n[stage1] Metrics endpoint OK.\n\n');
catch e
    fprintf('[stage1] WARNING: metrics error: %s\n\n', e.message);
end

% =========================================================================
% STAGE 2 -- B sweep at q=0  (fit alpha, gamma)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 2: B sweep at q=0  (fitting alpha, gamma)  [TTFT]\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', min(max(cfg.b_sweep)+2, feature('numcores')));
end
addAttachedFiles(gcp(), src_dir);
fprintf('[stage2] Pool: %d workers\n\n', gcp().NumWorkers);

% Warm up
fprintf('[stage2] Warming up (3 requests)...\n');
for w = 1:3
    f = parfeval(@vllm_ttft, 1, cfg.gen_url, cfg.model, 'Hello', cfg.num_predict, cfg.timeout);
    try; lat = fetchOutputs(f); fprintf('  warmup %d: %.0f ms\n', w, lat);
    catch e; fprintf('  warmup %d: error (%s)\n', w, e.message); end
end
fprintf('[stage2] Done.\n\n');

n_b      = numel(cfg.b_sweep);
l_mean_b = zeros(n_b, 1);
l_std_b  = zeros(n_b, 1);

for bi = 1:n_b
    b = cfg.b_sweep(bi);
    fprintf('[stage2] B = %2d  (%d reps)...\n', b, cfg.n_reps);
    rep_means = zeros(cfg.n_reps, 1);
    for r = 1:cfg.n_reps
        futures = cell(b, 1);
        for i = 1:b
            prompt = cfg.prompts{mod(i-1, numel(cfg.prompts))+1};
            futures{i} = parfeval(@vllm_ttft, 1, ...
                cfg.gen_url, cfg.model, prompt, cfg.num_predict, cfg.timeout);
        end
        lats = zeros(1, b);
        for i = 1:b
            try; lats(i) = fetchOutputs(futures{i}); catch; lats(i) = NaN; end
        end
        rep_means(r) = mean(lats, 'omitnan');
        fprintf('  rep %d: %.1f ms  [%s]\n', r, rep_means(r), num2str(round(lats), '%d '));
    end
    l_mean_b(bi) = mean(rep_means, 'omitnan');
    l_std_b(bi)  = std(rep_means,  'omitnan');
    fprintf('  --> B=%2d: %.1f ± %.1f ms\n\n', b, l_mean_b(bi), l_std_b(bi));
end

% Fit l(B) = alpha*B + gamma*B^2 (no intercept)
valid    = isfinite(l_mean_b) & l_mean_b > 0;
B_v      = cfg.b_sweep(valid)';
l_v      = l_mean_b(valid);
params2  = [B_v, B_v.^2] \ l_v;
alpha_id = params2(1);
gamma_id = params2(2);
l_fit2   = [B_v, B_v.^2] * params2;
r2_s2    = 1 - sum((l_v - l_fit2).^2) / sum((l_v - mean(l_v)).^2);

fprintf('[stage2] alpha = %.4f ms/req\n',   alpha_id);
fprintf('[stage2] gamma = %.4f ms/req^2\n', gamma_id);
fprintf('[stage2] R^2   = %.4f\n\n',        r2_s2);

fig2 = figure('Name','Stage 2','Visible','off');
errorbar(cfg.b_sweep, l_mean_b, l_std_b, 'bo', 'MarkerFaceColor','b', 'LineWidth',1.2);
hold on;
b_fine = linspace(0.5, max(cfg.b_sweep)+0.5, 200);
plot(b_fine, alpha_id*b_fine + gamma_id*b_fine.^2, 'r-', 'LineWidth', 2);
xlabel('B [req]'); ylabel('TTFT [ms]');
title(sprintf('Stage 2 — B sweep (q=0)\nalpha=%.3f  gamma=%.4f  R^2=%.3f', alpha_id, gamma_id, r2_s2));
legend('Measured','\alpha B + \gamma B^2','Location','northwest'); grid on;
saveas(fig2, fullfile(id_dir, 'ch5_stage2_b_sweep.png'));
fprintf('[stage2] Plot saved.\n\n');

% =========================================================================
% STAGE 3 -- Queue buildup  (fit beta)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 3: Queue buildup  (fitting beta)  [e2e, %d tokens]\n', cfg.e2e_tokens);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% Tick structure:
%   1. Read q[k] from /metrics  (leftover queue from previous tick)
%   2. Fire b_k requests via parfeval
%   3. Collect latencies (fetchOutputs blocks until all done)
%   No pause() needed -- q[k] is the state at the start of the tick.

fprintf('[stage3] %d req/tick, %d build + %d meas ticks...\n\n', ...
    cfg.lambda_build, cfg.build_ticks, cfg.meas_ticks3);

n3       = cfg.build_ticks + cfg.meas_ticks3;
q_log3   = zeros(n3, 1);
b_log3   = zeros(n3, 1);
l_log3   = zeros(n3, 1);
lat_buf3 = 200 * ones(1, 10);
buf3     = 0;

for tick = 1:n3
    phase = 'build';
    if tick > cfg.build_ticks; phase = 'MEAS '; end
    b_k = min(cfg.lambda_build, cfg.b_max_env);

    % --- 1. Read q[k] at start of tick ---
    q_k = 0;
    try
        raw = webread(cfg.metrics_url, weboptions('Timeout',3,'ContentType','text'));
        ms  = parse_vllm_metrics(raw);
        fn  = matlab.lang.makeValidName('vllm:num_requests_waiting');
        if isfield(ms, fn); q_k = ms.(fn); end
    catch; end

    % --- 2. Fire requests ---
    futures = cell(b_k, 1);
    for i = 1:b_k
        prompt = cfg.prompts{mod(i-1, numel(cfg.prompts))+1};
        futures{i} = parfeval(@vllm_e2e, 1, ...
            cfg.gen_url, cfg.model, prompt, cfg.e2e_tokens, cfg.timeout);
    end

    % --- 3. Collect latencies ---
    lats = zeros(1, b_k);
    for i = 1:b_k
        try; lats(i) = fetchOutputs(futures{i}); catch; lats(i) = NaN; end
    end
    for i = 1:b_k
        if ~isnan(lats(i))
            buf3 = mod(buf3, 10) + 1;
            lat_buf3(buf3) = lats(i);
        end
    end

    q_log3(tick) = q_k;
    b_log3(tick) = b_k;
    l_log3(tick) = mean(lat_buf3);
    fprintf('  tick %3d [%s]  q=%4.1f  b=%d  l=%.0f ms\n', tick, phase, q_k, b_k, l_log3(tick));
end

meas3 = (cfg.build_ticks+1):n3;
q_ss3 = mean(q_log3(meas3));
b_ss3 = mean(b_log3(meas3));
l_ss3 = mean(l_log3(meas3));
svc3  = alpha_id * b_ss3 + gamma_id * b_ss3^2;

if q_ss3 > 0.1
    beta_id = (l_ss3 - svc3) / q_ss3;
    fprintf('\n[stage3] q_ss=%.2f  b_ss=%.2f  l_ss=%.0f ms  svc=%.0f ms\n', ...
        q_ss3, b_ss3, l_ss3, svc3);
    fprintf('[stage3] beta = %.4f ms/req\n\n', beta_id);
else
    beta_id = NaN;
    fprintf('\n[stage3] WARNING: q_ss=%.2f -- queue did not build.\n', q_ss3);
    fprintf('[stage3] Increase cfg.lambda_build or decrease max_num_seqs in start_vllm.sh.\n\n');
end

fig3 = figure('Name','Stage 3','Visible','off');
plot(1:n3, q_log3, 'b-o', 'LineWidth', 1.5);
hold on;
xline(cfg.build_ticks + 0.5, 'r--', 'Measurement window');
xlabel('Tick'); ylabel('num\_requests\_waiting');
title('Stage 3 — Queue buildup'); grid on;
saveas(fig3, fullfile(id_dir, 'ch5_stage3_queue.png'));
fprintf('[stage3] Plot saved.\n\n');

% =========================================================================
% STAGE 4 -- Operating envelope
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 4: Operating envelope  [e2e, %d tokens]\n', cfg.e2e_tokens);
fprintf('═══════════════════════════════════════════════════════════════\n\n');

n_lam    = numel(cfg.lambda_sweep);
env_q    = zeros(n_lam, 1);
env_b    = zeros(n_lam, 1);
env_l    = zeros(n_lam, 1);
env_wall = zeros(n_lam, 1);

for li = 1:n_lam
    lam = cfg.lambda_sweep(li);
    b_k = min(lam, cfg.b_max_env);
    n4  = cfg.settle_ticks + cfg.meas_ticks;
    fprintf('[stage4] lambda=%d  b=%d  (%d settle + %d meas)...\n', ...
        lam, b_k, cfg.settle_ticks, cfg.meas_ticks);

    lat_buf4 = 200*ones(1,10); buf4 = 0;
    q_all = zeros(n4,1); l_all = zeros(n4,1); w_all = zeros(n4,1);

    for tick = 1:n4
        phase = 'settle';
        if tick > cfg.settle_ticks; phase = 'MEAS  '; end

        % --- 1. Read q[k] ---
        q_k4 = 0;
        try
            raw = webread(cfg.metrics_url, weboptions('Timeout',3,'ContentType','text'));
            ms  = parse_vllm_metrics(raw);
            fn  = matlab.lang.makeValidName('vllm:num_requests_waiting');
            if isfield(ms, fn); q_k4 = ms.(fn); end
        catch; end

        % --- 2. Fire requests ---
        t_tick  = tic;
        futures = cell(b_k, 1);
        for i = 1:b_k
            prompt = cfg.prompts{mod(i-1,numel(cfg.prompts))+1};
            futures{i} = parfeval(@vllm_e2e, 1, ...
                cfg.gen_url, cfg.model, prompt, cfg.e2e_tokens, cfg.timeout);
        end

        % --- 3. Collect ---
        lats4 = zeros(1,b_k);
        for i = 1:b_k
            try; lats4(i) = fetchOutputs(futures{i}); catch; lats4(i) = NaN; end
        end
        wall4 = toc(t_tick) * 1000;
        for i = 1:b_k
            if ~isnan(lats4(i)); buf4 = mod(buf4,10)+1; lat_buf4(buf4) = lats4(i); end
        end

        q_all(tick) = q_k4;
        l_all(tick) = mean(lat_buf4);
        w_all(tick) = wall4;
        fprintf('  tick %2d [%s]  q=%4.1f  b=%d  l=%.0f ms  wall=%.0f ms\n', ...
            tick, phase, q_k4, b_k, l_all(tick), wall4);
    end

    meas4        = (cfg.settle_ticks+1):n4;
    env_q(li)    = mean(q_all(meas4));
    env_l(li)    = mean(l_all(meas4));
    env_b(li)    = b_k;
    env_wall(li) = mean(w_all(meas4));
    fprintf('  --> q_ss=%.2f  l_ss=%.0f ms  wall=%.0f ms\n\n', ...
        env_q(li), env_l(li), env_wall(li));
end

fig4 = figure('Name','Stage 4','Visible','off','Position',[100 100 1100 380]);
subplot(1,3,1);
plot(cfg.lambda_sweep, env_l, 'b-o','LineWidth',1.5,'MarkerFaceColor','b');
xlabel('\lambda [req/tick]'); ylabel('l_{ss} [ms]'); title('Latency vs lambda'); grid on;
subplot(1,3,2);
plot(cfg.lambda_sweep, env_q, 'r-o','LineWidth',1.5,'MarkerFaceColor','r');
xlabel('\lambda [req/tick]'); ylabel('q_{ss} [req]'); title('Queue depth vs lambda'); grid on;
subplot(1,3,3);
plot(cfg.lambda_sweep, env_wall, 'g-o','LineWidth',1.5,'MarkerFaceColor','g');
yline(1000,'k--','dt=1s'); xlabel('\lambda [req/tick]'); ylabel('Wall [ms]');
title('Tick wall-clock vs lambda'); grid on;
saveas(fig4, fullfile(id_dir, 'ch5_stage4_envelope.png'));
fprintf('[stage4] Plot saved.\n\n');

% =========================================================================
% SUMMARY
% =========================================================================
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  IDENTIFIED PARAMETERS  (%s)\n', cfg.model);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  alpha = %8.4f  ms/req      R^2 = %.4f\n', alpha_id, r2_s2);
fprintf('║  gamma = %8.4f  ms/req^2    R^2 = %.4f\n', gamma_id, r2_s2);
if ~isnan(beta_id)
    fprintf('║  beta  = %8.4f  ms/req\n', beta_id);
else
    fprintf('║  beta  =      NaN         (q stayed 0 in stage3)\n');
end
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Operating envelope:\n');
for li = 1:n_lam
    ov = '';
    if env_wall(li) > 1000; ov = '  *** OVERRUN'; end
    fprintf('║    lambda=%2d  q=%4.1f  l=%6.0f ms  wall=%4.0f ms%s\n', ...
        cfg.lambda_sweep(li), env_q(li), env_l(li), env_wall(li), ov);
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
identified.envelope.wall_ms= env_wall';
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
function metrics = parse_vllm_metrics(raw_text)
%PARSE_VLLM_METRICS  Strip Prometheus label blocks then parse name/value.
%
%   vLLM lines look like:
%     vllm:num_requests_waiting{engine="0",model_name="..."} 2.0
%   We strip the {...} block, split on whitespace, and accumulate
%   values across duplicate keys (multiple engines).
    metrics = struct();
    lines   = strsplit(raw_text, newline);
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if isempty(line); continue; end
        if line(1) == '#'; continue; end
        clean = strtrim(regexprep(line, '\{[^}]*\}', ''));
        if isempty(clean); continue; end
        parts = regexp(clean, '\s+', 'split');
        if numel(parts) < 2; continue; end
        val = str2double(parts{2});
        if isnan(val); continue; end
        f = matlab.lang.makeValidName(parts{1});
        if isfield(metrics, f)
            metrics.(f) = metrics.(f) + val;
        else
            metrics.(f) = val;
        end
    end
end
