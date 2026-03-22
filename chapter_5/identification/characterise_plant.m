%% characterise_plant.m  --  Chapter 5: vLLM plant characterisation
%
% Standalone -- no dependency on setup_plant.m or any workspace variable.
%
% PURPOSE
% -------
% Chapter 3 taught us that Ollama has no real request queue: q≈0 always
% because Ollama dispatches requests immediately rather than scheduling them.
% Chapter 5 uses vLLM, which has a proper continuous-batching scheduler with
% an observable queue.  Before designing the cascade controller we need to
% confirm:
%
%   1. That num_requests_waiting > 0 under load (queue exists and is real)
%   2. The shape of TTFT(B, q): does beta*q actually appear in the data?
%   3. The values of alpha, gamma, beta for the real vLLM plant
%   4. The safe operating envelope: B_max, lambda_max, q_max
%
% WHAT THIS SCRIPT DOES
% ----------------------
% Stage 1 -- Metrics endpoint smoke test
%   Confirm vLLM is up, parse /metrics, print key gauges.
%
% Stage 2 -- B sweep at q=0 (fit alpha, gamma)
%   Fire B concurrent requests when the server is idle (queue empty).
%   TTFT(B, q=0) = alpha*B + gamma*B^2.
%   Uses parfeval so all B requests hit the GPU simultaneously.
%
% Stage 3 -- Queue buildup sweep (fit beta)
%   Sustain arrival rate lambda > max_num_seqs for several ticks to force
%   num_requests_waiting > 0, then measure TTFT at known (B, q).
%   Residual latency above the service term = beta * q_measured.
%
% Stage 4 -- Operating envelope
%   Sweep lambda from 1 to lambda_max, let queue settle at each value,
%   record steady-state (lambda, B_ss, q_ss, TTFT_ss).
%   Identifies the controllable region for the cascade controller.
%
% HOW TO RUN
% ----------
%   1. Start vLLM:  cd chapter_5 && ./start_vllm.sh --bg
%   2. Confirm up:  curl http://localhost:8001/health
%   3. Run this script from MATLAB
%
% OUTPUTS
%   identified_params.mat      -- alpha, gamma, beta, R2 values
%   ch5_stage1_metrics.png     -- metrics snapshot plot
%   ch5_stage2_b_sweep.png     -- TTFT(B) at q=0
%   ch5_stage3_queue.png       -- residual latency vs q
%   ch5_stage4_envelope.png    -- operating envelope

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
% Configuration -- all self-contained, no external structs
% -------------------------------------------------------------------------
cfg.base_url     = 'http://localhost:8001';          % vLLM server
cfg.model        = 'mlx-community/Qwen3-0.6B-4bit';
cfg.metrics_url  = 'http://localhost:8001/metrics';
cfg.gen_url      = 'http://localhost:8001/v1/completions';
cfg.timeout      = 30;      % s per request
cfg.num_predict  = 1;       % tokens to generate (minimise gen time for TTFT)
cfg.dt           = 1.0;     % s -- tick duration (matches controller)

% Stage 2: B sweep
cfg.b_sweep      = [1 2 3 4 5 6 8];
cfg.n_reps       = 5;       % repetitions per B

% Stage 3: queue buildup
%   Set lambda well above max_num_seqs=4 to force waiting requests
cfg.lambda_build = 10;      % req/tick -- high enough to fill queue
cfg.build_ticks  = 15;      % ticks to build queue before measuring
cfg.n_queue_meas = 10;      % measurement ticks at saturated queue

% Stage 4: operating envelope
cfg.lambda_sweep = [1 2 3 4 5 6 8 10];
cfg.settle_ticks = 15;
cfg.meas_ticks   = 10;
cfg.b_max_env    = 8;       % hard cap for envelope experiment

cfg.prompts = {
    'What is 2+2?'
    'Name a colour.'
    'What is the capital of France?'
    'How many days in a week?'
    'Name a planet.'
};

fprintf('[config] vLLM url    = %s\n', cfg.base_url);
fprintf('[config] model       = %s\n', cfg.model);
fprintf('[config] b_sweep     = %s\n', num2str(cfg.b_sweep));
fprintf('[config] n_reps      = %d\n', cfg.n_reps);
fprintf('[config] lambda_build= %d req/tick\n', cfg.lambda_build);
fprintf('[config] lambda_sweep= %s\n\n', num2str(cfg.lambda_sweep));

% =========================================================================
% STAGE 1 -- Metrics endpoint smoke test
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 1: vLLM metrics smoke test\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('[stage1] Checking health endpoint...\n');
try
    health = webread([cfg.base_url '/health'], weboptions('Timeout', 5));
    fprintf('[stage1] Health: OK\n');
catch e
    error('[stage1] vLLM not reachable at %s\nStart it with: ./start_vllm.sh --bg\nError: %s', ...
        cfg.base_url, e.message);
end

fprintf('[stage1] Fetching /metrics ...\n');
try
    raw_metrics = webread(cfg.metrics_url, weboptions('Timeout', 5, 'ContentType', 'text'));
    metrics = parse_vllm_metrics(raw_metrics);

    % Print key scheduler metrics
    keys_to_show = {'vllm:num_requests_running', 'vllm:num_requests_waiting', ...
                    'vllm:num_requests_swapped', 'vllm:gpu_cache_usage_perc', ...
                    'vllm:cpu_cache_usage_perc'};
    fprintf('[stage1] Key metrics (idle state):\n');
    for i = 1:numel(keys_to_show)
        k = keys_to_show{i};
        if isfield(metrics, matlab.lang.makeValidName(k))
            val = metrics.(matlab.lang.makeValidName(k));
            fprintf('  %-45s = %g\n', k, val);
        else
            fprintf('  %-45s = (not found)\n', k);
        end
    end
    fprintf('\n[stage1] Metrics endpoint confirmed working.\n\n');
catch e
    fprintf('[stage1] WARNING: metrics endpoint error: %s\n', e.message);
    fprintf('[stage1] Continuing -- will use wall-clock TTFT for identification.\n\n');
end

% =========================================================================
% STAGE 2 -- B sweep at q=0  (fit alpha, gamma)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 2: B sweep at q=0  (fitting alpha, gamma)\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% Open parallel pool for concurrent requests
pool = gcp('nocreate');
if isempty(pool)
    fprintf('[stage2] Starting parallel pool...\n');
    parpool('local', min(max(cfg.b_sweep)+2, feature('numcores')));
end
addAttachedFiles(gcp(), src_dir);
fprintf('[stage2] Pool ready: %d workers\n\n', gcp().NumWorkers);

% Warm up vLLM (load model weights, fill KV cache)
fprintf('[stage2] Warming up vLLM (3 requests)...\n');
for w = 1:3
    f = parfeval(@vllm_ttft, 1, cfg.gen_url, cfg.model, 'Hello', cfg.num_predict, cfg.timeout);
    try
        lat = fetchOutputs(f);
        fprintf('  warmup %d: %.0f ms\n', w, lat);
    catch e
        fprintf('  warmup %d: error (%s)\n', w, e.message);
    end
end
fprintf('[stage2] Warm-up done.\n\n');

n_b      = numel(cfg.b_sweep);
l_mean_b = zeros(n_b, 1);
l_std_b  = zeros(n_b, 1);
q_during = zeros(n_b, 1);   % queue depth observed during burst

for bi = 1:n_b
    b = cfg.b_sweep(bi);
    fprintf('[stage2] B = %2d  (%d reps x %d concurrent)...\n', b, cfg.n_reps, b);

    rep_means = zeros(cfg.n_reps, 1);

    for r = 1:cfg.n_reps
        % Submit B requests simultaneously
        futures = cell(b, 1);
        for i = 1:b
            prompt = cfg.prompts{mod(i-1, numel(cfg.prompts))+1};
            futures{i} = parfeval(@vllm_ttft, 1, ...
                cfg.gen_url, cfg.model, prompt, cfg.num_predict, cfg.timeout);
        end

        % Read queue depth immediately after submitting (before collecting)
        % This gives the num_requests_waiting at peak load
        pause(0.05);
        try
            raw = webread(cfg.metrics_url, weboptions('Timeout',3,'ContentType','text'));
            m   = parse_vllm_metrics(raw);
            fn  = matlab.lang.makeValidName('vllm:num_requests_waiting');
            if isfield(m, fn)
                q_during(bi) = max(q_during(bi), m.(fn));
            end
        catch
            % metrics read failed, skip
        end

        % Collect latencies
        lats = zeros(1, b);
        for i = 1:b
            try
                lats(i) = fetchOutputs(futures{i});
            catch
                lats(i) = NaN;
            end
        end

        rep_means(r) = mean(lats, 'omitnan');
        fprintf('  rep %d: mean=%.1f ms  q_peak=%g  [%s]\n', ...
            r, rep_means(r), q_during(bi), num2str(round(lats), '%d '));
    end

    l_mean_b(bi) = mean(rep_means, 'omitnan');
    l_std_b(bi)  = std(rep_means,  'omitnan');
    fprintf('  --> B=%2d: %.1f ± %.1f ms  (peak q=%g)\n\n', ...
        b, l_mean_b(bi), l_std_b(bi), q_during(bi));
end

% Fit l(B) = alpha*B + gamma*B^2  (no intercept -- TTFT→0 as B→0)
fprintf('[stage2] Fitting l = alpha*B + gamma*B^2 ...\n');
valid    = isfinite(l_mean_b) & l_mean_b > 0;
B_valid  = cfg.b_sweep(valid)';
l_valid  = l_mean_b(valid);
A2       = [B_valid, B_valid.^2];
params2  = A2 \ l_valid;
alpha_id = params2(1);
gamma_id = params2(2);
l_fit2   = A2 * params2;
r2_s2    = 1 - sum((l_valid-l_fit2).^2)/sum((l_valid-mean(l_valid)).^2);

fprintf('[stage2] alpha = %.4f  ms/req\n',   alpha_id);
fprintf('[stage2] gamma = %.4f  ms/req^2\n', gamma_id);
fprintf('[stage2] R^2   = %.4f\n\n',         r2_s2);

% Plot
fig2 = figure('Name','Stage 2','Visible','off');
errorbar(cfg.b_sweep, l_mean_b, l_std_b, 'bo', 'MarkerFaceColor','b', 'LineWidth',1.2);
hold on;
b_fine = linspace(0.5, max(cfg.b_sweep)+0.5, 200);
plot(b_fine, alpha_id*b_fine + gamma_id*b_fine.^2, 'r-', 'LineWidth',2);
xlabel('B [req]'); ylabel('TTFT [ms]');
title(sprintf('Stage 2 — B sweep at q=0\n\\alpha=%.3f  \\gamma=%.4f  R^2=%.3f', ...
    alpha_id, gamma_id, r2_s2));
legend('Measured','\\alpha B + \\gamma B^2','Location','northwest');
grid on;
saveas(fig2, fullfile(id_dir, 'ch5_stage2_b_sweep.png'));
fprintf('[stage2] Plot saved.\n\n');

% =========================================================================
% STAGE 3 -- Queue buildup: identify beta
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 3: Queue buildup sweep  (fitting beta)\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

% Strategy: fire cfg.lambda_build concurrent requests per tick continuously.
% vLLM's scheduler caps at max_num_seqs=4 running simultaneously, so
% the remaining requests pile up in num_requests_waiting.
% We measure TTFT at known (B, q) triplets and extract beta from:
%   l_ss - (alpha*B_ss + gamma*B_ss^2) = beta * q_ss

fprintf('[stage3] Firing %d req/tick for %d build ticks then %d meas ticks...\n', ...
    cfg.lambda_build, cfg.build_ticks, cfg.n_queue_meas);

q_log3 = zeros(cfg.build_ticks + cfg.n_queue_meas, 1);
b_log3 = zeros(cfg.build_ticks + cfg.n_queue_meas, 1);
l_log3 = zeros(cfg.build_ticks + cfg.n_queue_meas, 1);
lat_buf3 = 200 * ones(1, 10);
buf_idx3 = 0;

for tick = 1:(cfg.build_ticks + cfg.n_queue_meas)
    phase = 'build';
    if tick > cfg.build_ticks; phase = 'MEAS '; end

    b_k = min(cfg.lambda_build, cfg.b_max_env);

    % Fire b_k requests concurrently
    futures = cell(b_k, 1);
    for i = 1:b_k
        prompt = cfg.prompts{mod(i-1, numel(cfg.prompts))+1};
        futures{i} = parfeval(@vllm_ttft, 1, ...
            cfg.gen_url, cfg.model, prompt, cfg.num_predict, cfg.timeout);
    end

    % Read queue depth from metrics immediately after submitting
    pause(0.05);
    q_meas = 0;
    try
        raw = webread(cfg.metrics_url, weboptions('Timeout',3,'ContentType','text'));
        m   = parse_vllm_metrics(raw);
        fn  = matlab.lang.makeValidName('vllm:num_requests_waiting');
        if isfield(m, fn); q_meas = m.(fn); end
    catch; end

    % Collect latencies
    lats = zeros(1, b_k);
    for i = 1:b_k
        try; lats(i) = fetchOutputs(futures{i}); catch; lats(i) = NaN; end
    end
    for i = 1:b_k
        if ~isnan(lats(i))
            buf_idx3 = mod(buf_idx3, 10) + 1;
            lat_buf3(buf_idx3) = lats(i);
        end
    end
    l_meas = mean(lat_buf3);

    q_log3(tick) = q_meas;
    b_log3(tick) = b_k;
    l_log3(tick) = l_meas;

    fprintf('  tick %3d [%s] b=%d q=%.1f l=%.1f ms\n', tick, phase, b_k, q_meas, l_meas);
end

% Fit beta from measurement window
meas_idx3 = (cfg.build_ticks+1):(cfg.build_ticks+cfg.n_queue_meas);
q_ss3 = mean(q_log3(meas_idx3));
b_ss3 = mean(b_log3(meas_idx3));
l_ss3 = mean(l_log3(meas_idx3));

svc3     = alpha_id * b_ss3 + gamma_id * b_ss3^2;
if q_ss3 > 0.1
    beta_id = (l_ss3 - svc3) / q_ss3;
    r2_s3   = NaN;   % single-point fit
    fprintf('\n[stage3] q_ss=%.2f  b_ss=%.2f  l_ss=%.1f ms  svc=%.1f ms\n', ...
        q_ss3, b_ss3, l_ss3, svc3);
    fprintf('[stage3] beta = %.4f ms/req\n\n', beta_id);
else
    beta_id = NaN;
    fprintf('\n[stage3] WARNING: q_ss=%.2f (too small, beta not identifiable)\n', q_ss3);
    fprintf('[stage3] vLLM may be draining queue faster than expected.\n');
    fprintf('[stage3] Try increasing cfg.lambda_build or decreasing max_num_seqs.\n\n');
    r2_s3 = NaN;
end

% Plot
fig3 = figure('Name','Stage 3','Visible','off');
plot(1:(cfg.build_ticks+cfg.n_queue_meas), q_log3, 'b-o','LineWidth',1.5);
hold on;
xline(cfg.build_ticks+0.5, 'r--', 'Measurement window', 'LabelHorizontalAlignment','left');
xlabel('Tick'); ylabel('num\_requests\_waiting  [req]');
title('Stage 3 — Queue buildup under sustained load');
grid on;
saveas(fig3, fullfile(id_dir, 'ch5_stage3_queue.png'));
fprintf('[stage3] Plot saved.\n\n');

% =========================================================================
% STAGE 4 -- Operating envelope
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 4: Operating envelope sweep\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

n_lam     = numel(cfg.lambda_sweep);
env_q     = zeros(n_lam, 1);
env_b     = zeros(n_lam, 1);
env_l     = zeros(n_lam, 1);
env_wall  = zeros(n_lam, 1);

for li = 1:n_lam
    lam = cfg.lambda_sweep(li);
    b_k = min(lam, cfg.b_max_env);
    fprintf('[stage4] lambda=%d  (b=%d per tick, %d settle + %d meas ticks)...\n', ...
        lam, b_k, cfg.settle_ticks, cfg.meas_ticks);

    lat_buf4 = 200*ones(1,10); buf_idx4 = 0;
    q_all4   = zeros(cfg.settle_ticks+cfg.meas_ticks, 1);
    l_all4   = zeros(cfg.settle_ticks+cfg.meas_ticks, 1);
    w_all4   = zeros(cfg.settle_ticks+cfg.meas_ticks, 1);

    for tick = 1:(cfg.settle_ticks+cfg.meas_ticks)
        futures = cell(b_k, 1);
        t_tick  = tic;
        for i = 1:b_k
            prompt = cfg.prompts{mod(i-1,numel(cfg.prompts))+1};
            futures{i} = parfeval(@vllm_ttft, 1, ...
                cfg.gen_url, cfg.model, prompt, cfg.num_predict, cfg.timeout);
        end
        pause(0.05);
        q_meas4 = 0;
        try
            raw = webread(cfg.metrics_url,weboptions('Timeout',3,'ContentType','text'));
            m   = parse_vllm_metrics(raw);
            fn  = matlab.lang.makeValidName('vllm:num_requests_waiting');
            if isfield(m,fn); q_meas4 = m.(fn); end
        catch; end

        lats4 = zeros(1,b_k);
        for i = 1:b_k
            try; lats4(i) = fetchOutputs(futures{i}); catch; lats4(i) = NaN; end
        end
        wall4 = toc(t_tick)*1000;

        for i = 1:b_k
            if ~isnan(lats4(i))
                buf_idx4 = mod(buf_idx4,10)+1;
                lat_buf4(buf_idx4) = lats4(i);
            end
        end
        q_all4(tick) = q_meas4;
        l_all4(tick) = mean(lat_buf4);
        w_all4(tick) = wall4;

        phase4 = 'settle';
        if tick > cfg.settle_ticks; phase4 = 'MEAS  '; end
        fprintf('  tick %2d [%s] b=%d q=%.1f l=%.1f ms wall=%.0f ms\n', ...
            tick, phase4, b_k, q_meas4, mean(lat_buf4), wall4);
    end

    meas4 = (cfg.settle_ticks+1):(cfg.settle_ticks+cfg.meas_ticks);
    env_q(li)    = mean(q_all4(meas4));
    env_l(li)    = mean(l_all4(meas4));
    env_b(li)    = b_k;
    env_wall(li) = mean(w_all4(meas4));
    fprintf('  --> lambda=%d: q_ss=%.2f  l_ss=%.1f ms  wall=%.0f ms\n\n', ...
        lam, env_q(li), env_l(li), env_wall(li));
end

% Plot
fig4 = figure('Name','Stage 4','Visible','off','Position',[100 100 1000 400]);
subplot(1,3,1);
plot(cfg.lambda_sweep, env_l, 'b-o','LineWidth',1.5,'MarkerFaceColor','b');
xlabel('\lambda [req/tick]'); ylabel('Mean TTFT [ms]');
title('TTFT vs lambda'); grid on;

subplot(1,3,2);
plot(cfg.lambda_sweep, env_q, 'r-o','LineWidth',1.5,'MarkerFaceColor','r');
xlabel('\lambda [req/tick]'); ylabel('q\_ss [req]');
title('Queue depth vs lambda'); grid on;

subplot(1,3,3);
plot(cfg.lambda_sweep, env_wall, 'g-o','LineWidth',1.5,'MarkerFaceColor','g');
yline(1000, 'k--', 'dt=1s', 'LabelHorizontalAlignment','left');
xlabel('\lambda [req/tick]'); ylabel('Wall-clock [ms]');
title('Tick wall-clock vs lambda'); grid on;

saveas(fig4, fullfile(id_dir, 'ch5_stage4_envelope.png'));
fprintf('[stage4] Plot saved.\n\n');

% =========================================================================
% SUMMARY
% =========================================================================
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  IDENTIFIED PARAMETERS  (%s)\n', cfg.model);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  alpha = %8.4f  ms/req        R^2 stage2 = %.4f\n', alpha_id, r2_s2);
fprintf('║  gamma = %8.4f  ms/req^2      R^2 stage2 = %.4f\n', gamma_id, r2_s2);
if ~isnan(beta_id)
    fprintf('║  beta  = %8.4f  ms/req        (from stage3)\n',    beta_id);
else
    fprintf('║  beta  =      NaN              (q stayed 0 -- see notes)\n');
end
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Operating envelope (lambda sweep):\n');
for li = 1:n_lam
    flag = '';
    if env_wall(li) > 1000; flag = ' *** TICK OVERRUN'; end
    fprintf('║    lambda=%2d: q_ss=%4.1f  TTFT=%5.1f ms  wall=%4.0f ms%s\n', ...
        cfg.lambda_sweep(li), env_q(li), env_l(li), env_wall(li), flag);
end
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% =========================================================================
% SAVE
% =========================================================================
identified.alpha        = alpha_id;
identified.gamma        = gamma_id;
identified.beta         = beta_id;
identified.r2_stage2    = r2_s2;
identified.model        = cfg.model;
identified.timestamp    = char(datetime('now'));
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
fprintf('[save] Saved to: %s\n\n', out_path);

fprintf('Next step: load identified_params.mat in setup_plant.m\n');
fprintf('  perturbed.alpha = identified.alpha;\n');
fprintf('  perturbed.gamma = identified.gamma;\n');
fprintf('  perturbed.beta  = identified.beta;\n');

% =========================================================================
% =========================================================================
% HELPER FUNCTIONS
% =========================================================================
function metrics = parse_vllm_metrics(raw_text)
%PARSE_VLLM_METRICS  Parse Prometheus text format from vLLM /metrics endpoint.
    metrics = struct();
    lines   = strsplit(raw_text, newline);
    for i = 1:numel(lines)
        line = strtrim(lines{i});
        if isempty(line); continue; end
        if line(1) == '#'; continue; end
        if any(line == '{'); continue; end
        parts = strsplit(line, ' ');
        if numel(parts) < 2; continue; end
        val = str2double(parts{2});
        if isnan(val); continue; end
        metrics.(matlab.lang.makeValidName(parts{1})) = val;
    end
end
