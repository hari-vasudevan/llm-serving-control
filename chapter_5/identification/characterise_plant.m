%% characterise_plant.m  --  Chapter 5: vLLM plant characterisation
%
% Standalone -- no dependency on setup_plant.m.
%
% WHY PREVIOUS ATTEMPTS FAILED
% -----------------------------
% We tried using MATLAB parfeval workers as a load generator.
% The problem: parfeval takes ~600ms to dispatch 8 calls because each
% worker has inter-process overhead (~75ms per call).  With requests
% completing in ~265ms each, by the time the last request reaches vLLM,
% the first ones have finished.  The queue never accumulates.
%
% THE FIX
% -------
% Separate load generation from measurement:
%   - load_gen.py runs in the background using Python threads.
%     Python can fire concurrent HTTP requests with near-zero overhead,
%     saturating vLLM's scheduler at any desired rate.
%   - MATLAB reads /metrics at each tick boundary to observe q[k] and l[k].
%   - MATLAB controls the experiment timing; Python drives the load.
%
% STAGES
%   1 -- /metrics smoke test  (confirms queue metrics are readable)
%   2 -- B sweep at q=0       (blocking parfeval, TTFT, fits alpha+gamma)
%   3 -- Queue buildup        (Python load at lambda > max_num_seqs, fits beta)
%   4 -- Operating envelope   (Python load at swept lambda values)
%
% HOW TO RUN
%   1. ./start_vllm.sh --bg
%   2. curl http://localhost:8001/health
%   3. Run this script from MATLAB (it manages load_gen.py automatically)

clear; clc;

id_dir  = fileparts(mfilename('fullpath'));
src_dir = fullfile(id_dir, '..', 'src');
py_gen  = fullfile(id_dir, '..', 'load_gen.py');
pid_file = '/tmp/load_gen.pid';
addpath(src_dir);

fprintf('[init] id_dir  = %s\n', id_dir);
fprintf('[init] src_dir = %s\n', src_dir);
fprintf('[init] load_gen= %s\n\n', py_gen);

% -------------------------------------------------------------------------
% Configuration
% -------------------------------------------------------------------------
cfg.base_url    = 'http://localhost:8001';
cfg.model       = 'mlx-community/Qwen3-0.6B-4bit';
cfg.metrics_url = 'http://localhost:8001/metrics';
cfg.gen_url     = 'http://localhost:8001/v1/completions';
cfg.timeout     = 30;
cfg.dt          = 1.0;       % s -- observation tick

cfg.num_predict = 1;         % Stage 2: TTFT
cfg.e2e_tokens  = 20;        % Stage 2 e2e and load_gen tokens

cfg.b_sweep     = [1 2 3 4 5 6 8];
cfg.n_reps      = 5;

% Stage 3: sustained load to identify beta
cfg.s3_rate     = 16;        % req/s fed to vLLM (> max_num_seqs*2 = 8)
cfg.s3_workers  = 16;        % Python threads
cfg.build_ticks = 20;        % ticks with load before measuring
cfg.meas_ticks3 = 15;

% Stage 4: envelope
cfg.lambda_sweep = [2 4 6 8 12 16];  % req/s for Python load generator
cfg.settle_ticks = 15;
cfg.meas_ticks4  = 10;

cfg.prompts = {'What is 2+2?'; 'Name a colour.'; 'Capital of France?'; 'Days in a week?'; 'Name a planet.'};

fprintf('[config] model        = %s\n',   cfg.model);
fprintf('[config] dt           = %.1fs\n', cfg.dt);
fprintf('[config] s3_rate      = %d req/s\n', cfg.s3_rate);
fprintf('[config] lambda_sweep = %s req/s\n\n', num2str(cfg.lambda_sweep));

% =========================================================================
% STAGE 1 -- Smoke test
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 1: Smoke test\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

try
    webread([cfg.base_url '/health'], weboptions('Timeout',5));
    fprintf('[stage1] vLLM health: OK\n');
catch e
    error('vLLM not reachable: %s', e.message);
end

m = read_metrics(cfg.metrics_url);
fprintf('[stage1] num_requests_running = %g\n', safe_field(m,'vllm_num_requests_running'));
fprintf('[stage1] num_requests_waiting = %g\n', safe_field(m,'vllm_num_requests_waiting'));

% Confirm Python and load_gen.py are available
[rc, ~] = system('python3 --version 2>&1');
if rc ~= 0
    error('python3 not found -- needed for load generator');
end
if ~exist(py_gen, 'file')
    error('load_gen.py not found at %s', py_gen);
end
fprintf('[stage1] Python3 OK, load_gen.py found.\n\n');

% =========================================================================
% STAGE 2 -- B sweep at q=0  (blocking parfeval, TTFT)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 2: B sweep at q=0  [TTFT, blocking parfeval]\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', min(max(cfg.b_sweep)+2, feature('numcores')));
end
addAttachedFiles(gcp(), src_dir);
fprintf('[stage2] Pool: %d workers\n\n', gcp().NumWorkers);

fprintf('[stage2] Warming up...\n');
for w = 1:3
    f = parfeval(@vllm_ttft, 1, cfg.gen_url, cfg.model, 'Hello', cfg.num_predict, cfg.timeout);
    try; lat = fetchOutputs(f); fprintf('  warmup %d: %.0fms\n',w,lat); catch; end
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

v  = isfinite(l_mean_b) & l_mean_b > 0;
p2 = [cfg.b_sweep(v)', cfg.b_sweep(v)'.^2] \ l_mean_b(v);
alpha_id = p2(1); gamma_id = p2(2);
l_fit = [cfg.b_sweep(v)', cfg.b_sweep(v)'.^2] * p2;
r2_s2 = 1 - sum((l_mean_b(v)-l_fit).^2)/sum((l_mean_b(v)-mean(l_mean_b(v))).^2);

fprintf('[stage2] alpha = %.4f ms/req\n',   alpha_id);
fprintf('[stage2] gamma = %.4f ms/req^2\n', gamma_id);
fprintf('[stage2] R^2   = %.4f\n\n',        r2_s2);

fig2 = figure('Visible','off');
errorbar(cfg.b_sweep, l_mean_b, l_std_b, 'bo','MarkerFaceColor','b','LineWidth',1.2); hold on;
bf = linspace(0.5,max(cfg.b_sweep)+0.5,200);
plot(bf, alpha_id*bf+gamma_id*bf.^2, 'r-','LineWidth',2);
xlabel('B [req]'); ylabel('TTFT [ms]');
title(sprintf('Stage 2 (q=0): alpha=%.3f  gamma=%.4f  R^2=%.3f',alpha_id,gamma_id,r2_s2));
grid on; saveas(fig2, fullfile(id_dir,'ch5_stage2_b_sweep.png'));
fprintf('[stage2] Plot saved.\n\n');

% =========================================================================
% STAGE 3 -- Queue buildup  (Python load generator, MATLAB observes)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 3: Queue buildup  [Python load gen, MATLAB observes]\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');
fprintf('Strategy:\n');
fprintf('  Python fires %d req/s continuously in background threads.\n', cfg.s3_rate);
fprintf('  MATLAB reads /metrics every dt=%.1fs to observe q[k].\n', cfg.dt);
fprintf('  Residual l_ss - (alpha*B_ss + gamma*B_ss^2) = beta * q_ss\n\n');

% Start Python load generator
stop_load_gen(pid_file);   % kill any existing instance
cmd = sprintf('source ~/.venv-vllm-metal/bin/activate && python3 %s --rate %d --workers %d --tokens %d --port 8001 --pid_file %s > /tmp/load_gen.log 2>&1 &', ...
    py_gen, cfg.s3_rate, cfg.s3_workers, cfg.e2e_tokens, pid_file);
system(cmd);
pause(2);  % let load generator start up
fprintf('[stage3] Load generator started at %d req/s\n', cfg.s3_rate);

n3      = cfg.build_ticks + cfg.meas_ticks3;
q_log3  = zeros(n3,1);
l_log3  = zeros(n3,1);

for tick = 1:n3
    phase = 'build';
    if tick > cfg.build_ticks; phase = 'MEAS '; end

    t_tick = tic;

    % Read q[k] and l_meas from vLLM metrics
    m    = read_metrics(cfg.metrics_url);
    q_k  = safe_field(m,'vllm_num_requests_waiting');
    % l_p95 from histogram sum/count
    l_k  = get_ttft_mean_ms(m);

    q_log3(tick) = q_k;
    l_log3(tick) = l_k;

    fprintf('  tick %3d [%s]  q=%5.1f  l=%5.0f ms\n', tick, phase, q_k, l_k);

    % Wait remainder of dt
    elapsed = toc(t_tick);
    if elapsed < cfg.dt; pause(cfg.dt - elapsed); end
end

% Stop load generator and drain queue before Stage 4
stop_load_gen(pid_file);
drain_queue(cfg.metrics_url);
fprintf('[stage3] Load generator stopped and queue drained.\n\n');

meas3 = (cfg.build_ticks+1):n3;
q_ss3 = mean(q_log3(meas3));
l_ss3 = mean(l_log3(meas3));
% Use mode of b during meas window from vLLM perspective (running + waiting capped at max_num_seqs*2)
% We don't have b_ss directly -- approximate from running at measurement time
m_tmp = read_metrics(cfg.metrics_url);
b_ss3 = safe_field(m_tmp,'vllm_num_requests_running');  % after load stops, ~0
% Use alpha/gamma fit to estimate effective B during load
% b_ss ≈ min(rate, max_num_seqs) = 4
b_ss3 = 4;  % max_num_seqs
svc3  = alpha_id*b_ss3 + gamma_id*b_ss3^2;

fprintf('[stage3] q_ss=%.2f  l_ss=%.0fms  b_ss(approx)=%d  svc=%.0fms\n', ...
    q_ss3, l_ss3, b_ss3, svc3);

if q_ss3 > 0.5 && ~isnan(l_ss3)
    beta_id = (l_ss3 - svc3) / q_ss3;
    fprintf('[stage3] beta = %.4f ms/req\n\n', beta_id);
else
    beta_id = NaN;
    fprintf('[stage3] WARNING: q_ss=%.2f or l_ss=NaN -- queue did not build.\n\n', q_ss3);
end

fig3 = figure('Visible','off');
yyaxis left;
plot(1:n3, q_log3, 'b-o','LineWidth',1.5); ylabel('q [req]');
yyaxis right;
plot(1:n3, l_log3, 'r-s','LineWidth',1.5); ylabel('l_{meas} [ms]');
hold on; xline(cfg.build_ticks+0.5,'k--','Measurement window');
xlabel('Tick'); title('Stage 3 — Queue buildup (Python load generator)'); grid on;
saveas(fig3, fullfile(id_dir,'ch5_stage3_queue.png'));
fprintf('[stage3] Plot saved.\n\n');

% =========================================================================
% STAGE 4 -- Operating envelope
% =========================================================================
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('STAGE 4: Operating envelope  [Python load swept over lambda]\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

n_lam = numel(cfg.lambda_sweep);
env_q = zeros(n_lam,1);
env_l = zeros(n_lam,1);

for li = 1:n_lam
    rate = cfg.lambda_sweep(li);
    workers = max(rate, 8);   % enough threads to sustain the rate
    fprintf('[stage4] rate=%d req/s  workers=%d  (%d settle + %d meas ticks)...\n', ...
        rate, workers, cfg.settle_ticks, cfg.meas_ticks4);

    % Start load generator at this rate
    stop_load_gen(pid_file);
    cmd = sprintf('source ~/.venv-vllm-metal/bin/activate && python3 %s --rate %d --workers %d --tokens %d --port 8001 --pid_file %s > /tmp/load_gen.log 2>&1 &', ...
        py_gen, rate, workers, cfg.e2e_tokens, pid_file);
    system(cmd);
    pause(2);

    n4 = cfg.settle_ticks + cfg.meas_ticks4;
    q_all = zeros(n4,1);
    l_all = zeros(n4,1);

    for tick = 1:n4
        phase = 'settle'; if tick > cfg.settle_ticks; phase = 'MEAS  '; end
        t_tick = tic;

        m   = read_metrics(cfg.metrics_url);
        q_k = safe_field(m,'vllm_num_requests_waiting');
        l_k = get_ttft_mean_ms(m);

        q_all(tick) = q_k;
        l_all(tick) = l_k;

        fprintf('  tick %2d [%s]  q=%5.1f  l=%5.0f ms\n', tick, phase, q_k, l_k);

        elapsed = toc(t_tick);
        if elapsed < cfg.dt; pause(cfg.dt - elapsed); end
    end

    stop_load_gen(pid_file);
    drain_queue(cfg.metrics_url);  % poll until running=0 AND waiting=0

    meas4 = (cfg.settle_ticks+1):n4;
    env_q(li) = mean(q_all(meas4));
    env_l(li) = mean(l_all(meas4));
    fprintf('  --> q_ss=%.2f  l_ss=%.0fms\n\n', env_q(li), env_l(li));
end

fig4 = figure('Visible','off','Position',[100 100 900 350]);
subplot(1,2,1);
plot(cfg.lambda_sweep, env_l, 'b-o','LineWidth',1.5,'MarkerFaceColor','b');
xlabel('\lambda [req/s]'); ylabel('l_{ss} [ms]'); title('Latency vs rate'); grid on;
subplot(1,2,2);
plot(cfg.lambda_sweep, env_q, 'r-o','LineWidth',1.5,'MarkerFaceColor','r');
xlabel('\lambda [req/s]'); ylabel('q_{ss} [req]'); title('Queue depth vs rate'); grid on;
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
fprintf('║  Operating envelope (rate in req/s):\n');
for li = 1:n_lam
    fprintf('║    rate=%3d  q=%5.1f  l=%6.0f ms\n', cfg.lambda_sweep(li), env_q(li), env_l(li));
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
    raw = webread(url, weboptions('Timeout',5,'ContentType','text'));
    m   = struct();
    for line = strsplit(raw, newline)
        s = strtrim(line{1});
        if isempty(s)||s(1)=='#'; continue; end
        c = strtrim(regexprep(s,'\{[^}]*\}',''));
        p = regexp(c,'\s+','split');
        if numel(p)<2; continue; end
        v = str2double(p{2});
        if isnan(v); continue; end
        f = matlab.lang.makeValidName(p{1});
        if isfield(m,f); m.(f)=m.(f)+v; else; m.(f)=v; end
    end
end

function v = safe_field(m, field)
    if isfield(m, field); v = m.(field); else; v = 0; end
end

function l_ms = get_ttft_mean_ms(m)
% Per-tick mean TTFT using histogram deltas between successive calls.
%
% vllm:time_to_first_token_seconds_sum and _count are CUMULATIVE counters.
% Dividing sum/count gives the all-time average since vLLM started.
% To get the per-tick mean we track deltas: (Δsum / Δcount) * 1000 ms.
% On the first call (or after a reset) returns NaN -- no prior baseline.
    persistent prev_sum prev_count
    if isempty(prev_sum); prev_sum = 0; prev_count = 0; end
    curr_sum   = safe_field(m, 'vllm_time_to_first_token_seconds_sum');
    curr_count = safe_field(m, 'vllm_time_to_first_token_seconds_count');
    delta_sum   = curr_sum   - prev_sum;
    delta_count = curr_count - prev_count;
    prev_sum   = curr_sum;
    prev_count = curr_count;
    if delta_count > 0
        l_ms = (delta_sum / delta_count) * 1000;
    else
        l_ms = NaN;
    end
end

function drain_queue(metrics_url)
% Block until vLLM reports both running=0 AND waiting=0.
% Polls every 0.5s, times out after 60s with a warning.
    fprintf('  [drain] Waiting for queue to empty...');
    t0 = tic;
    while toc(t0) < 60
        try
            raw = webread(metrics_url, weboptions('Timeout',3,'ContentType','text'));
            running = 0; waiting = 0;
            for line = strsplit(raw, newline)
                s = strtrim(line{1});
                if isempty(s)||s(1)=='#'; continue; end
                c = strtrim(regexprep(s,'\{[^}]*\}',''));
                p = regexp(c,'\s+','split');
                if numel(p)<2; continue; end
                v = str2double(p{2}); if isnan(v); continue; end
                if contains(p{1},'num_requests_running'); running=running+v; end
                if contains(p{1},'num_requests_waiting'); waiting=waiting+v; end
            end
            if running == 0 && waiting == 0
                fprintf(' done (%.0fs)  running=%g  waiting=%g\n', toc(t0), running, waiting);
                return
            end
        catch; end
        pause(0.5);
    end
    fprintf(' TIMEOUT after 60s\n');
end

function stop_load_gen(pid_file)
% Kill existing load_gen.py process if running.
    if exist(pid_file, 'file')
        pid_txt = strtrim(fileread(pid_file));
        system(sprintf('kill %s 2>/dev/null; rm -f %s', pid_txt, pid_file));
        pause(0.5);
    end
    % Also kill any stray python load_gen processes
    system('pkill -f load_gen.py 2>/dev/null; true');
end
