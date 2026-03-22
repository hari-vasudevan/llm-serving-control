%% identify_plant.m  --  Chapter 3: Plant identification for qwen2.5:3b
%
% Standalone — no dependency on setup_plant.m or any workspace variable.
% Run this script independently to measure real TTFT from Ollama and fit
% the three model parameters:
%
%   l_mean = alpha*B + gamma*B^2 + beta*q
%
% Strategy (two sequential stages, no parallel pool):
%
%   Stage 1 -- B sweep at q=0
%     Single-shot bursts (no sustained arrivals, queue stays empty).
%     Fits alpha and gamma from l(B) = alpha*B + gamma*B^2.
%
%   Stage 2 -- Sustained load sweep
%     Run sustained arrivals at multiple lambda values, let queue settle,
%     then measure steady-state (B_ss, q_ss, l_ss).
%     Residual: l_ss - (alpha*B_ss + gamma*B_ss^2) = beta*q_ss
%     Fits beta by linear regression through the origin.
%
% Outputs
%   identified_params.mat   -- struct with alpha, gamma, beta, R2 values
%   id_stage1.png           -- TTFT vs B fit plot
%   id_stage2.png           -- queuing term vs q_ss fit plot

clear; clc;

% -------------------------------------------------------------------------
% Paths  (self-contained -- resolve relative to this file's location)
% -------------------------------------------------------------------------
id_dir  = fileparts(mfilename('fullpath'));
src_dir = fullfile(id_dir, '..', 'src');
addpath(src_dir);

fprintf('[init] id_dir  = %s\n', id_dir);
fprintf('[init] src_dir = %s\n', src_dir);

% Confirm ollama_ttft.m is reachable
if ~exist('ollama_ttft', 'file')
    error('ollama_ttft.m not found on path. Check src_dir = %s', src_dir);
end
fprintf('[init] ollama_ttft found on path\n\n');

% -------------------------------------------------------------------------
% Configuration
% -------------------------------------------------------------------------
cfg.url         = 'http://localhost:11434/api/generate';
cfg.model       = 'qwen2.5:3b';
cfg.num_predict = 1;      % 1 token: minimises generation component in TTFT
cfg.timeout     = 20;     % s per request (sequential, so generous)

% Stage 1
cfg.b_sweep     = [1 2 3 4 5 6 8 10 12];
cfg.n_reps      = 3;      % bursts per B value (averaged)

% Stage 2
cfg.lambda_vals  = [2 3 5 7 9];   % req/tick — each drives a different q_ss
cfg.settle_ticks = 20;            % ticks to reach steady state
cfg.measure_ticks= 15;            % ticks to average over
cfg.b_min        = 1;
cfg.b_max        = 12;
cfg.q_max        = 20;

fprintf('[config] model       = %s\n',   cfg.model);
fprintf('[config] b_sweep     = %s\n',   num2str(cfg.b_sweep));
fprintf('[config] n_reps      = %d\n',   cfg.n_reps);
fprintf('[config] lambda_vals = %s\n',   num2str(cfg.lambda_vals));
fprintf('[config] settle/meas = %d / %d ticks\n\n', cfg.settle_ticks, cfg.measure_ticks);

% -------------------------------------------------------------------------
% Confirm Ollama is reachable before starting
% -------------------------------------------------------------------------
fprintf('[check] Pinging Ollama at %s ...\n', cfg.url);
try
    opts = weboptions('Timeout', 5);
    webread('http://localhost:11434/api/tags', opts);
    fprintf('[check] Ollama is up.\n\n');
catch
    error('Ollama not reachable at localhost:11434. Is it running?');
end

% -------------------------------------------------------------------------
% Helper: single blocking TTFT call (no parfeval)
% -------------------------------------------------------------------------
    function lat_ms = single_ttft(url, model, prompt, n_pred, timeout_sec)
        % Direct Java streaming call — no parallel pool needed.
        timeout_ms = int32(timeout_sec * 1000);
        body = sprintf( ...
            '{"model":"%s","prompt":"%s","stream":true,"options":{"num_predict":%d}}', ...
            model, prompt, n_pred);
        url_obj = java.net.URL(url);
        conn    = url_obj.openConnection();
        conn.setRequestMethod('POST');
        conn.setRequestProperty('Content-Type', 'application/json');
        conn.setDoOutput(true);
        conn.setConnectTimeout(timeout_ms);
        conn.setReadTimeout(timeout_ms);
        out = conn.getOutputStream();
        out.write(int8(body));
        out.flush();
        out.close();
        t_start = tic;
        reader  = java.io.BufferedReader(java.io.InputStreamReader(conn.getInputStream()));
        reader.readLine();   % blocks until first token = TTFT
        lat_ms  = toc(t_start) * 1000;
        reader.close();
        conn.disconnect();
    end

% -------------------------------------------------------------------------
% Warm-up: 3 single requests to load model weights into GPU memory
% -------------------------------------------------------------------------
fprintf('[warmup] Firing 3 warm-up requests (sequential)...\n');
for w = 1:3
    t = tic;
    lat = single_ttft(cfg.url, cfg.model, 'Hello', 1, cfg.timeout);
    fprintf('  warmup %d: %.0f ms  (elapsed %.1f s)\n', w, lat, toc(t));
end
fprintf('[warmup] Done.\n\n');

% =========================================================================
% STAGE 1 -- B sweep at q = 0  (fit alpha, gamma)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('STAGE 1: B sweep at q=0  (fitting alpha and gamma)\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

% Each "burst" fires B sequential requests immediately one after another.
% Because no arrivals come between requests the queue stays at 0.
% We average over n_reps bursts to reduce noise.

n_b      = numel(cfg.b_sweep);
l_mean_b = zeros(n_b, 1);
l_std_b  = zeros(n_b, 1);

for bi = 1:n_b
    b = cfg.b_sweep(bi);
    fprintf('[stage1] B = %2d  (%d reps x %d requests) ...\n', b, cfg.n_reps, b);

    rep_means = zeros(cfg.n_reps, 1);
    for r = 1:cfg.n_reps
        lats = zeros(1, b);
        for i = 1:b
            lats(i) = single_ttft(cfg.url, cfg.model, 'What is 2+2?', ...
                cfg.num_predict, cfg.timeout);
        end
        rep_means(r) = mean(lats);
        fprintf('  rep %d: mean TTFT = %.1f ms  [%s]\n', r, rep_means(r), ...
            num2str(round(lats), '%d '));
    end

    l_mean_b(bi) = mean(rep_means);
    l_std_b(bi)  = std(rep_means);
    fprintf('  --> B=%2d average: %.1f ms  (std=%.1f ms)\n\n', b, l_mean_b(bi), l_std_b(bi));
end

% Least-squares fit: l = alpha*B + gamma*B^2
fprintf('[stage1] Fitting l = alpha*B + gamma*B^2 ...\n');
A1       = [cfg.b_sweep(:), cfg.b_sweep(:).^2];
params1  = A1 \ l_mean_b;
alpha_id = params1(1);
gamma_id = params1(2);
l_fit_b  = A1 * params1;
r2_s1    = 1 - sum((l_mean_b - l_fit_b).^2) / sum((l_mean_b - mean(l_mean_b)).^2);

fprintf('[stage1] alpha = %.4f  ms/req\n',   alpha_id);
fprintf('[stage1] gamma = %.4f  ms/req^2\n', gamma_id);
fprintf('[stage1] R^2   = %.4f\n\n',         r2_s1);

% Plot
fig1 = figure('Name','Stage 1','Visible','off');
errorbar(cfg.b_sweep, l_mean_b, l_std_b, 'bo', 'MarkerFaceColor','b', 'LineWidth',1.2);
hold on;
b_fine = linspace(1, max(cfg.b_sweep), 200);
plot(b_fine, alpha_id*b_fine + gamma_id*b_fine.^2, 'r-', 'LineWidth', 2);
xlabel('B  [req]'); ylabel('TTFT  [ms]');
title(sprintf('Stage 1 — B sweep\nalpha=%.3f  gamma=%.4f  R^2=%.3f', ...
    alpha_id, gamma_id, r2_s1));
legend('Measured', 'Fit: \alpha B + \gamma B^2', 'Location','northwest');
grid on;
s1_path = fullfile(id_dir, 'id_stage1.png');
saveas(fig1, s1_path);
fprintf('[stage1] Plot saved: %s\n\n', s1_path);

% =========================================================================
% STAGE 2 -- Sustained load sweep  (fit beta)
% =========================================================================
fprintf('═══════════════════════════════════════════════════════\n');
fprintf('STAGE 2: Sustained load  (fitting beta)\n');
fprintf('═══════════════════════════════════════════════════════\n\n');

% Simple drain rule (NOT the cascade controller):
%   B[k] = clamp(round(q[k] + lambda), b_min, b_max)
% This drives the system to a non-trivial steady-state queue for each lambda.
%
% Rolling buffer for l_mean: 10-sample window.
% Queue state tracked as a software counter.

n_lam    = numel(cfg.lambda_vals);
q_ss_vec = zeros(n_lam, 1);
b_ss_vec = zeros(n_lam, 1);
l_ss_vec = zeros(n_lam, 1);
beta_vec = zeros(n_lam, 1);

for li = 1:n_lam
    lam = cfg.lambda_vals(li);
    fprintf('[stage2] lambda = %d req/tick\n', lam);
    fprintf('  settling for %d ticks, then measuring %d ticks ...\n', ...
        cfg.settle_ticks, cfg.measure_ticks);

    q_k     = lam;                       % start near natural equilibrium
    lat_buf = 200 * ones(1, 10);         % 10-sample rolling buffer, warm-start
    buf_idx = 0;

    total_ticks = cfg.settle_ticks + cfg.measure_ticks;
    q_log = zeros(total_ticks, 1);
    b_log = zeros(total_ticks, 1);
    l_log = zeros(total_ticks, 1);

    for tick = 1:total_ticks
        % Poisson arrivals
        a_k = poissrnd(lam);

        % Drain rule: absorb arrivals + drain a bit of queue
        b_k = min(cfg.b_max, max(cfg.b_min, round(q_k + a_k)));

        % Fire b_k SEQUENTIAL requests (no parallel pool)
        lats_tick = zeros(1, b_k);
        for i = 1:b_k
            lats_tick(i) = single_ttft(cfg.url, cfg.model, ...
                'What is 2+2?', cfg.num_predict, cfg.timeout);
        end

        % Update rolling buffer
        for i = 1:b_k
            buf_idx = mod(buf_idx, 10) + 1;
            lat_buf(buf_idx) = lats_tick(i);
        end
        l_meas = mean(lat_buf);

        % Queue update
        q_k = max(0, min(cfg.q_max, q_k + a_k - b_k));

        q_log(tick) = q_k;
        b_log(tick) = b_k;
        l_log(tick) = l_meas;

        % Print every tick so you can see progress
        phase = 'settle';
        if tick > cfg.settle_ticks; phase = 'MEAS  '; end
        fprintf('  tick %3d [%s] a=%d b=%d q=%.1f l=%.1f ms\n', ...
            tick, phase, a_k, b_k, q_k, l_meas);
    end

    % Steady-state averages over measurement window
    idx_meas = (cfg.settle_ticks+1):total_ticks;
    q_ss = mean(q_log(idx_meas));
    b_ss = mean(b_log(idx_meas));
    l_ss = mean(l_log(idx_meas));

    q_ss_vec(li) = q_ss;
    b_ss_vec(li) = b_ss;
    l_ss_vec(li) = l_ss;

    % Beta estimate from this operating point
    svc = alpha_id * b_ss + gamma_id * b_ss^2;
    if q_ss > 0.1
        beta_vec(li) = (l_ss - svc) / q_ss;
        fprintf('  --> q_ss=%.2f  b_ss=%.2f  l_ss=%.1f ms  svc=%.1f ms  beta_est=%.4f\n\n', ...
            q_ss, b_ss, l_ss, svc, beta_vec(li));
    else
        beta_vec(li) = NaN;
        fprintf('  --> q_ss=%.2f (too small, skipping beta estimate)\n\n', q_ss);
    end
end

% Fit beta: linear regression through origin on residual vs q_ss
fprintf('[stage2] Fitting beta by regression: (l_ss - svc) = beta * q_ss ...\n');
valid     = ~isnan(beta_vec) & q_ss_vec > 0.1;
svc_all   = alpha_id * b_ss_vec + gamma_id * b_ss_vec.^2;
residuals = l_ss_vec - svc_all;
beta_id   = (q_ss_vec(valid)' * residuals(valid)) / ...
            (q_ss_vec(valid)' * q_ss_vec(valid));

l_fit_s2  = svc_all + beta_id * q_ss_vec;
r2_s2     = 1 - sum((l_ss_vec(valid) - l_fit_s2(valid)).^2) / ...
                sum((l_ss_vec(valid) - mean(l_ss_vec(valid))).^2);

fprintf('[stage2] beta = %.4f  ms/req\n', beta_id);
fprintf('[stage2] R^2  = %.4f\n\n',       r2_s2);

% Plot
fig2 = figure('Name','Stage 2','Visible','off');
scatter(q_ss_vec, residuals, 60, 'b', 'filled');
hold on;
q_fine = linspace(0, max(q_ss_vec)*1.2, 100);
plot(q_fine, beta_id * q_fine, 'r-', 'LineWidth', 2);
xlabel('q_{ss}  [req]');
ylabel('l_{ss} - (\alpha B + \gamma B^2)  [ms]');
title(sprintf('Stage 2 — Queuing term\nbeta=%.4f ms/req   R^2=%.3f', beta_id, r2_s2));
legend('Measured', sprintf('\\beta q  (\\beta=%.4f)', beta_id));
grid on;
s2_path = fullfile(id_dir, 'id_stage2.png');
saveas(fig2, s2_path);
fprintf('[stage2] Plot saved: %s\n\n', s2_path);

% =========================================================================
% SUMMARY
% =========================================================================
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║  IDENTIFIED PARAMETERS  (%s)\n', cfg.model);
fprintf('╠══════════════════════════════════════════════════════════╣\n');
fprintf('║  alpha = %8.4f  ms/req        R^2 stage1 = %.4f\n', alpha_id, r2_s1);
fprintf('║  gamma = %8.4f  ms/req^2      R^2 stage1 = %.4f\n', gamma_id, r2_s1);
fprintf('║  beta  = %8.4f  ms/req        R^2 stage2 = %.4f\n', beta_id,  r2_s2);
fprintf('╠══════════════════════════════════════════════════════════╣\n');
fprintf('║  Assumed (stochastic model):                            ║\n');
fprintf('║    alpha=0.1000  gamma=0.8000  beta=2.0000              ║\n');
fprintf('╠══════════════════════════════════════════════════════════╣\n');

dt_s     = 1.0;
tau_out  = 30.0;
z_cl     = exp(-dt_s / tau_out);
K_il_id  = (z_cl - 1) / beta_id;
K_il_old = (z_cl - 1) / 2.0;
fprintf('║  Outer loop gain (tau_out=%.0fs):\n', tau_out);
fprintf('║    K_il (identified) = %.6f\n', K_il_id);
fprintf('║    K_il (assumed)    = %.6f\n', K_il_old);
fprintf('║    Ratio             = %.2fx\n', K_il_id / K_il_old);
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

% Copy these lines into setup_plant.m:
fprintf('Paste into setup_plant.m:\n');
fprintf('  perturbed.alpha = %.4f;\n', alpha_id);
fprintf('  perturbed.gamma = %.4f;\n', gamma_id);
fprintf('  perturbed.beta  = %.4f;\n', beta_id);

% =========================================================================
% SAVE
% =========================================================================
identified.alpha     = alpha_id;
identified.gamma     = gamma_id;
identified.beta      = beta_id;
identified.r2_stage1 = r2_s1;
identified.r2_stage2 = r2_s2;
identified.model     = cfg.model;
identified.timestamp = char(datetime('now'));
identified.raw.b_sweep      = cfg.b_sweep;
identified.raw.l_mean_b     = l_mean_b;
identified.raw.lambda_vals  = cfg.lambda_vals;
identified.raw.q_ss         = q_ss_vec;
identified.raw.b_ss         = b_ss_vec;
identified.raw.l_ss         = l_ss_vec;

out_path = fullfile(id_dir, 'identified_params.mat');
save(out_path, 'identified');
fprintf('\n[save] Results saved to: %s\n', out_path);
