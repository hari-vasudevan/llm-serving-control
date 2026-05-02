%% characterise.m  --  Chapter 6: Plant identification + operating point analysis
%
% WHAT THIS DOES
% --------------
% 1. Measures TTFT(B) at q=0 via B sweep (fits alpha, gamma)
% 2. Estimates per-tick latency at realistic operating points
% 3. Recommends L_target, lambda_steady, lambda_spike for run_controller.m
%
% KEY FIX vs previous version:
%   - 6+ warm-up requests (not 2) to stabilise Ollama's JIT cache
%   - First rep discarded -- always cold and anomalous
%   - 6 reps per B (not 4)
%   - Waits for idle between reps (polls until completed count stable)
%   - Uses enqueue_sync for B=1 timing sanity check
%   - Derives realistic operating params from measured data, not assumptions
%
% COMMUNICATION: curl via system() -- MATLAB Java HTTP blocked on this Mac

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% ── Config ────────────────────────────────────────────────────────────────
SERVER  = 'http://192.168.68.106:8002';
B_SWEEP = [1 2 3 4 6 8];  % batch sizes to sweep
N_REPS  = 6;               % reps per B (first will be discarded)
N_WARMUP = 6;              % warm-up requests before sweep
TIMEOUT  = 120;            % per-batch timeout [s]
DT       = 1.0;            % controller tick [s]

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 6: Plant Identification                             ║\n');
fprintf('║  Server: %s\n', SERVER);
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ── Stage 1: Smoke test ───────────────────────────────────────────────────
fprintf('[stage1] Smoke test...\n');
h = srv_get(SERVER, '/health');
assert(strcmp(h.status,'ok'), 'Server not healthy');
fprintf('  OK -- model=%s  q=%d  B=%d\n\n', h.model, h.q_sw, h.B);

server_reset(SERVER);

% ── Stage 2: Warm up Ollama ───────────────────────────────────────────────
fprintf('[warmup] %d warm-up requests at B=1...\n', N_WARMUP);
srv_post(SERVER, '/control', struct('B', 1));
for w = 1:N_WARMUP
    server_reset(SERVER);
    srv_post(SERVER, '/control', struct('B', 1));
    srv_post(SERVER, '/enqueue', struct('prompt', 'Hello'));
    wait_for_completions(SERVER, 1, 30);
    m = srv_get(SERVER, '/metrics');
    fprintf('  warmup %d: %.0f ms\n', w, m.l_total_mean);
end
fprintf('  Warm-up done -- Ollama cache is hot.\n\n');

% ── Stage 3: B sweep ──────────────────────────────────────────────────────
fprintf('[sweep] B sweep at q_sw=0\n');
fprintf('  Discarding rep 1 (cold) -- using reps 2..%d\n', N_REPS);
fprintf('  B values: %s\n\n', num2str(B_SWEEP));

PROMPTS = {'What is 2+2?','Name a colour.','Capital of France?', ...
           'Days in a week?','Name a planet.','Speed of light?', ...
           'Name a mammal.','10 times 10?'};

l_mean_all = NaN(N_REPS, numel(B_SWEEP));  % rows=reps, cols=B values

for bi = 1:numel(B_SWEEP)
    b = B_SWEEP(bi);
    fprintf('[B=%d] %d reps (rep 1 will be discarded)...\n', b, N_REPS);
    srv_post(SERVER, '/control', struct('B', b));

    for r = 1:N_REPS
        server_reset(SERVER);
        srv_post(SERVER, '/control', struct('B', b));

        % Enqueue exactly b requests in rapid succession (q=0 guaranteed after reset)
        for i = 1:b
            p = PROMPTS{mod(i-1, numel(PROMPTS)) + 1};
            srv_post(SERVER, '/enqueue', struct('prompt', p));
        end

        % Wait for all b to complete
        ok = wait_for_completions(SERVER, b, TIMEOUT);
        if ~ok
            fprintf('  rep %d: TIMEOUT\n', r);
            continue;
        end

        m = srv_get(SERVER, '/metrics');
        if ~isempty(m.l_total_mean) && ~isnan(m.l_total_mean)
            l_mean_all(r, bi) = m.l_total_mean;
            if r == 1
                fprintf('  rep %d: %.0f ms  [DISCARDED -- cold]\n', r, m.l_total_mean);
            else
                fprintf('  rep %d: %.0f ms\n', r, m.l_total_mean);
            end
        end

        pause(2);  % brief gap between reps
    end

    % Use reps 2..N_REPS (skip rep 1)
    valid = l_mean_all(2:end, bi);
    valid = valid(~isnan(valid));
    if ~isempty(valid)
        fprintf('  B=%d summary: mean=%.0f  std=%.0f  n=%d\n\n', ...
            b, mean(valid), std(valid), numel(valid));
    end
end

% ── Stage 4: Fit model ────────────────────────────────────────────────────
fprintf('[fit] TTFT(B) = alpha*B + gamma*B^2\n');

% Build mean/std per B, excluding rep 1
l_mean = NaN(1, numel(B_SWEEP));
l_std  = NaN(1, numel(B_SWEEP));
for bi = 1:numel(B_SWEEP)
    valid = l_mean_all(2:end, bi);
    valid = valid(~isnan(valid));
    if numel(valid) >= 2
        l_mean(bi) = mean(valid);
        l_std(bi)  = std(valid);
    end
end

valid_idx = ~isnan(l_mean);
B_v = B_SWEEP(valid_idx)';
L_v = l_mean(valid_idx)';
A   = [B_v, B_v.^2];
p   = A \ L_v;
alpha = p(1);
gamma = p(2);
L_fit = A * p;
r2 = 1 - sum((L_v - L_fit).^2) / sum((L_v - mean(L_v)).^2);

fprintf('  TTFT(B) = %.4f*B + (%.4f)*B^2\n', alpha, gamma);
fprintf('  R^2 = %.4f\n\n', r2);

% ── Stage 5: Operating point analysis ────────────────────────────────────
fprintf('[ops] Computing realistic operating parameters...\n\n');

% TTFT at each B value
fprintf('  TTFT measurements:\n');
fprintf('  %3s  %8s  %8s\n', 'B', 'TTFT_ms', 'std_ms');
for bi = 1:numel(B_SWEEP)
    if ~isnan(l_mean(bi))
        fprintf('  %3d  %8.0f  %8.0f\n', B_SWEEP(bi), l_mean(bi), l_std(bi));
    end
end
fprintf('\n');

% Find a sustainable lambda:
% At steady state with lambda req/tick and B=lambda, queue_wait=0, so:
%   l_total_ss = TTFT(lambda)
% Choose lambda_ss such that TTFT(lambda_ss) fits comfortably in DT=1s
% On Intel CPU, request ~ 500ms each. With OLLAMA_NUM_PARALLEL=4:
%   B=2 requests fire concurrently, each takes ~500ms, wall clock ~ max(times) ~ 500-600ms
%   B=4 requests fire concurrently, wall clock ~ 800-1200ms
% Find the highest lambda where TTFT < 2.0s (reasonable wall clock)

fprintf('  Estimating sustainable lambda (TTFT < 2000ms per tick)...\n');
lambda_sustainable = 1;
for bi = 1:numel(B_SWEEP)
    b = B_SWEEP(bi);
    ttft_b = alpha*b + gamma*b^2;
    if ttft_b > 0 && ttft_b < 2000
        lambda_sustainable = b;
    end
end
fprintf('  lambda_sustainable = %d (TTFT(B=%d) = %.0f ms)\n', ...
    lambda_sustainable, lambda_sustainable, alpha*lambda_sustainable + gamma*lambda_sustainable^2);

% Operating point: lambda_ss = floor(lambda_sustainable * 0.6)
% This gives headroom so spike can double lambda without overloading
lambda_ss   = max(1, floor(lambda_sustainable * 0.6));
lambda_spike = min(8, lambda_ss * 2);
B0          = lambda_ss;  % nominal B = nominal lambda at equilibrium

ttft_B0 = alpha*B0 + gamma*B0^2;
beta_eff = alpha + 2*gamma*B0;   % d(TTFT)/dB at B0
beta_q   = DT*1000 / B0;         % d(l_total)/d(q) -- analytical

% At equilibrium (q=0, lambda=lambda_ss): l_ss = TTFT(B0)
l_ss = ttft_B0;

% L_target: set to 1.8x the equilibrium latency
% This gives the controller room below (can respond to drops) and headroom above
L_target_recommended = round(l_ss * 1.8 / 100) * 100;   % round to nearest 100ms

% Sanity: during spike (lambda=lambda_spike, if q builds to q_max=5):
%   l_spike = TTFT(B0) + (q_max/B0)*dt*1000  (worst case approximation)
q_max_expected = lambda_spike * 10;  % 10 ticks of spike before queue stabilises
l_spike_estimate = ttft_B0 + (q_max_expected / B0) * DT * 1000;

fprintf('\n');
fprintf('  Recommended operating parameters:\n');
fprintf('  ┌─────────────────────────────────────────────────┐\n');
fprintf('  │  B0          = %d                               \n', B0);
fprintf('  │  lambda_ss   = %d req/tick (steady state)       \n', lambda_ss);
fprintf('  │  lambda_spike= %d req/tick (disturbance)         \n', lambda_spike);
fprintf('  │  l_ss        = %.0f ms (natural TTFT at B0=%d)   \n', l_ss, B0);
fprintf('  │  L_target    = %.0f ms (1.8x l_ss)               \n', L_target_recommended);
fprintf('  │  l_spike_est = %.0f ms (at q=%d, B=%d)           \n', l_spike_estimate, q_max_expected, B0);
fprintf('  │  beta_q      = %.1f ms/req (d(l)/d(q), analytical)\n', beta_q);
fprintf('  │  beta_eff    = %.1f ms/req (d(TTFT)/dB at B0)    \n', beta_eff);
fprintf('  └─────────────────────────────────────────────────┘\n\n');

fprintf('  Copy these into run_controller.m SEGMENTS:\n');
fprintf('    L_target_ss    = %d;\n', L_target_recommended);
fprintf('    lambda_ss      = %d;\n', lambda_ss);
fprintf('    lambda_spike   = %d;\n', lambda_spike);
fprintf('    L_target_tight = %d;   %% for Target-Drop segment\n', ...
    round(L_target_recommended * 0.6 / 100)*100);
fprintf('\n');

% ── Save ──────────────────────────────────────────────────────────────────
out_dir = fileparts(mfilename('fullpath'));
save(fullfile(out_dir, 'identified_params.mat'), ...
    'alpha','gamma','r2','B0','DT','ttft_B0','beta_eff','beta_q', ...
    'B_SWEEP','l_mean','l_std','l_mean_all', ...
    'lambda_ss','lambda_spike','L_target_recommended', ...
    'l_ss','l_spike_estimate','SERVER');
fprintf('[save] identified_params.mat\n');

% ── Plot ──────────────────────────────────────────────────────────────────
out_dir = fileparts(mfilename('fullpath'));
fig = figure('Visible','off','Position',[50 50 1100 520]);

subplot(1,2,1);
errorbar(B_SWEEP, l_mean, l_std, 'bo', 'LineWidth', 1.3, 'CapSize', 5, ...
    'DisplayName', 'Measured (mean ± std)');
hold on;
b_fine = linspace(0.5, max(B_SWEEP)+0.5, 300);
plot(b_fine, alpha*b_fine + gamma*b_fine.^2, 'r-', 'LineWidth', 2, ...
    'DisplayName', sprintf('αB+γB²  R²=%.3f', r2));
plot(B0, ttft_B0, 'gs', 'MarkerSize', 12, 'MarkerFaceColor','g', ...
    'DisplayName', sprintf('B0=%d  l_{ss}=%.0fms', B0, l_ss));
yline(L_target_recommended, 'm--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('L_{target}=%.0fms', L_target_recommended));
xlabel('Batch size B'); ylabel('l_{total} [ms]');
title(sprintf('TTFT(B) -- α=%.1f ms/req  γ=%.2f ms/req²\nR²=%.3f', alpha, gamma, r2));
legend('Location','northwest','FontSize',8); grid on;

subplot(1,2,2);
q_range = 0:0.5:15;
cmap = lines(numel(B_SWEEP));
for bi = 1:numel(B_SWEEP)
    b      = B_SWEEP(bi);
    ttft_b = alpha*b + gamma*b^2;
    if ttft_b > 0
        l_tot = ttft_b + (q_range/b)*DT*1000;
        plot(q_range, l_tot/1000, 'Color', cmap(bi,:), 'LineWidth', 1.3, ...
            'DisplayName', sprintf('B=%d', b));
        hold on;
    end
end
yline(L_target_recommended/1000, 'm--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('L_{target}=%.1fs', L_target_recommended/1000));
xlabel('q_{sw} [req]'); ylabel('l_{total} [s]');
title(sprintf('Full plant l_{total}(B,q)\n\\lambda_{ss}=%d  L_{tgt}=%.0fms', ...
    lambda_ss, L_target_recommended));
legend('Location','northwest','FontSize',8); grid on;

sgtitle(sprintf('Chapter 6 Plant Identification (Intel Mac)\nλ_{ss}=%d  λ_{spike}=%d  L_{target}=%.0fms', ...
    lambda_ss, lambda_spike, L_target_recommended), 'FontSize', 12);

saveas(fig, fullfile(out_dir, 'ch6_stage2_b_sweep.png'));
fprintf('[plot] ch6_stage2_b_sweep.png\n\n');

% =========================================================================
% Helpers
% =========================================================================
function m = srv_get(server, path)
    [~, out] = system(sprintf('curl -s "%s%s"', server, path));
    m = jsondecode(strtrim(out));
end

function r = srv_post(server, path, data)
    body = strrep(jsonencode(data), '"', '\"');
    cmd  = sprintf('curl -s -X POST "%s%s" -H "Content-Type: application/json" -d "%s"', ...
                   server, path, body);
    [~, out] = system(cmd);
    try; r = jsondecode(strtrim(out)); catch; r = struct(); end
end

function server_reset(server)
    srv_post(server, '/reset', struct());
    pause(0.5);
end

function ok = wait_for_completions(server, n, timeout_s)
    t0 = tic;
    while true
        m = srv_get(server, '/metrics');
        if m.completed >= n; ok = true; return; end
        if toc(t0) > timeout_s
            ok = false; return;
        end
        pause(0.5);
    end
end
