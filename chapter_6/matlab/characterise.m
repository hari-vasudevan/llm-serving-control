%% characterise.m  --  Chapter 6: Plant identification via queue server
%
% Identifies TTFT(B) = alpha*B + gamma*B^2 at q=0.
% Full plant model: l_total(B,q) = alpha*B + gamma*B^2 + (q/B)*dt*1000
%
% Uses the Intel Mac queue server at SERVER_URL.
% Communication via curl (MATLAB's Java HTTP is blocked on this Mac).
%
% Outputs: identified_params.mat  +  ch6_stage2_b_sweep.png

clear; clc;
addpath(fileparts(mfilename('fullpath')));

SERVER  = 'http://192.168.68.106:8002';
B0      = 3;          % nominal operating batch size
DT      = 1.0;        % dispatcher tick [s]
B_SWEEP = [1 2 3 4 5 6 8];
N_REPS  = 4;          % repetitions per B value
TIMEOUT = 90;         % per-request timeout [s]

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 6: Plant Identification                             ║\n');
fprintf('║  Server: %s\n', SERVER);
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% -------------------------------------------------------------------------
% Stage 1: Smoke test
% -------------------------------------------------------------------------
fprintf('[stage1] Smoke test...\n');
h = server_get(SERVER, '/health');
fprintf('  status=%s  model=%s  q=%d  B=%d\n', h.status, h.model, h.q_sw, h.B);

% Confirm queue is empty
if h.q_sw > 0
    fprintf('  WARNING: q_sw=%d -- resetting server\n', h.q_sw);
    server_post(SERVER, '/reset', struct());
    pause(1);
end
fprintf('[stage1] OK\n\n');

% -------------------------------------------------------------------------
% Stage 2: Warm up
% -------------------------------------------------------------------------
fprintf('[warmup] 2 warm-up requests (q=0)...\n');
for w = 1:2
    server_post(SERVER, '/reset', struct());
    server_post(SERVER, '/control', struct('B', 1));
    server_post(SERVER, '/enqueue', struct('prompt', 'Hello'));
    pause(3);
    m = server_get(SERVER, '/metrics');
    fprintf('  warmup %d: completed=%d  l_mean=%s ms\n', ...
        w, m.completed, num2str(m.l_total_mean));
end
fprintf('\n');

% -------------------------------------------------------------------------
% Stage 2: B sweep
% -------------------------------------------------------------------------
fprintf('[stage2] B sweep at q=0\n');
fprintf('  B sweep: %s\n', num2str(B_SWEEP));
fprintf('  %d reps per B\n\n', N_REPS);

l_mean = zeros(1, numel(B_SWEEP));
l_std  = zeros(1, numel(B_SWEEP));

for bi = 1:numel(B_SWEEP)
    b = B_SWEEP(bi);
    fprintf('[stage2] B=%d (%d reps)...\n', b, N_REPS);

    % Set dispatcher B so it will fire exactly b per tick
    server_post(SERVER, '/control', struct('B', b));

    rep_means = zeros(1, N_REPS);
    for r = 1:N_REPS
        % Reset metrics so we get a clean l_total measurement
        server_post(SERVER, '/reset', struct());
        server_post(SERVER, '/control', struct('B', b));

        % Enqueue exactly b requests in rapid succession
        prompts = {'What is 2+2?','Name a colour.','Capital of France?', ...
                   'Days in a week?','Name a planet.','Speed of light?', ...
                   'Name a mammal.','10 times 10?'};
        for i = 1:b
            p = prompts{mod(i-1, numel(prompts)) + 1};
            server_post(SERVER, '/enqueue', struct('prompt', p));
        end

        % Wait for dispatcher to fire and complete all b requests
        % Poll metrics until completed == b
        t_wait = tic;
        while true
            m = server_get(SERVER, '/metrics');
            if m.completed >= b
                break;
            end
            if toc(t_wait) > TIMEOUT
                fprintf('  WARNING: timeout waiting for completions\n');
                break;
            end
            pause(0.5);
        end

        if ~isempty(m.l_total_mean) && ~isnan(m.l_total_mean)
            rep_means(r) = m.l_total_mean;
            fprintf('  rep %d: %.1f ms  (completed=%d)\n', r, m.l_total_mean, m.completed);
        else
            rep_means(r) = NaN;
            fprintf('  rep %d: NaN (no completions)\n', r);
        end

        % Brief pause between reps to let queue drain fully
        pause(2);
    end

    valid = rep_means(~isnan(rep_means));
    if ~isempty(valid)
        l_mean(bi) = mean(valid);
        l_std(bi)  = std(valid);
    else
        l_mean(bi) = NaN;
        l_std(bi)  = 0;
    end
    fprintf('  --> B=%d: %.1f ± %.1f ms\n\n', b, l_mean(bi), l_std(bi));
end

% -------------------------------------------------------------------------
% Fit TTFT(B) = alpha*B + gamma*B^2  (no intercept)
% -------------------------------------------------------------------------
valid_idx = ~isnan(l_mean);
B_v = B_SWEEP(valid_idx)';
L_v = l_mean(valid_idx)';
A   = [B_v, B_v.^2];
p   = A \ L_v;
alpha = p(1);  gamma = p(2);
L_fit = A * p;
ss_res = sum((L_v - L_fit).^2);
ss_tot = sum((L_v - mean(L_v)).^2);
r2    = 1 - ss_res/ss_tot;

% Derived quantities
ttft_B0  = alpha*B0 + gamma*B0^2;
beta_eff = alpha + 2*gamma*B0;   % d(TTFT)/dB at B0
beta_q   = DT*1000 / B0;         % d(l_total)/d(q) -- analytical

fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('  TTFT(B) = %.4f*B + (%.4f)*B^2   R^2=%.4f\n', alpha, gamma, r2);
fprintf('  TTFT(B0=%d)  = %.2f ms\n', B0, ttft_B0);
fprintf('  beta_eff    = %.4f ms/req  [d(TTFT)/dB at B0]\n', beta_eff);
fprintf('  beta_q      = %.4f ms/req  [dt*1000/B0, analytical]\n', beta_q);
fprintf('═══════════════════════════════════════════════════════════\n\n');

% -------------------------------------------------------------------------
% Save
% -------------------------------------------------------------------------
out_dir = fileparts(mfilename('fullpath'));
save(fullfile(out_dir, 'identified_params.mat'), ...
    'alpha','gamma','r2','B0','DT','ttft_B0','beta_eff','beta_q', ...
    'B_SWEEP','l_mean','l_std','SERVER');
fprintf('[save] identified_params.mat\n');

% -------------------------------------------------------------------------
% Plot
% -------------------------------------------------------------------------
fig = figure('Visible','off','Position',[100 100 900 500]);

subplot(1,2,1);
errorbar(B_SWEEP, l_mean, l_std, 'bo', 'LineWidth', 1.2, 'CapSize', 5);
hold on;
b_fine = linspace(0.5, max(B_SWEEP)+0.5, 300);
plot(b_fine, alpha*b_fine + gamma*b_fine.^2, 'r-', 'LineWidth', 2);
plot(B0, ttft_B0, 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
x_tan = [B0-1.5, B0+1.5];
plot(x_tan, ttft_B0 + beta_eff*(x_tan-B0), 'g--', 'LineWidth', 1.5);
xlabel('Batch size B'); ylabel('l_{total} [ms]');
title(sprintf('Stage 2: TTFT(B)\n\\alpha=%.3f  \\gamma=%.4f  R^2=%.3f', alpha, gamma, r2));
legend('Measured','Fit: \alphaB+\gammaB^2','B_0','Tangent at B_0','Location','northwest');
grid on;

subplot(1,2,2);
q_range = 0:0.5:20;
colors  = lines(numel(B_SWEEP));
for bi = 1:numel(B_SWEEP)
    b = B_SWEEP(bi);
    ttft_b   = alpha*b + gamma*b^2;
    l_total  = ttft_b + (q_range/b)*DT*1000;
    plot(q_range, l_total, 'Color', colors(bi,:), 'LineWidth', 1.3, ...
         'DisplayName', sprintf('B=%d', b));
    hold on;
end
yline(600, 'k--', 'LineWidth', 1.5, 'DisplayName', 'L_{target}=600ms');
xlabel('q_{sw} [req]'); ylabel('l_{total} [ms]');
title('Full plant: l_{total}(B,q) = TTFT(B) + (q/B)\cdotdt\cdot1000');
legend('Location','northwest','FontSize',7);
grid on;

sgtitle(sprintf('Chapter 6 — Plant Identification  (Intel Mac)\n\\beta_q=%.1f ms/req  \\beta_{eff}=%.2f ms/req', ...
    beta_q, beta_eff));

saveas(fig, fullfile(out_dir, 'ch6_stage2_b_sweep.png'));
fprintf('[plot] ch6_stage2_b_sweep.png\n\n');

% =========================================================================
% Helper functions
% =========================================================================
function m = server_get(server, path)
    cmd = sprintf('curl -s "%s%s"', server, path);
    [~, out] = system(cmd);
    m = jsondecode(strtrim(out));
end

function r = server_post(server, path, data)
    body = jsonencode(data);
    % Escape double quotes for shell
    body = strrep(body, '"', '\"');
    cmd  = sprintf('curl -s -X POST "%s%s" -H "Content-Type: application/json" -d "%s"', ...
                   server, path, body);
    [~, out] = system(cmd);
    try
        r = jsondecode(strtrim(out));
    catch
        r = struct('raw', out);
    end
end
