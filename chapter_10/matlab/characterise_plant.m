%% characterise_plant.m  --  Chapter 8 plant identification from MATLAB
%
% Characterise the wrapper-queue plant using smoother within-tick arrivals
% and the wrapper's tick-averaged queue metrics. This makes the Chapter 8
% plant much closer to the Chapter 2 cascade structure.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

SERVER = 'https://hvasudevan--chapter-8-vllm-wrapper-serve.modal.run';
DT = 1.0;
B_SWEEP = [64 80 96 112 128];
LAMBDA_SWEEP = [72 84 96 108 120];
PROMPT_REPEAT = 256;
MAX_TOKENS = 64;
TRACE_FILE = fullfile(fileparts(mfilename('fullpath')), 'characterise_trace.txt');
TICKS_PER_POINT = 8;
SETTLE_TICKS = 3;
SPREAD_BINS = 6;
Q_TARGET_NOMINAL = 24;
Q_REF_MIN = 10;
Q_REF_MAX = 90;
Q_MAX = 180;

if exist(TRACE_FILE, 'file')
    delete(TRACE_FILE);
end
diary(TRACE_FILE);
diary on;

fprintf('=== Chapter 8 Characterise Plant ===\n');
fprintf('SERVER=%s\nDT=%.1fs\nB_SWEEP=%s\nLAMBDA_SWEEP=%s\n\n', ...
    SERVER, DT, mat2str(B_SWEEP), mat2str(LAMBDA_SWEEP));

smoke = srv_get(SERVER, '/health');
disp(smoke);

lambda_char = 96;
l_mean = NaN(size(B_SWEEP));
ttft_mean = NaN(size(B_SWEEP));
qwait_mean = NaN(size(B_SWEEP));
qmean_mean = NaN(size(B_SWEEP));
qmax_mean = NaN(size(B_SWEEP));
service_rate_mean = NaN(size(B_SWEEP));
rep_logs = cell(numel(B_SWEEP), 1);

for bi = 1:numel(B_SWEEP)
    b = B_SWEEP(bi);
    fprintf('\n=== Sweep B=%d at lambda=%d ===\n', b, lambda_char);
    rep_logs{bi} = run_operating_block(SERVER, b, lambda_char, TICKS_PER_POINT, SETTLE_TICKS, ...
        DT, SPREAD_BINS, PROMPT_REPEAT, MAX_TOKENS, sprintf('char_B_%03d', b));
    summary = summarise_block(rep_logs{bi}, SETTLE_TICKS);
    l_mean(bi) = summary.l_mean_ms;
    ttft_mean(bi) = summary.ttft_mean_ms;
    qwait_mean(bi) = summary.qwait_mean_ms;
    qmean_mean(bi) = summary.q_mean_tick;
    qmax_mean(bi) = summary.q_max_tick;
    service_rate_mean(bi) = summary.service_rate_tick;
    fprintf('[B=%d] q_mean=%.2f q_max=%.2f l_mean=%.1f ttft=%.1f q_wait=%.1f sr=%.2f\n', ...
        b, qmean_mean(bi), qmax_mean(bi), l_mean(bi), ttft_mean(bi), qwait_mean(bi), service_rate_mean(bi));
end

lambda_l_mean = NaN(size(LAMBDA_SWEEP));
lambda_q_mean = NaN(size(LAMBDA_SWEEP));
lambda_q_max = NaN(size(LAMBDA_SWEEP));
lambda_service_rate = NaN(size(LAMBDA_SWEEP));
lambda_logs = cell(numel(LAMBDA_SWEEP), 1);
B0_probe = 96;

fprintf('\n=== Load sweep at fixed B=%d ===\n', B0_probe);
for li = 1:numel(LAMBDA_SWEEP)
    lam = LAMBDA_SWEEP(li);
    fprintf('[lambda sweep] B=%d lambda=%d\n', B0_probe, lam);
    lambda_logs{li} = run_operating_block(SERVER, B0_probe, lam, TICKS_PER_POINT, SETTLE_TICKS, ...
        DT, SPREAD_BINS, PROMPT_REPEAT, MAX_TOKENS, sprintf('char_lambda_%03d', lam));
    summary = summarise_block(lambda_logs{li}, SETTLE_TICKS);
    lambda_l_mean(li) = summary.l_mean_ms;
    lambda_q_mean(li) = summary.q_mean_tick;
    lambda_q_max(li) = summary.q_max_tick;
    lambda_service_rate(li) = summary.service_rate_tick;
    fprintf('[lambda=%d] q_mean=%.2f q_max=%.2f l_mean=%.1f service_rate=%.2f\n', ...
        lam, lambda_q_mean(li), lambda_q_max(li), lambda_l_mean(li), lambda_service_rate(li));
end

q_fit = polyfit(B_SWEEP(:), qmean_mean(:), 1);
beta_q = max(-q_fit(1), 0.5);
fprintf('\n[fit] q_mean(B) = %.4f * B + %.4f\n', q_fit(1), q_fit(2));
fprintf('[fit] beta_q=%.4f queue_units_per_batch\n', beta_q);

lambda_fit = polyfit(lambda_q_mean(:), lambda_l_mean(:), 1);
beta_l = max(lambda_fit(1), 1.0);
fprintf('[fit] l_mean(q_mean) = %.4f * q + %.4f\n', lambda_fit(1), lambda_fit(2));
fprintf('[fit] beta_l=%.4f latency_ms_per_queue_unit\n', beta_l);

B0 = B0_probe;
target_idx = pick_operating_index(lambda_q_mean, Q_TARGET_NOMINAL);
lambda_mean = LAMBDA_SWEEP(target_idx);
q0 = max(lambda_q_mean(target_idx), Q_REF_MIN);
L0 = lambda_l_mean(target_idx);
L_target = L0;

fprintf('\n[operating point]\n');
fprintf('B0=%d lambda_mean=%.1f q0=%.2f L0=%.1f beta_q=%.2f beta_l=%.2f\n', ...
    B0, lambda_mean, q0, L0, beta_q, beta_l);

save(fullfile(fileparts(mfilename('fullpath')), 'identified_params.mat'), ...
    'SERVER', 'DT', 'B_SWEEP', 'LAMBDA_SWEEP', 'PROMPT_REPEAT', 'MAX_TOKENS', ...
    'TICKS_PER_POINT', 'SETTLE_TICKS', 'SPREAD_BINS', ...
    'l_mean', 'ttft_mean', 'qwait_mean', 'qmean_mean', 'qmax_mean', 'service_rate_mean', ...
    'lambda_l_mean', 'lambda_q_mean', 'lambda_q_max', 'lambda_service_rate', ...
    'rep_logs', 'lambda_logs', ...
    'beta_q', 'beta_l', 'B0', 'lambda_mean', 'q0', 'Q_REF_MIN', 'Q_REF_MAX', 'Q_MAX', 'L0', 'L_target');
fprintf('[save] identified_params.mat\n');

fig = figure('Visible', 'off', 'Position', [50 50 1180 460]);
subplot(1, 2, 1);
yyaxis left;
plot(B_SWEEP, qmean_mean, 'ko-', 'LineWidth', 1.5); hold on;
plot(B_SWEEP, qmax_mean, 'g^-', 'LineWidth', 1.2);
ylabel('Queue');
yyaxis right;
plot(B_SWEEP, l_mean, 'bo-', 'LineWidth', 1.5); hold on;
plot(B_SWEEP, ttft_mean, 'rs-', 'LineWidth', 1.2);
grid on;
xlabel('B');
ylabel('Latency [ms]');
legend('q_{mean,tick}', 'q_{max,tick}', 'l_{mean}', 'TTFT', 'Location', 'northwest');
title('Characterisation at fixed \lambda');

subplot(1, 2, 2);
yyaxis left;
plot(LAMBDA_SWEEP, lambda_q_mean, 'm^-', 'LineWidth', 1.5); hold on;
plot(LAMBDA_SWEEP, lambda_q_max, 'k--', 'LineWidth', 1.2);
ylabel('Queue');
yyaxis right;
plot(LAMBDA_SWEEP, lambda_l_mean, 'cv-', 'LineWidth', 1.5);
grid on;
xlabel('\lambda [req/s]');
ylabel('Latency [ms]');
legend('q_{mean,tick}', 'q_{max,tick}', 'l_{mean}', 'Location', 'northwest');
title('Load calibration at fixed B');

saveas(fig, fullfile(fileparts(mfilename('fullpath')), 'ch8_characterise.png'));
fprintf('[save] ch8_characterise.png\n');

diary off;


function logs = run_operating_block(server, B_cmd, lambda_cmd, n_ticks, settle_ticks, dt, spread_bins, prompt_repeat, max_tokens, source_prefix)
srv_post(server, '/reset', struct('source', source_prefix));
pause(1.0);
srv_post(server, '/control', struct('B', B_cmd, 'source', source_prefix, 'note', 'block start'));
pause(0.5);

logs = cell(n_ticks, 1);
prompt_idx = 1;
for k = 1:n_ticks
    fprintf('\n[block %s] tick=%d/%d B=%d lambda=%.2f\n', source_prefix, k, n_ticks, B_cmd, lambda_cmd);
    inject_spread_arrivals(server, lambda_cmd, dt, spread_bins, prompt_library(prompt_idx), prompt_repeat, max_tokens, ...
        sprintf('%s_tick_%02d', source_prefix, k));
    prompt_idx = prompt_idx + 1;
    m = srv_get(server, '/metrics');
    logs{k} = m;
    fprintf('[metrics] q_mean=%.2f q_max=%.2f arrivals=%d comps=%d l_mean=%.1f ttft=%.1f q_wait=%.1f B=%d\n', ...
        metric_or_nan(m, 'q_mean_tick'), metric_or_nan(m, 'q_max_tick'), metric_or_nan(m, 'arrivals_tick'), ...
        metric_or_nan(m, 'completions_tick'), metric_or_nan(m, 'l_mean_ms'), metric_or_nan(m, 'ttft_mean_ms'), ...
        metric_or_nan(m, 'queue_wait_mean_ms'), metric_or_nan(m, 'B_current'));
    if k == settle_ticks
        fprintf('[block %s] settle window completed\n', source_prefix);
    end
end
wait_for_quiet_tick(server, 20.0);
end

function summary = summarise_block(logs, settle_ticks)
start_idx = min(numel(logs), settle_ticks + 1);
use = logs(start_idx:end);
summary = struct();
summary.q_mean_tick = average_metric(use, 'q_mean_tick');
summary.q_max_tick = average_metric(use, 'q_max_tick');
summary.service_rate_tick = average_metric(use, 'service_rate_tick');
summary.l_mean_ms = average_metric(use, 'l_mean_ms');
summary.ttft_mean_ms = average_metric(use, 'ttft_mean_ms');
summary.qwait_mean_ms = average_metric(use, 'queue_wait_mean_ms');
end

function idx = pick_operating_index(q_values, q_target)
[~, idx] = min(abs(q_values - q_target));
end

function value = average_metric(logs, field_name)
vals = NaN(numel(logs), 1);
for i = 1:numel(logs)
    vals(i) = metric_or_nan(logs{i}, field_name);
end
value = mean(vals, 'omitnan');
end

function inject_spread_arrivals(server, lambda_cmd, dt, spread_bins, prompt, prompt_repeat, max_tokens, source_tag)
arrivals = poissrnd(lambda_cmd * dt);
fprintf('[arrivals] lambda=%.2f dt=%.2f sampled=%d bins=%d\n', lambda_cmd, dt, arrivals, spread_bins);
if arrivals <= 0
    pause(dt);
    return;
end

sub_times = sort(rand(arrivals, 1) * max(dt - 0.05, 0.05));
edges = linspace(0, dt, spread_bins + 1);
counts = histcounts(sub_times, edges);
clock0 = tic;
for bi = 1:spread_bins
    target_t = edges(bi);
    elapsed = toc(clock0);
    if target_t > elapsed
        pause(target_t - elapsed);
    end
    if counts(bi) > 0
        enqueue_burst(server, prompt, counts(bi), prompt_repeat, max_tokens, sprintf('%s_bin_%02d', source_tag, bi));
    end
end
elapsed = toc(clock0);
if elapsed < dt
    pause(dt - elapsed);
end
end

function enqueue_burst(server, prompt, count, prompt_repeat, max_tokens, source_tag)
payload = struct( ...
    'prompt', prompt, ...
    'count', count, ...
    'prompt_repeat', prompt_repeat, ...
    'max_tokens', max_tokens, ...
    'temperature', 0.0, ...
    'source', source_tag, ...
    'client_ts', char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZZZZZ')));
srv_post(server, '/enqueue_batch', payload);
end

function wait_for_quiet_tick(server, timeout_s)
t0 = tic;
while toc(t0) < timeout_s
    m = srv_get(server, '/metrics');
    if metric_or_nan(m, 'q_sw') <= 0.5 && metric_or_nan(m, 'vllm_num_requests_running') <= 0.5
        fprintf('[quiet] wrapper and vLLM are idle\n');
        return;
    end
    pause(1.0);
end
fprintf('[quiet] timeout waiting for idle; continuing\n');
end

function out = srv_get(server, path)
cmd = sprintf('curl -sS "%s%s"', server, path);
fprintf('[MATLAB GET] %s\n', cmd);
[status, raw] = system(cmd);
assert(status == 0, 'GET failed: %s', raw);
fprintf('[MATLAB GET <-] %s\n', strtrim(raw));
out = jsondecode(strtrim(raw));
end

function out = srv_post(server, path, payload)
body = jsonencode(payload);
body_escaped = strrep(body, '"', '\"');
cmd = sprintf('curl -sS -X POST "%s%s" -H "Content-Type: application/json" -d "%s"', ...
    server, path, body_escaped);
fprintf('[MATLAB POST] %s\n', cmd);
[status, raw] = system(cmd);
assert(status == 0, 'POST failed: %s', raw);
fprintf('[MATLAB POST <-] %s\n', strtrim(raw));
out = jsondecode(strtrim(raw));
end

function p = prompt_library(i)
base = { ...
    'Explain what a queue is in one sentence.', ...
    'List three uses of control theory in computing.', ...
    'Give a short definition of GPU batching.', ...
    'What is first-token latency?', ...
    'Describe a scheduler in one sentence.', ...
    'Explain why queueing can increase latency.', ...
    'What is a disturbance in feedback control?', ...
    'Why do inference servers batch requests?'};
p = base{mod(i - 1, numel(base)) + 1};
end

function value = metric_or_nan(s, field_name)
if isfield(s, field_name)
    value = s.(field_name);
    if isempty(value)
        value = NaN;
    end
else
    value = NaN;
end
end
