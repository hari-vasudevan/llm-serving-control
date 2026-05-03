%% characterise_plant.m  --  Chapter 8 plant identification from MATLAB
%
% This script talks directly to the Chapter 8 Modal wrapper and identifies a
% Chapter 2 style plant using the wrapper-controlled batch size B.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

SERVER = 'https://hvasudevan--chapter-8-vllm-wrapper-serve.modal.run';
DT = 1.0;
% Quick-trial profile to complete inside the current MATLAB tool runtime.
B_SWEEP = [1 4 8];
N_REPS = 2;
PROMPT_REPEAT = 192;
MAX_TOKENS = 32;
WARMUP_ENQS = 4;
TIMEOUT_S = 120;
TRACE_FILE = fullfile(fileparts(mfilename('fullpath')), 'characterise_trace.txt');

if exist(TRACE_FILE, 'file')
    delete(TRACE_FILE);
end
diary(TRACE_FILE);
diary on;

fprintf('=== Chapter 8 Characterise Plant ===\n');
fprintf('SERVER=%s\nDT=%.1fs\nB_SWEEP=%s\n\n', SERVER, DT, mat2str(B_SWEEP));

smoke = srv_get(SERVER, '/health');
disp(smoke);

fprintf('[reset] clearing server buffers\n');
srv_post(SERVER, '/reset', struct('source', 'matlab_characterise'));
pause(1);

fprintf('[warmup] %d requests at B=4\n', WARMUP_ENQS);
srv_post(SERVER, '/control', struct('B', 4, 'source', 'matlab_characterise', 'note', 'warmup'));
for i = 1:WARMUP_ENQS
    enqueue_prompt(SERVER, prompt_library(i), PROMPT_REPEAT, MAX_TOKENS, 'warmup');
end
wait_for_delta_completions(SERVER, WARMUP_ENQS, TIMEOUT_S);
disp(srv_get(SERVER, '/metrics'));

l_mean = NaN(size(B_SWEEP));
ttft_mean = NaN(size(B_SWEEP));
qwait_mean = NaN(size(B_SWEEP));
vllm_waiting = NaN(size(B_SWEEP));
vllm_running = NaN(size(B_SWEEP));
rep_logs = cell(numel(B_SWEEP), N_REPS);

for bi = 1:numel(B_SWEEP)
    b = B_SWEEP(bi);
    fprintf('\n=== Sweep B=%d ===\n', b);
    rep_l = NaN(1, N_REPS);
    rep_t = NaN(1, N_REPS);
    rep_q = NaN(1, N_REPS);

    for r = 1:N_REPS
        fprintf('[B=%d rep=%d] reset and control\n', b, r);
        srv_post(SERVER, '/reset', struct('source', 'matlab_characterise'));
        pause(0.5);
        srv_post(SERVER, '/control', struct('B', b, 'source', 'matlab_characterise', ...
            'note', sprintf('characterise rep %d', r)));

        for k = 1:b
            enqueue_prompt(SERVER, prompt_library(k + 10 * r + b), PROMPT_REPEAT, MAX_TOKENS, ...
                sprintf('char_B%d_rep%d', b, r));
        end

        wait_for_delta_completions(SERVER, b, TIMEOUT_S);
        pause(1.0);
        m = srv_get(SERVER, '/metrics');
        rep_logs{bi, r} = m;
        rep_l(r) = metric_or_nan(m, 'l_mean_ms');
        rep_t(r) = metric_or_nan(m, 'ttft_mean_ms');
        rep_q(r) = metric_or_nan(m, 'queue_wait_mean_ms');
        fprintf('[B=%d rep=%d] l_mean=%.1f ttft=%.1f q_wait=%.1f q_sw=%d lambda_10s=%.2f\n', ...
            b, r, rep_l(r), rep_t(r), rep_q(r), m.q_sw, m.lambda_10s_est);
    end

    l_mean(bi) = mean(rep_l, 'omitnan');
    ttft_mean(bi) = mean(rep_t, 'omitnan');
    qwait_mean(bi) = mean(rep_q, 'omitnan');
    vllm_waiting(bi) = metric_or_nan(rep_logs{bi, end}, 'vllm_num_requests_waiting');
    vllm_running(bi) = metric_or_nan(rep_logs{bi, end}, 'vllm_num_requests_running');
end

valid = ~isnan(l_mean);
beta_B = NaN;
gamma_B = NaN;
if nnz(valid) >= 3
    Bv = B_SWEEP(valid)';
    Lv = l_mean(valid)';
    coeff = [Bv, Bv.^2] \ Lv;
    beta_B = coeff(1);
    gamma_B = coeff(2);
    fprintf('\n[fit] l_mean(B) = %.4f*B + %.4f*B^2\n', beta_B, gamma_B);
end

B0 = 16;
lambda_mean = 12;
q0 = 5;
q_max = 200;
L_target = interp1(B_SWEEP(valid), l_mean(valid), B0, 'linear', 'extrap');
beta_q = DT * 1000 / max(B0, 1);

fprintf('\n[operating point]\n');
fprintf('B0=%d lambda_mean=%.1f q0=%.1f beta_q=%.2f L_target=%.1f\n', ...
    B0, lambda_mean, q0, beta_q, L_target);

save(fullfile(fileparts(mfilename('fullpath')), 'identified_params.mat'), ...
    'SERVER', 'DT', 'B_SWEEP', 'N_REPS', 'PROMPT_REPEAT', 'MAX_TOKENS', ...
    'l_mean', 'ttft_mean', 'qwait_mean', 'vllm_waiting', 'vllm_running', ...
    'beta_B', 'gamma_B', 'B0', 'lambda_mean', 'q0', 'q_max', 'L_target', 'beta_q', ...
    'rep_logs');
fprintf('[save] identified_params.mat\n');

fig = figure('Visible', 'off', 'Position', [50 50 1100 420]);
subplot(1, 2, 1);
plot(B_SWEEP, l_mean, 'bo-', 'LineWidth', 1.5);
hold on;
plot(B_SWEEP, ttft_mean, 'rs-', 'LineWidth', 1.5);
plot(B_SWEEP, qwait_mean, 'kd-', 'LineWidth', 1.5);
grid on;
xlabel('B');
ylabel('Latency [ms]');
legend('l_{mean}', 'TTFT', 'queue wait', 'Location', 'northwest');
title('Chapter 8 characterization sweep');

subplot(1, 2, 2);
plot(B_SWEEP, vllm_waiting, 'm^-', 'LineWidth', 1.5);
hold on;
plot(B_SWEEP, vllm_running, 'cv-', 'LineWidth', 1.5);
grid on;
xlabel('B');
ylabel('Native vLLM counts');
legend('num\_requests\_waiting', 'num\_requests\_running', 'Location', 'northwest');
title('Native vLLM metrics during sweep');

saveas(fig, fullfile(fileparts(mfilename('fullpath')), 'ch8_characterise.png'));
fprintf('[save] ch8_characterise.png\n');

diary off;


function enqueue_prompt(server, prompt, prompt_repeat, max_tokens, source_tag)
payload = struct( ...
    'prompt', prompt, ...
    'prompt_repeat', prompt_repeat, ...
    'max_tokens', max_tokens, ...
    'temperature', 0.0, ...
    'source', source_tag, ...
    'client_ts', char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZZZZZ')));
srv_post(server, '/enqueue', payload);
end

function wait_for_delta_completions(server, delta_needed, timeout_s)
t0 = tic;
start_metrics = srv_get(server, '/metrics');
start_completed = start_metrics.completed;
while toc(t0) < timeout_s
    pause(1.0);
    m = srv_get(server, '/metrics');
    if (m.completed - start_completed) >= delta_needed
        fprintf('[wait] completed delta reached: %d/%d\n', m.completed - start_completed, delta_needed);
        return;
    end
    fprintf('[wait] completed delta=%d/%d q=%d B=%d lambda_10s=%.2f\n', ...
        m.completed - start_completed, delta_needed, m.q_sw, m.B_current, m.lambda_10s_est);
end
error('Timed out waiting for completions');
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
