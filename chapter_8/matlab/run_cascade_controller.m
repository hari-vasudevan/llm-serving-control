%% run_cascade_controller.m  --  Chapter 8 closed-loop cascade controller

clear; clc;
addpath(fileparts(mfilename('fullpath')));

load(fullfile(fileparts(mfilename('fullpath')), 'controller_params.mat'));

TRACE_FILE = fullfile(fileparts(mfilename('fullpath')), 'run_cascade_trace.txt');
if exist(TRACE_FILE, 'file')
    delete(TRACE_FILE);
end
diary(TRACE_FILE);
diary on;

DT = perturbed.dt;
lambda_spike_1 = round(lambda_mean * 1.25);
lambda_spike_2 = round(lambda_mean * 1.5);
lambda_recover = max(round(lambda_mean * 0.85), 1);
SEGMENTS = struct( ...
    'ticks', {20, 14, 16, 14, 12}, ...
    'lambda', {lambda_mean, lambda_spike_1, lambda_recover, lambda_spike_2, lambda_mean}, ...
    'label', {'steady', 'spike_1', 'recover_1', 'spike_2', 'steady_restore'});
PROMPT_REPEAT_VALUE = PROMPT_REPEAT;
MAX_TOKENS_VALUE = MAX_TOKENS;
SPREAD_BINS = 6;

fprintf('=== Chapter 8 Run Cascade ===\n');
fprintf('Target L_mean=%.1f ms\n', controller.outer_c.L_mean_target);
for i = 1:numel(SEGMENTS)
    fprintf('segment %d: ticks=%d lambda=%.2f label=%s\n', i, SEGMENTS(i).ticks, SEGMENTS(i).lambda, SEGMENTS(i).label);
end

srv_post(SERVER, '/reset', struct('source', 'matlab_run'));
pause(1);
srv_post(SERVER, '/control', struct('B', round(controller.inner_c.B0), 'source', 'matlab_run', 'note', 'initial B'));

xi_q = 0;
xi_l = 0;
q_ref = controller.outer_c.q0;
prompt_idx = 1;
N = sum([SEGMENTS.ticks]);

log_tick = zeros(N, 1);
log_lambda = zeros(N, 1);
log_q_sw = zeros(N, 1);
log_q_ref = zeros(N, 1);
log_B = zeros(N, 1);
log_l_mean = zeros(N, 1);
log_ttft = zeros(N, 1);
log_q_wait = zeros(N, 1);
log_vllm_waiting = zeros(N, 1);
log_vllm_running = zeros(N, 1);
log_e_l = zeros(N, 1);
log_e_q = zeros(N, 1);
log_label = strings(N, 1);

tick = 0;
for si = 1:numel(SEGMENTS)
    seg = SEGMENTS(si);
    for local_tick = 1:seg.ticks
        tick = tick + 1;
        t0 = tic;

        m = srv_get(SERVER, '/metrics');
        q_sw = metric_or_nan(m, 'q_mean_tick');
        q_pk = metric_or_nan(m, 'q_max_tick');
        l_mean_ms = metric_or_nan(m, 'l_mean_ms');
        ttft_ms = metric_or_nan(m, 'ttft_mean_ms');
        q_wait_ms = metric_or_nan(m, 'queue_wait_mean_ms');
        vllm_wait = metric_or_nan(m, 'vllm_num_requests_waiting');
        vllm_run = metric_or_nan(m, 'vllm_num_requests_running');
        arrivals_tick = metric_or_nan(m, 'arrivals_tick');
        completions_tick = metric_or_nan(m, 'completions_tick');
        if isnan(l_mean_ms)
            l_mean_ms = controller.outer_c.L_mean_target;
        end
        if isnan(q_sw)
            q_sw = controller.outer_c.q0;
        end

        e_l = controller.outer_c.L_mean_target - l_mean_ms;
        q_ref_unsat = controller.outer_c.q0 + controller.outer_c.K_i_l * (xi_l + e_l);
        q_ref = clamp(q_ref_unsat, controller.outer_c.q_min, controller.outer_c.q_max);
        if ~(q_ref == controller.outer_c.q_min && e_l < 0) && ~(q_ref == controller.outer_c.q_max && e_l > 0)
            xi_l = clamp(xi_l + e_l, controller.outer_c.xi_min, controller.outer_c.xi_max);
            q_ref = clamp(controller.outer_c.q0 + controller.outer_c.K_i_l * xi_l, controller.outer_c.q_min, controller.outer_c.q_max);
        end

        e_q = q_ref - q_sw;
        B_unsat = controller.inner_c.B0 - controller.inner_c.K_q * (q_sw - controller.outer_c.q0) ...
            - controller.inner_c.K_i_q * (xi_q + e_q);
        B_cmd = round(clamp(B_unsat, controller.inner_c.B_min, controller.inner_c.B_max));
        if ~(B_cmd == controller.inner_c.B_min && e_q > 0) && ~(B_cmd == controller.inner_c.B_max && e_q < 0)
            xi_q = clamp(xi_q + e_q, controller.inner_c.xi_min, controller.inner_c.xi_max);
            B_cmd = round(clamp(controller.inner_c.B0 - controller.inner_c.K_q * (q_sw - controller.outer_c.q0) ...
                - controller.inner_c.K_i_q * xi_q, controller.inner_c.B_min, controller.inner_c.B_max));
        end

        fprintf('\n[tick=%d] seg=%s lambda=%.2f q_mean=%.1f q_max=%.1f q_ref=%.1f arrivals_tick=%.1f comps_tick=%.1f l_mean=%.1f q_wait=%.1f B_cmd=%d\n', ...
            tick, seg.label, seg.lambda, q_sw, q_pk, q_ref, arrivals_tick, completions_tick, l_mean_ms, q_wait_ms, B_cmd);
        srv_post(SERVER, '/control', struct('B', B_cmd, 'source', 'matlab_run', ...
            'note', sprintf('tick %d q_ref=%.2f e_l=%.2f e_q=%.2f', tick, q_ref, e_l, e_q)));

        injected = inject_spread_arrivals(SERVER, seg.lambda, DT, SPREAD_BINS, ...
            prompt_library(prompt_idx), PROMPT_REPEAT_VALUE, MAX_TOKENS_VALUE, sprintf('run_tick_%03d', tick));
        prompt_idx = prompt_idx + 1;

        log_tick(tick) = tick;
        log_lambda(tick) = seg.lambda;
        log_q_sw(tick) = q_sw;
        log_q_ref(tick) = q_ref;
        log_B(tick) = B_cmd;
        log_l_mean(tick) = l_mean_ms;
        log_ttft(tick) = ttft_ms;
        log_q_wait(tick) = q_wait_ms;
        log_vllm_waiting(tick) = vllm_wait;
        log_vllm_running(tick) = vllm_run;
        log_e_l(tick) = e_l;
        log_e_q(tick) = e_q;
        log_label(tick) = string(seg.label);

        elapsed = toc(t0);
        if elapsed < DT
            pause(DT - elapsed);
        end
    end
end

wait_for_quiet_tick(SERVER, 30.0);

run_log = struct( ...
    'tick', log_tick, ...
    'lambda', log_lambda, ...
    'q_sw', log_q_sw, ...
    'q_ref', log_q_ref, ...
    'B', log_B, ...
    'l_mean_ms', log_l_mean, ...
    'ttft_ms', log_ttft, ...
    'q_wait_ms', log_q_wait, ...
    'vllm_waiting', log_vllm_waiting, ...
    'vllm_running', log_vllm_running, ...
    'e_l', log_e_l, ...
    'e_q', log_e_q, ...
    'label', {cellstr(log_label)});
save(fullfile(fileparts(mfilename('fullpath')), 'run_log.mat'), 'run_log', 'SEGMENTS');
fprintf('[save] run_log.mat\n');

fig = figure('Visible', 'off', 'Position', [40 40 1200 720]);
subplot(4,1,1);
plot(log_tick, log_l_mean, 'b-', 'LineWidth', 1.5); hold on;
yline(controller.outer_c.L_mean_target, 'r--', 'LineWidth', 1.2);
grid on; ylabel('l_{mean} ms'); title('Closed-loop Chapter 8');

subplot(4,1,2);
plot(log_tick, log_q_sw, 'k-', 'LineWidth', 1.5); hold on;
plot(log_tick, log_q_ref, 'g--', 'LineWidth', 1.2);
grid on; ylabel('q');

subplot(4,1,3);
plot(log_tick, log_B, 'm-', 'LineWidth', 1.5);
grid on; ylabel('B');

subplot(4,1,4);
stairs(log_tick, log_lambda, 'c-', 'LineWidth', 1.5); hold on;
plot(log_tick, log_vllm_waiting, 'Color', [0.8 0.4 0.1], 'LineWidth', 1.2);
plot(log_tick, log_vllm_running, 'Color', [0.2 0.5 0.2], 'LineWidth', 1.2);
grid on; ylabel('\lambda / vLLM'); xlabel('tick');
legend('\lambda', 'vllm waiting', 'vllm running', 'Location', 'northwest');

saveas(fig, fullfile(fileparts(mfilename('fullpath')), 'ch8_closed_loop.png'));
fprintf('[save] ch8_closed_loop.png\n');

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

function out = srv_get(server, path)
cmd = sprintf('curl -sS "%s%s"', server, path);
fprintf('[MATLAB GET] %s\n', cmd);
[status, raw] = system(cmd);
assert(status == 0, 'GET failed: %s', raw);
fprintf('[MATLAB GET <-] %s\n', strtrim(raw));
out = jsondecode(strtrim(raw));
end

function total = inject_spread_arrivals(server, lambda_cmd, dt, spread_bins, prompt, prompt_repeat, max_tokens, source_tag)
total = poissrnd(lambda_cmd * dt);
fprintf('[inject] lambda=%.2f sampled=%d bins=%d\n', lambda_cmd, total, spread_bins);
if total <= 0
    pause(dt);
    return;
end

sub_times = sort(rand(total, 1) * max(dt - 0.05, 0.05));
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
    'Explain feedback control in one short paragraph.', ...
    'Describe a queueing system in one short paragraph.', ...
    'What causes first-token latency in inference?', ...
    'Explain how batching changes GPU utilization.', ...
    'What is a disturbance rejection experiment?', ...
    'Define service rate in one paragraph.', ...
    'Explain the difference between TTFT and end-to-end latency.', ...
    'Why might a scheduler queue build up under spikes?'};
p = base{mod(i - 1, numel(base)) + 1};
end

function v = clamp(x, lo, hi)
v = max(lo, min(hi, x));
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
