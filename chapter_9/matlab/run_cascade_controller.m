%% run_cascade_controller.m -- Chapter 9 Modal-side closed-loop cascade
%
% MATLAB designs/uploads the controller. Modal owns the closed-loop clock,
% arrival generator, scheduler, metrics snapshot, and GPU dispatch.

clear; clc;
addpath(fileparts(mfilename('fullpath')));
load(fullfile(fileparts(mfilename('fullpath')), 'controller_params.mat'));

TRACE_FILE = fullfile(fileparts(mfilename('fullpath')), 'run_cascade_trace.txt');
if exist(TRACE_FILE, 'file'); delete(TRACE_FILE); end
diary(TRACE_FILE); diary on;

DT = perturbed.dt;
SEGMENTS = struct( ...
    'label', {'steady', 'spike_1', 'recover', 'spike_2', 'steady_restore'}, ...
    'ticks', {10, 6, 6, 6, 10}, ...
    'lambda', {perturbed.lambda_mean, ...
               min(perturbed.B_max, perturbed.lambda_mean * 1.10), ...
               max(perturbed.lambda_mean * 0.82, 1), ...
               min(3200, perturbed.lambda_mean * 1.22), ...
               perturbed.lambda_mean});

fprintf('=== Chapter 9 Closed-loop Cascade ===\n');
fprintf('Modal-side clock. Inner loop: B->q. Outer loop: q_ref->L_mean.\n');
fprintf('Target L_mean=%.2f ms, B0=%d, Bmax=%d, lambda0=%.2f\n', ...
    controller.outer_c.L_mean_target, controller.inner_c.B0, controller.inner_c.B_max, perturbed.lambda_mean);
for i = 1:numel(SEGMENTS)
    fprintf('segment %d: label=%s ticks=%d lambda=%.2f\n', ...
        i, SEGMENTS(i).label, SEGMENTS(i).ticks, SEGMENTS(i).lambda);
end

upload_controller_if_available(SERVER);
payload = struct( ...
    'dt', DT, ...
    'seed', 9, ...
    'segments', SEGMENTS, ...
    'source', 'matlab_run_cascade_controller');
result = srv_post(SERVER, '/run_closed_loop', payload);
assert(strcmp(result.status, 'ok'), 'run_closed_loop failed');

remote_log = result.run_log;
log_tick = extract_field(remote_log, 'tick');
log_lambda = extract_field(remote_log, 'lambda');
log_B = extract_field(remote_log, 'B');
log_q = extract_field(remote_log, 'q');
log_q_ref = extract_field(remote_log, 'q_ref');
log_L = extract_field(remote_log, 'L_mean');
log_L_p95 = extract_field(remote_log, 'L_p95');
log_service = extract_field(remote_log, 'service_ms');
log_arrivals = extract_field(remote_log, 'arrivals');
log_completions = extract_field(remote_log, 'completions');
log_label = extract_labels(remote_log);

for i = 1:numel(log_tick)
    fprintf('[tick=%03d] seg=%s lambda=%.2f arrivals=%.0f q=%.2f q_ref=%.2f B=%.0f L=%.2f Lp95=%.2f service=%.2f comps=%.2f\n', ...
        log_tick(i), log_label(i), log_lambda(i), log_arrivals(i), log_q(i), log_q_ref(i), ...
        log_B(i), log_L(i), log_L_p95(i), log_service(i), log_completions(i));
end

run_log = struct( ...
    'tick', log_tick, ...
    'lambda', log_lambda, ...
    'B', log_B, ...
    'q', log_q, ...
    'q_ref', log_q_ref, ...
    'L_mean', log_L, ...
    'L_p95', log_L_p95, ...
    'service_ms', log_service, ...
    'arrivals', log_arrivals, ...
    'completions', log_completions, ...
    'label', {cellstr(log_label)});
save(fullfile(fileparts(mfilename('fullpath')), 'run_log.mat'), 'run_log', 'SEGMENTS', 'result');
fprintf('[save] run_log.mat\n');

fig = figure('Visible', 'off', 'Position', [50 50 1200 760]);
subplot(4,1,1);
plot(log_tick, log_L, 'b-', 'LineWidth', 1.5); hold on;
yline(controller.outer_c.L_mean_target, 'r--', 'LineWidth', 1.2);
grid on; ylabel('L_{mean} [ms]'); title('Chapter 9 Modal-side cascade: q_{ref}->L_{mean}, B->q');

subplot(4,1,2);
plot(log_tick, log_q, 'k-', 'LineWidth', 1.5); hold on;
plot(log_tick, log_q_ref, 'g--', 'LineWidth', 1.2);
grid on; ylabel('q');
legend('q', 'q_{ref}', 'Location', 'northwest');

subplot(4,1,3);
stairs(log_tick, log_B, 'm-', 'LineWidth', 1.5); grid on;
ylabel('B');

subplot(4,1,4);
stairs(log_tick, log_lambda, 'c-', 'LineWidth', 1.4); hold on;
plot(log_tick, log_completions, 'Color', [0.2 0.5 0.2], 'LineWidth', 1.2);
grid on; ylabel('\lambda / completions'); xlabel('tick');
legend('\lambda', 'completions', 'Location', 'northwest');

saveas(fig, fullfile(fileparts(mfilename('fullpath')), 'ch9_closed_loop.png'));
fprintf('[save] ch9_closed_loop.png\n');
diary off;


function vals = extract_field(s, field_name)
vals = NaN(numel(s), 1);
for i = 1:numel(s)
    if isfield(s(i), field_name) && ~isempty(s(i).(field_name))
        vals(i) = double(s(i).(field_name));
    end
end
end

function labels = extract_labels(s)
labels = strings(numel(s), 1);
for i = 1:numel(s)
    if isfield(s(i), 'label') && ~isempty(s(i).label)
        labels(i) = string(s(i).label);
    end
end
end

function out = srv_post(server, path, payload)
body = jsonencode(payload);
body_escaped = strrep(body, '"', '\"');
cmd = sprintf('curl -sS -X POST "%s%s" -H "Content-Type: application/json" -d "%s"', ...
    server, path, body_escaped);
[status, raw] = system(cmd);
assert(status == 0, 'POST failed: %s', raw);
assert(strlength(strtrim(raw)) > 0, 'POST returned empty response for %s%s', server, path);
out = jsondecode(strtrim(raw));
end

function upload_controller_if_available(server)
xml_path = fullfile(fileparts(mfilename('fullpath')), 'controller_config.xml');
if exist(xml_path, 'file') && ~contains(server, 'REPLACE_WITH_MODAL')
    payload = struct('xml', fileread(xml_path), 'source', 'matlab_run_cascade_controller');
    out = srv_post(server, '/controller_config', payload);
    fprintf('[upload] controller_config status=%s\n', string(out.status));
end
end
