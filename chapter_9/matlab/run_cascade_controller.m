%% run_cascade_controller.m -- Chapter 9 closed-loop cascade experiment

clear; clc;
addpath(fileparts(mfilename('fullpath')));
load(fullfile(fileparts(mfilename('fullpath')), 'controller_params.mat'));

TRACE_FILE = fullfile(fileparts(mfilename('fullpath')), 'run_cascade_trace.txt');
if exist(TRACE_FILE, 'file'); delete(TRACE_FILE); end
diary(TRACE_FILE); diary on;

DT = perturbed.dt;
SEGMENTS = struct( ...
    'ticks', {35, 25, 25, 25, 35}, ...
    'lambda', {perturbed.lambda_mean, min(perturbed.B_max * 1.12, 3200), max(perturbed.lambda_mean * 0.82, 1), min(perturbed.B_max * 1.25, 3200), perturbed.lambda_mean}, ...
    'label', {'steady', 'spike_1', 'recover', 'spike_2', 'steady_restore'});
N = sum([SEGMENTS.ticks]);

fprintf('=== Chapter 9 Closed-loop Cascade ===\n');
fprintf('Inner loop: B->q. Outer loop: q_ref->L_mean. Target L_mean=%.2f ms\n', controller.outer_c.L_mean_target);

upload_controller_if_available(SERVER);
srv_post(SERVER, '/reset', struct());
srv_post(SERVER, '/control', struct('B', round(controller.inner_c.B0)));

xi_q = 0;
xi_l = 0;
q_ref = controller.outer_c.q0;

log_tick = zeros(N,1);
log_lambda = zeros(N,1);
log_B = zeros(N,1);
log_q = zeros(N,1);
log_q_ref = zeros(N,1);
log_L = zeros(N,1);
log_L_p95 = zeros(N,1);
log_service = zeros(N,1);
log_arrivals = zeros(N,1);
log_completions = zeros(N,1);
log_label = strings(N,1);

tick = 0;
for si = 1:numel(SEGMENTS)
    seg = SEGMENTS(si);
    for sk = 1:seg.ticks
        tick = tick + 1;
        t0 = tic;

        m = srv_get(SERVER, '/metrics');
        q = metric_or_default(m, 'q_mean_tick', controller.outer_c.q0);
        L_mean = metric_or_default(m, 'l_mean_ms', controller.outer_c.L_mean_target);
        L_p95 = metric_or_default(m, 'l_p95_ms', controller.outer_c.L_p95_target);
        service_ms = metric_or_default(m, 'service_mean_ms', NaN);
        comps = metric_or_default(m, 'completions_tick', 0);

        % Outer loop: q_ref -> L_mean
        e_l = controller.outer_c.L_mean_target - L_mean;
        xi_l_trial = clamp(xi_l + e_l, controller.outer_c.xi_min, controller.outer_c.xi_max);
        q_ref_trial = clamp(controller.outer_c.q0 + controller.outer_c.K_i_l * xi_l_trial, ...
            controller.outer_c.q_min, controller.outer_c.q_max);
        if ~(q_ref_trial == controller.outer_c.q_min && e_l < 0) && ~(q_ref_trial == controller.outer_c.q_max && e_l > 0)
            xi_l = xi_l_trial;
            q_ref = q_ref_trial;
        end

        % Inner loop: B -> q
        e_q = q_ref - q;
        xi_q_trial = clamp(xi_q + e_q, controller.inner_c.xi_min, controller.inner_c.xi_max);
        B_unsat = controller.inner_c.B0 + controller.inner_c.K_q * (q - q_ref) ...
            - controller.inner_c.K_i_q * xi_q_trial;
        B_cmd = round(clamp(B_unsat, controller.inner_c.B_min, controller.inner_c.B_max));
        if ~(B_cmd == controller.inner_c.B_min && e_q > 0) && ~(B_cmd == controller.inner_c.B_max && e_q < 0)
            xi_q = xi_q_trial;
        end

        srv_post(SERVER, '/control', struct('B', B_cmd));
        arrivals = poissrnd(seg.lambda);
        srv_post(SERVER, '/enqueue_batch', struct('count', arrivals, 'source', sprintf('run_tick_%03d', tick)));

        fprintf('[tick=%03d] seg=%s lambda=%.2f arrivals=%d q=%.2f q_ref=%.2f B=%d L=%.2f Lp95=%.2f service=%.2f comps=%.2f\n', ...
            tick, seg.label, seg.lambda, arrivals, q, q_ref, B_cmd, L_mean, L_p95, service_ms, comps);

        log_tick(tick) = tick;
        log_lambda(tick) = seg.lambda;
        log_B(tick) = B_cmd;
        log_q(tick) = q;
        log_q_ref(tick) = q_ref;
        log_L(tick) = L_mean;
        log_L_p95(tick) = L_p95;
        log_service(tick) = service_ms;
        log_arrivals(tick) = arrivals;
        log_completions(tick) = comps;
        log_label(tick) = string(seg.label);

        elapsed = toc(t0);
        if elapsed < DT
            pause(DT - elapsed);
        end
    end
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
save(fullfile(fileparts(mfilename('fullpath')), 'run_log.mat'), 'run_log', 'SEGMENTS');
fprintf('[save] run_log.mat\n');

fig = figure('Visible', 'off', 'Position', [50 50 1200 760]);
subplot(4,1,1);
plot(log_tick, log_L, 'b-', 'LineWidth', 1.5); hold on;
yline(controller.outer_c.L_mean_target, 'r--', 'LineWidth', 1.2);
grid on; ylabel('L_{mean} [ms]'); title('Chapter 9 cascade: q_{ref}->L_{mean}, B->q');

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


function value = metric_or_default(s, field_name, default_value)
if isfield(s, field_name) && ~isempty(s.(field_name))
    value = double(s.(field_name));
else
    value = default_value;
end
end

function y = clamp(x, lo, hi)
y = min(max(x, lo), hi);
end

function out = srv_get(server, path)
cmd = sprintf('curl -sS "%s%s"', server, path);
[status, raw] = system(cmd);
assert(status == 0, 'GET failed: %s', raw);
out = jsondecode(strtrim(raw));
end

function out = srv_post(server, path, payload)
body = jsonencode(payload);
body_escaped = strrep(body, '"', '\"');
cmd = sprintf('curl -sS -X POST "%s%s" -H "Content-Type: application/json" -d "%s"', ...
    server, path, body_escaped);
[status, raw] = system(cmd);
assert(status == 0, 'POST failed: %s', raw);
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
