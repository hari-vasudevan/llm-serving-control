%% characterise_plant.m -- Chapter 9 plant identification
%
% Lower-level GPU scheduling/batching experiment using Chapter 2 terminology:
%   inner loop: B -> q
%   outer loop: q_ref -> L_mean / L_p95

clear; clc;
addpath(fileparts(mfilename('fullpath')));

SERVER = getenv('CH9_SERVER');
if strlength(SERVER) == 0
    SERVER = 'https://REPLACE_WITH_MODAL_CH9_URL';
end
DT = 0.5;                         % controller/sample tick [s]
B_SWEEP = [2 4 6 8 10 12 16];     % batch-size actuator sweep
LAMBDA_SWEEP = [4 6 8 10 12 14];  % arrivals/tick sweep
TICKS_PER_POINT = 16;
SETTLE_TICKS = 5;
Q_TARGET_NOMINAL = 16;
Q_REF_MIN = 0;
Q_MAX = 200;

TRACE_FILE = fullfile(fileparts(mfilename('fullpath')), 'characterise_trace.txt');
if exist(TRACE_FILE, 'file'); delete(TRACE_FILE); end
diary(TRACE_FILE); diary on;

fprintf('=== Chapter 9 Characterise Plant ===\n');
fprintf('Plant vocabulary: inner B->q, outer q_ref->L_mean/L_p95\n');
fprintf('SERVER=%s DT=%.3fs\n', SERVER, DT);

disp(srv_get(SERVER, '/health'));

lambda_char = 10;
qmean_B = NaN(size(B_SWEEP));
lmean_B = NaN(size(B_SWEEP));
lp95_B = NaN(size(B_SWEEP));
service_B = NaN(size(B_SWEEP));
completion_B = NaN(size(B_SWEEP));

for i = 1:numel(B_SWEEP)
    B = B_SWEEP(i);
    fprintf('\n=== B sweep: B=%d lambda=%.2f arrivals/tick ===\n', B, lambda_char);
    logs = run_block(SERVER, B, lambda_char, DT, TICKS_PER_POINT);
    s = summarise_block(logs, SETTLE_TICKS);
    qmean_B(i) = s.q_mean_tick;
    lmean_B(i) = s.l_mean_ms;
    lp95_B(i) = s.l_p95_ms;
    service_B(i) = s.service_mean_ms;
    completion_B(i) = s.completions_tick;
    fprintf('[B=%d] q=%.2f L_mean=%.2f L_p95=%.2f service=%.2fms comps=%.2f\n', ...
        B, qmean_B(i), lmean_B(i), lp95_B(i), service_B(i), completion_B(i));
end

B0_probe = 10;
qmean_lambda = NaN(size(LAMBDA_SWEEP));
lmean_lambda = NaN(size(LAMBDA_SWEEP));
lp95_lambda = NaN(size(LAMBDA_SWEEP));
service_lambda = NaN(size(LAMBDA_SWEEP));
completion_lambda = NaN(size(LAMBDA_SWEEP));

for i = 1:numel(LAMBDA_SWEEP)
    lam = LAMBDA_SWEEP(i);
    fprintf('\n=== lambda sweep: B=%d lambda=%.2f arrivals/tick ===\n', B0_probe, lam);
    logs = run_block(SERVER, B0_probe, lam, DT, TICKS_PER_POINT);
    s = summarise_block(logs, SETTLE_TICKS);
    qmean_lambda(i) = s.q_mean_tick;
    lmean_lambda(i) = s.l_mean_ms;
    lp95_lambda(i) = s.l_p95_ms;
    service_lambda(i) = s.service_mean_ms;
    completion_lambda(i) = s.completions_tick;
    fprintf('[lambda=%.2f] q=%.2f L_mean=%.2f L_p95=%.2f service=%.2fms comps=%.2f\n', ...
        lam, qmean_lambda(i), lmean_lambda(i), lp95_lambda(i), service_lambda(i), completion_lambda(i));
end

% Chapter 2 inner linearisation: dq[k+1] = dq[k] - dB[k].
% In this real plant, estimate an empirical queue sensitivity around B0.
q_fit = polyfit(B_SWEEP(:), qmean_B(:), 1);
beta_q = max(-q_fit(1), 0.25);

% Chapter 2 outer static plant: Delta L = beta * Delta q_ref.
valid_outer = isfinite(qmean_lambda) & isfinite(lmean_lambda);
l_fit = polyfit(qmean_lambda(valid_outer), lmean_lambda(valid_outer), 1);
beta = max(l_fit(1), 1e-3);

% Fit service-time part of Chapter 2 latency law: alpha*B + gamma*B^2.
service_fit = polyfit(B_SWEEP(:), service_B(:), 2);
gamma = max(service_fit(1), 0);
alpha = service_fit(2);

[~, op_idx] = min(abs(qmean_lambda - Q_TARGET_NOMINAL));
B0 = B0_probe;
lambda_mean = LAMBDA_SWEEP(op_idx);
q0 = qmean_lambda(op_idx);
L_mean_target = lmean_lambda(op_idx);
L_p95_target = lp95_lambda(op_idx);
B_min = min(B_SWEEP);
B_max = max(B_SWEEP);

fprintf('\n[fit] q_mean(B)=%.4f*B + %.4f -> beta_q=%.4f\n', q_fit(1), q_fit(2), beta_q);
fprintf('[fit] L_mean(q)=%.4f*q + %.4f -> beta=%.4f ms/request\n', l_fit(1), l_fit(2), beta);
fprintf('[fit] service(B)=%.4f*B^2 + %.4f*B + %.4f ms\n', service_fit(1), service_fit(2), service_fit(3));
fprintf('[op] B0=%d lambda_mean=%.2f q0=%.2f L_mean_target=%.2f L_p95_target=%.2f\n', ...
    B0, lambda_mean, q0, L_mean_target, L_p95_target);

save(fullfile(fileparts(mfilename('fullpath')), 'identified_params.mat'), ...
    'SERVER', 'DT', 'B_SWEEP', 'LAMBDA_SWEEP', 'TICKS_PER_POINT', 'SETTLE_TICKS', ...
    'qmean_B', 'lmean_B', 'lp95_B', 'service_B', 'completion_B', ...
    'qmean_lambda', 'lmean_lambda', 'lp95_lambda', 'service_lambda', 'completion_lambda', ...
    'beta_q', 'beta', 'alpha', 'gamma', 'B0', 'lambda_mean', 'q0', ...
    'L_mean_target', 'L_p95_target', 'B_min', 'B_max', 'Q_REF_MIN', 'Q_MAX');
fprintf('[save] identified_params.mat\n');

fig = figure('Visible', 'off', 'Position', [60 60 1200 700]);
subplot(2,2,1);
plot(B_SWEEP, qmean_B, 'ko-', 'LineWidth', 1.5); grid on;
xlabel('B'); ylabel('q_{mean}'); title('Inner plant sweep: B -> q');

subplot(2,2,2);
plot(B_SWEEP, service_B, 'bo-', 'LineWidth', 1.5); grid on;
xlabel('B'); ylabel('service time [ms]'); title('Measured GPU batch service');

subplot(2,2,3);
plot(qmean_lambda, lmean_lambda, 'rs-', 'LineWidth', 1.5); grid on;
xlabel('q_{mean}'); ylabel('L_{mean} [ms]'); title('Outer plant: q -> L_{mean}');

subplot(2,2,4);
plot(LAMBDA_SWEEP, qmean_lambda, 'm^-', 'LineWidth', 1.5); hold on;
plot(LAMBDA_SWEEP, completion_lambda, 'cv-', 'LineWidth', 1.2); grid on;
xlabel('\lambda [arrivals/tick]'); ylabel('requests/tick'); title('Queue/load calibration');
legend('q_{mean}', 'completions/tick', 'Location', 'northwest');

saveas(fig, fullfile(fileparts(mfilename('fullpath')), 'ch9_characterise.png'));
fprintf('[save] ch9_characterise.png\n');
diary off;


function logs = run_block(server, B_cmd, lambda_tick, dt, n_ticks)
srv_post(server, '/reset', struct());
pause(0.5);
srv_post(server, '/control', struct('B', B_cmd));
pause(0.2);
logs = cell(n_ticks, 1);
for k = 1:n_ticks
    arrivals = poissrnd(lambda_tick);
    srv_post(server, '/enqueue_batch', struct('count', arrivals, 'source', sprintf('char_tick_%03d', k)));
    pause(dt);
    logs{k} = srv_get(server, '/metrics');
    fprintf('[tick=%03d] B=%d arrivals=%d q=%.2f comps=%.2f L=%.2f service=%.2f\n', ...
        k, B_cmd, arrivals, metric_or_nan(logs{k}, 'q_mean_tick'), ...
        metric_or_nan(logs{k}, 'completions_tick'), metric_or_nan(logs{k}, 'l_mean_ms'), ...
        metric_or_nan(logs{k}, 'service_mean_ms'));
end
end

function summary = summarise_block(logs, settle_ticks)
use = logs(min(numel(logs), settle_ticks + 1):end);
summary = struct();
summary.q_mean_tick = average_metric(use, 'q_mean_tick');
summary.l_mean_ms = average_metric(use, 'l_mean_ms');
summary.l_p95_ms = average_metric(use, 'l_p95_ms');
summary.service_mean_ms = average_metric(use, 'service_mean_ms');
summary.completions_tick = average_metric(use, 'completions_tick');
end

function value = average_metric(logs, field_name)
vals = NaN(numel(logs), 1);
for i = 1:numel(logs)
    vals(i) = metric_or_nan(logs{i}, field_name);
end
value = mean(vals, 'omitnan');
end

function value = metric_or_nan(s, field_name)
if isfield(s, field_name) && ~isempty(s.(field_name))
    value = double(s.(field_name));
else
    value = NaN;
end
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
