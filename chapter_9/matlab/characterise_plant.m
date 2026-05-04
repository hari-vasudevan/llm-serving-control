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
DT = 0.5;                                  % Chapter 2 scheduling tick [s]
B_SWEEP = [400 800 1200 1600 2000 2200 2400 2800 3200];
LAMBDA_SWEEP = [1200 1600 1800 2000 2200 2400 2600 2800 3200];
TICKS_PER_POINT = 24;
SETTLE_TICKS = 6;
LAMBDA_CHAR = 2400;
B0_PROBE = 2200;
B0_NOMINAL = 2200;
LAMBDA_NOMINAL = 2200;
Q_REF_MIN = 0;
Q_MAX = 40000;

TRACE_FILE = fullfile(fileparts(mfilename('fullpath')), 'characterise_trace.txt');
if exist(TRACE_FILE, 'file'); delete(TRACE_FILE); end
diary(TRACE_FILE); diary on;

fprintf('=== Chapter 9 Characterise Plant ===\n');
fprintf('Plant vocabulary: inner B->q, outer q_ref->L_mean/L_p95\n');
fprintf('SERVER=%s DT=%.3fs\n', SERVER, DT);

disp(srv_get(SERVER, '/health'));

fprintf('[modal] running per-point in-container characterisation; this removes MATLAB/network timing from load generation\n');

B_results = [];
for i = 1:numel(B_SWEEP)
    payload = block_payload(DT, B_SWEEP(i), LAMBDA_CHAR, TICKS_PER_POINT, SETTLE_TICKS, ...
        sprintf('matlab_characterise_B_%03d', B_SWEEP(i)));
    fprintf('[modal] B sweep %d/%d: B=%d lambda=%.2f\n', i, numel(B_SWEEP), B_SWEEP(i), LAMBDA_CHAR);
    result = srv_post(SERVER, '/characterise_block', payload);
    if i == 1
        B_results = repmat(result, numel(B_SWEEP), 1);
    end
    B_results(i) = result;
end

lambda_results = [];
for i = 1:numel(LAMBDA_SWEEP)
    payload = block_payload(DT, B0_PROBE, LAMBDA_SWEEP(i), TICKS_PER_POINT, SETTLE_TICKS, ...
        sprintf('matlab_characterise_lambda_%03d', LAMBDA_SWEEP(i)));
    fprintf('[modal] lambda sweep %d/%d: B=%d lambda=%.2f\n', i, numel(LAMBDA_SWEEP), B0_PROBE, LAMBDA_SWEEP(i));
    result = srv_post(SERVER, '/characterise_block', payload);
    if i == 1
        lambda_results = repmat(result, numel(LAMBDA_SWEEP), 1);
    end
    lambda_results(i) = result;
end

char_result = struct( ...
    'status', 'ok', ...
    'mode', 'per_point_characterise_block', ...
    'dt', DT, ...
    'lambda_char', LAMBDA_CHAR, ...
    'B0_probe', B0_PROBE, ...
    'ticks_per_point', TICKS_PER_POINT, ...
    'settle_ticks', SETTLE_TICKS, ...
    'B_results', B_results, ...
    'lambda_results', lambda_results);

qmean_B = extract_field(B_results, 'q_mean_tick');
lmean_B = extract_field(B_results, 'l_mean_ms');
lp95_B = extract_field(B_results, 'l_p95_ms');
service_B = extract_field(B_results, 'service_mean_ms');
completion_B = extract_field(B_results, 'completions_tick');

qmean_lambda = extract_field(lambda_results, 'q_mean_tick');
lmean_lambda = extract_field(lambda_results, 'l_mean_ms');
lp95_lambda = extract_field(lambda_results, 'l_p95_ms');
service_lambda = extract_field(lambda_results, 'service_mean_ms');
completion_lambda = extract_field(lambda_results, 'completions_tick');

for i = 1:numel(B_SWEEP)
    fprintf('[B=%d] q=%.2f L_mean=%.2f L_p95=%.2f service=%.2fms comps=%.2f\n', ...
        B_SWEEP(i), qmean_B(i), lmean_B(i), lp95_B(i), service_B(i), completion_B(i));
end
for i = 1:numel(LAMBDA_SWEEP)
    fprintf('[lambda=%.2f] q=%.2f L_mean=%.2f L_p95=%.2f service=%.2fms comps=%.2f\n', ...
        LAMBDA_SWEEP(i), qmean_lambda(i), lmean_lambda(i), lp95_lambda(i), service_lambda(i), completion_lambda(i));
end

% Effective useful B saturates once completions stop improving materially.
completion_max = max(completion_B, [], 'omitnan');
sat_idx = find(completion_B >= 0.95 * completion_max & B_SWEEP >= LAMBDA_CHAR, 1, 'first');
if isempty(sat_idx)
    sat_idx = numel(B_SWEEP);
end
B_max_effective = B_SWEEP(sat_idx);

% Chapter 2 inner linearisation: dq[k+1] = dq[k] - beta_q*dB[k].
% Fit only the useful region up to the throughput knee; beyond that, larger B
% is not an effective actuator.
inner_fit_idx = B_SWEEP <= B_max_effective & isfinite(qmean_B);
q_fit = polyfit(B_SWEEP(inner_fit_idx).', qmean_B(inner_fit_idx).', 1);
beta_q = max(-q_fit(1), 0.25);  % positive magnitude, plant input gain is -beta_q

% Chapter 2 outer static plant: Delta L = beta * Delta q_ref.
valid_outer = isfinite(qmean_lambda) & isfinite(lmean_lambda);
l_fit = polyfit(qmean_lambda(valid_outer), lmean_lambda(valid_outer), 1);
beta = max(l_fit(1), 1e-3);

% Fit service-time part of Chapter 2 latency law: alpha*B + gamma*B^2.
service_fit = polyfit(B_SWEEP(:), service_B(:), 2);
gamma = max(service_fit(1), 0);
alpha = service_fit(2);

[~, op_idx] = min(abs(LAMBDA_SWEEP - LAMBDA_NOMINAL));
B0 = min(B0_NOMINAL, B_max_effective);
lambda_mean = LAMBDA_SWEEP(op_idx);
q0 = qmean_lambda(op_idx);
L_mean_target = lmean_lambda(op_idx);
L_p95_target = lp95_lambda(op_idx);
B_min = min(B_SWEEP);
B_max = B_max_effective;

fprintf('\n[fit] q_mean(B)=%.4f*B + %.4f -> beta_q=%.4f\n', q_fit(1), q_fit(2), beta_q);
fprintf('[fit] B_max_effective=%d from completion plateau %.2f req/tick\n', B_max_effective, completion_max);
fprintf('[fit] L_mean(q)=%.4f*q + %.4f -> beta=%.4f ms/request\n', l_fit(1), l_fit(2), beta);
fprintf('[fit] service(B)=%.4f*B^2 + %.4f*B + %.4f ms\n', service_fit(1), service_fit(2), service_fit(3));
fprintf('[op] B0=%d lambda_mean=%.2f q0=%.2f L_mean_target=%.2f L_p95_target=%.2f\n', ...
    B0, lambda_mean, q0, L_mean_target, L_p95_target);

save(fullfile(fileparts(mfilename('fullpath')), 'identified_params.mat'), ...
    'SERVER', 'DT', 'B_SWEEP', 'LAMBDA_SWEEP', 'TICKS_PER_POINT', 'SETTLE_TICKS', ...
    'LAMBDA_CHAR', 'B0_PROBE', 'B0_NOMINAL', 'LAMBDA_NOMINAL', 'B_max_effective', 'char_result', ...
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


function out = srv_get(server, path)
cmd = sprintf('curl -sS "%s%s"', server, path);
[status, raw] = system(cmd);
assert(status == 0, 'GET failed: %s', raw);
assert(strlength(strtrim(raw)) > 0, 'GET returned empty response for %s%s', server, path);
out = jsondecode(strtrim(raw));
end

function vals = extract_field(s, field_name)
vals = NaN(numel(s), 1);
for i = 1:numel(s)
    if isfield(s(i), field_name) && ~isempty(s(i).(field_name))
        vals(i) = double(s(i).(field_name));
    end
end
vals = vals(:).';
end

function out = srv_post(server, path, payload)
body = jsonencode(payload);
body_escaped = strrep(body, '"', '\"');
cmd = sprintf('curl -sSL -X POST "%s%s" -H "Content-Type: application/json" -d "%s"', ...
    server, path, body_escaped);
[status, raw] = system(cmd);
assert(status == 0, 'POST failed: %s', raw);
assert(strlength(strtrim(raw)) > 0, 'POST returned empty response for %s%s', server, path);
out = jsondecode(strtrim(raw));
end

function payload = block_payload(dt, B, lambda_tick, ticks_per_point, settle_ticks, source)
payload = struct( ...
    'dt', dt, ...
    'B', B, ...
    'lambda', lambda_tick, ...
    'ticks_per_point', ticks_per_point, ...
    'settle_ticks', settle_ticks, ...
    'source', source);
end
