%% design_controller.m -- Chapter 9 cascade controller design
%
% Same terminology as Chapter 2:
%   controller.inner_c: B -> q
%   controller.outer_c: q_ref -> L_mean

clear; clc;
addpath(fileparts(mfilename('fullpath')));
load(fullfile(fileparts(mfilename('fullpath')), 'identified_params.mat'));

fprintf('=== Chapter 9 Cascade Design ===\n');

% Demonstration operating point.  The plant fit still comes from the full
% characterization, but the closed-loop demo should not sit next to B_max.
DEMO_B0 = 1200;
DEMO_LAMBDA_NOMINAL = 1200;
DEMO_Q0 = 1800;
DEMO_L_MEAN_TARGET = 300;
if exist('LAMBDA_SWEEP', 'var') && exist('qmean_lambda', 'var') && exist('lmean_lambda', 'var')
    [~, demo_idx] = min(abs(LAMBDA_SWEEP - DEMO_LAMBDA_NOMINAL));
    B0 = DEMO_B0;
    lambda_mean = LAMBDA_SWEEP(demo_idx);
    q0 = DEMO_Q0;
    L_mean_target = DEMO_L_MEAN_TARGET;
    L_p95_target = DEMO_L_MEAN_TARGET;
    fprintf('[demo op] overriding saved op to B0=%d lambda0=%.2f q0=%.2f L_target=%.2f using characterization point index %d\n', ...
        B0, lambda_mean, q0, L_mean_target, demo_idx);
end

perturbed = struct();
perturbed.dt = DT;
perturbed.B0 = B0;
perturbed.q0 = q0;
perturbed.beta_q = beta_q;
perturbed.beta_q_sign = -1;  % q decreases when B increases
perturbed.beta = beta;
perturbed.alpha = alpha;
perturbed.gamma = gamma;
perturbed.lambda_mean = lambda_mean;
perturbed.L_mean_target = L_mean_target;
perturbed.L_p95_target = L_p95_target;
perturbed.B_min = B_min;
perturbed.B_max = B_max;
perturbed.q_min = max(Q_REF_MIN, lambda_mean);
perturbed.q_max = Q_MAX;
perturbed.tau_in = 1.0;
perturbed.tau_out = 300.0;
perturbed.inner_integral_fraction = 0.60;
perturbed.inner_xi_leak = 0.85;

fprintf('[op] B0=%d B_max=%d lambda_mean=%.2f q0=%.2f L_target=%.2f\n', ...
    perturbed.B0, perturbed.B_max, perturbed.lambda_mean, perturbed.q0, perturbed.L_mean_target);

controller = design_cascade(perturbed);
save(fullfile(fileparts(mfilename('fullpath')), 'controller_params.mat'), ...
    'controller', 'perturbed', 'SERVER');
fprintf('[save] controller_params.mat\n');

xml_path = fullfile(fileparts(mfilename('fullpath')), 'controller_config.xml');
write_controller_xml(xml_path, controller, perturbed);
fprintf('[save] controller_config.xml\n');

if ~contains(SERVER, 'REPLACE_WITH_MODAL')
    upload_controller_config(SERVER, xml_path);
else
    fprintf('[skip] SERVER placeholder still set; not uploading controller_config.xml\n');
end


function controller = design_cascade(perturbed)
B0 = perturbed.B0;
q0 = perturbed.q0;
dt = perturbed.dt;
beta_q = max(perturbed.beta_q, 1e-3);
beta = max(perturbed.beta, 1e-3);

% Inner loop, Chapter 2 form:
%   dq[k+1] = dq[k] - beta_q*dB[k]
%   dB[k] = -K_q*dq[k] - K_i_q*xi_q[k]
rho_in = exp(-dt / perturbed.tau_in);
K_q = (1 - rho_in) / beta_q;
K_i_q = perturbed.inner_integral_fraction * K_q;

% Outer loop, Chapter 2 static-gain form:
%   Delta L = beta * Delta q_ref
%   e_l = L_target - L
%   q_ref = q0 + K_i_l * xi_l, with K_i_l > 0 for this error definition
rho_out = exp(-dt / perturbed.tau_out);
K_i_l = (1 - rho_out) / beta;

xi_q_max = (perturbed.B_max - B0) / max(abs(K_i_q), 1e-9);
xi_q_min = -(B0 - perturbed.B_min) / max(abs(K_i_q), 1e-9);
xi_l_max = (perturbed.q_max - q0) / max(abs(K_i_l), 1e-9);
xi_l_min = -(q0 - perturbed.q_min) / max(abs(K_i_l), 1e-9);

fprintf('Inner B->q: beta_q=%.4f K_q=%.6f K_i_q=%.6f rho=%.4f\n', beta_q, K_q, K_i_q, rho_in);
fprintf('  sign convention: dq[k+1] = dq[k] - beta_q*dB[k]\n');
fprintf('Outer q_ref->L_mean: beta=%.4f K_i_l=%.6f rho=%.4f\n', beta, K_i_l, rho_out);

controller = struct();
controller.inner_c = struct( ...
    'K_q', K_q, ...
    'K_i_q', K_i_q, ...
    'xi_leak', perturbed.inner_xi_leak, ...
    'B0', B0, ...
    'B_min', perturbed.B_min, ...
    'B_max', perturbed.B_max, ...
    'xi_min', xi_q_min, ...
    'xi_max', xi_q_max, ...
    'rho', rho_in);
controller.outer_c = struct( ...
    'K_i_l', K_i_l, ...
    'q0', q0, ...
    'q_min', perturbed.q_min, ...
    'q_max', perturbed.q_max, ...
    'L_mean_target', perturbed.L_mean_target, ...
    'L_p95_target', perturbed.L_p95_target, ...
    'xi_min', xi_l_min, ...
    'xi_max', xi_l_max, ...
    'rho', rho_out);
end

function write_controller_xml(path, controller, perturbed)
fid = fopen(path, 'w');
assert(fid > 0, 'Could not open %s for writing', path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '<chapter9_controller>\n');
fprintf(fid, '  <inner_c>\n');
write_struct_fields(fid, controller.inner_c, 4);
fprintf(fid, '  </inner_c>\n');
fprintf(fid, '  <outer_c>\n');
write_struct_fields(fid, controller.outer_c, 4);
fprintf(fid, '  </outer_c>\n');
fprintf(fid, '  <perturbed>\n');
write_struct_fields(fid, perturbed, 4);
fprintf(fid, '  </perturbed>\n');
fprintf(fid, '</chapter9_controller>\n');
end

function write_struct_fields(fid, s, indent)
fields = fieldnames(s);
pad = repmat(' ', 1, indent);
for i = 1:numel(fields)
    name = fields{i};
    value = s.(name);
    if isnumeric(value) || islogical(value)
        value_txt = num2str(double(value), '%.15g');
    else
        value_txt = char(string(value));
    end
    fprintf(fid, '%s<%s>%s</%s>\n', pad, name, value_txt, name);
end
end

function upload_controller_config(server, xml_path)
xml_text = fileread(xml_path);
payload = struct('xml', xml_text, 'source', 'matlab_design_controller');
out = srv_post(server, '/controller_config', payload);
fprintf('[upload] controller_config status=%s\n', string(out.status));
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
