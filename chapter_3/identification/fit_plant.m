%% fit_plant.m  --  Chapter 3: Plant parameter fitting
%
% Standalone. Reads ttft_data.json produced by collect_ttft.py and fits:
%   l_mean = alpha*B + gamma*B^2 + beta*q
%
% Outputs identified_params.mat and two diagnostic plots.

clear; clc;
out_dir  = fileparts(mfilename('fullpath'));
json_path = fullfile(out_dir, 'ttft_data.json');

if ~isfile(json_path)
    error('ttft_data.json not found. Run collect_ttft.py first.');
end

data = jsondecode(fileread(json_path));
fprintf('Loaded: %s\n', json_path);
fprintf('Model:  %s\n\n', data.model);

% =========================================================================
% STAGE 1 — Fit alpha, gamma from B sweep at q=0
% =========================================================================
fprintf('─── Stage 1: fitting alpha, gamma ───\n');
b_fields = fieldnames(data.stage1);
n_b      = numel(b_fields);
b_vals   = zeros(n_b,1);
l_mean   = zeros(n_b,1);
l_std    = zeros(n_b,1);

for i = 1:n_b
    b_vals(i) = str2double(b_fields{i});
    s = data.stage1.(b_fields{i});
    l_mean(i) = s.mean;
    l_std(i)  = s.std;
end
[b_vals, idx] = sort(b_vals);
l_mean = l_mean(idx);
l_std  = l_std(idx);

% Linear least squares: l = [B, B^2] * [alpha; gamma]
A1       = [b_vals, b_vals.^2];
params1  = A1 \ l_mean;
alpha_id = params1(1);
gamma_id = params1(2);

l_fit1  = A1 * params1;
r2_s1   = 1 - sum((l_mean - l_fit1).^2) / sum((l_mean - mean(l_mean)).^2);

fprintf('  alpha = %8.4f  ms/req\n',    alpha_id);
fprintf('  gamma = %8.4f  ms/req^2\n',  gamma_id);
fprintf('  R^2   = %8.4f\n\n',          r2_s1);

% =========================================================================
% STAGE 2 — Fit beta from sustained load sweep
% =========================================================================
fprintf('─── Stage 2: fitting beta ───\n');
lam_fields = fieldnames(data.stage2);
n_lam      = numel(lam_fields);
q_ss_vec   = zeros(n_lam,1);
b_ss_vec   = zeros(n_lam,1);
l_ss_vec   = zeros(n_lam,1);

for i = 1:n_lam
    s = data.stage2.(lam_fields{i});
    q_ss_vec(i) = s.q_ss;
    b_ss_vec(i) = s.b_ss;
    l_ss_vec(i) = s.l_ss;
    fprintf('  lambda=%-3s  q_ss=%5.2f  b_ss=%5.2f  l_ss=%6.1f ms\n', ...
        lam_fields{i}, s.q_ss, s.b_ss, s.l_ss);
end

% Residual: l_ss - (alpha*B + gamma*B^2) = beta * q_ss
service   = alpha_id * b_ss_vec + gamma_id * b_ss_vec.^2;
residual  = l_ss_vec - service;

% Only use points where queue is non-trivial (q_ss > 0.2)
valid     = q_ss_vec > 0.2;
if sum(valid) < 2
    warning('Too few valid q_ss points for beta regression — using all points.');
    valid = true(n_lam, 1);
end

% Linear regression through origin: residual = beta * q_ss
beta_id   = (q_ss_vec(valid)' * residual(valid)) / (q_ss_vec(valid)' * q_ss_vec(valid));

l_fit2    = service + beta_id * q_ss_vec;
r2_s2     = 1 - sum((l_ss_vec(valid) - l_fit2(valid)).^2) / ...
                sum((l_ss_vec(valid) - mean(l_ss_vec(valid))).^2);

fprintf('\n  beta  = %8.4f  ms/req\n', beta_id);
fprintf('  R^2   = %8.4f\n\n',          r2_s2);

% =========================================================================
% SUMMARY
% =========================================================================
dt      = 1.0;
tau_out = 30.0;
z_cl    = exp(-dt / tau_out);
K_il_id       = (z_cl - 1) / beta_id;
K_il_assumed  = (z_cl - 1) / 2.0;

fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║  IDENTIFIED PARAMETERS  (%s)\n', data.model);
fprintf('╠══════════════════════════════════════════════════╣\n');
fprintf('║  alpha = %7.4f  ms/req      R^2(s1) = %.4f\n', alpha_id, r2_s1);
fprintf('║  gamma = %7.4f  ms/req^2    R^2(s1) = %.4f\n', gamma_id, r2_s1);
fprintf('║  beta  = %7.4f  ms/req      R^2(s2) = %.4f\n', beta_id,  r2_s2);
fprintf('╠══════════════════════════════════════════════════╣\n');
fprintf('║  Previously assumed:  alpha=0.1  gamma=0.8  beta=2\n');
fprintf('╠══════════════════════════════════════════════════╣\n');
fprintf('║  K_il (identified beta): %.6f\n', K_il_id);
fprintf('║  K_il (assumed beta=2):  %.6f\n', K_il_assumed);
fprintf('║  Ratio:                  %.2fx\n', K_il_id / K_il_assumed);
fprintf('╚══════════════════════════════════════════════════╝\n\n');

% =========================================================================
% PLOTS
% =========================================================================
fig1 = figure('Name','Stage 1','Position',[100 100 600 420]);
errorbar(b_vals, l_mean, l_std, 'bo', 'MarkerFaceColor','b', 'LineWidth',1.2);
hold on;
b_fine = linspace(1, max(b_vals), 200);
plot(b_fine, alpha_id*b_fine + gamma_id*b_fine.^2, 'r-', 'LineWidth', 2);
xlabel('Batch size B  [req]'); ylabel('Mean TTFT  [ms]');
title(sprintf('Stage 1 — B sweep (q=0)\n\\alpha=%.3f, \\gamma=%.4f, R^2=%.3f', ...
    alpha_id, gamma_id, r2_s1));
legend('Measured (mean ± std)', 'Fit: \alpha B + \gamma B^2', 'Location','northwest');
grid on;
saveas(fig1, fullfile(out_dir, 'id_stage1.png'));

fig2 = figure('Name','Stage 2','Position',[720 100 900 420]);
subplot(1,2,1);
scatter(q_ss_vec, residual, 70, 'b', 'filled');
hold on;
q_fine = linspace(0, max(q_ss_vec)*1.2, 100);
plot(q_fine, beta_id * q_fine, 'r-', 'LineWidth', 2);
xlabel('q_{ss}  [req]'); ylabel('l_{ss} - service term  [ms]');
title(sprintf('Queuing term\n\\beta=%.3f ms/req, R^2=%.3f', beta_id, r2_s2));
legend('Measured', sprintf('\\beta q'), 'Location','northwest'); grid on;

subplot(1,2,2);
lam_nums = cellfun(@str2double, lam_fields);
scatter(b_ss_vec, l_ss_vec, 70, lam_nums, 'filled');
colorbar; xlabel('B_{ss}  [req]'); ylabel('l_{ss}  [ms]');
hold on;
b_fine2 = linspace(1, max(b_ss_vec)*1.1, 100);
for qi = [0 1 2 3 4]
    plot(b_fine2, alpha_id*b_fine2 + gamma_id*b_fine2.^2 + beta_id*qi, '--', ...
        'DisplayName', sprintf('q=%d', qi));
end
title('Identified surface l(B,q)'); legend('show','Location','northwest'); grid on;
saveas(fig2, fullfile(out_dir, 'id_stage2.png'));

% =========================================================================
% SAVE
% =========================================================================
identified.alpha     = alpha_id;
identified.gamma     = gamma_id;
identified.beta      = beta_id;
identified.r2_stage1 = r2_s1;
identified.r2_stage2 = r2_s2;
identified.model     = data.model;
identified.timestamp = char(datetime('now'));
identified.raw.b_vals   = b_vals;
identified.raw.l_mean   = l_mean;
identified.raw.q_ss     = q_ss_vec;
identified.raw.b_ss     = b_ss_vec;
identified.raw.l_ss     = l_ss_vec;

save_path = fullfile(out_dir, 'identified_params.mat');
save(save_path, 'identified');
fprintf('Saved: %s\n', save_path);
fprintf('\nPaste into setup_plant.m:\n');
fprintf('  id = load(''%s'', ''identified'').identified;\n', save_path);
fprintf('  perturbed.alpha = %.4f;\n', alpha_id);
fprintf('  perturbed.gamma = %.4f;\n', gamma_id);
fprintf('  perturbed.beta  = %.4f;\n', beta_id);
