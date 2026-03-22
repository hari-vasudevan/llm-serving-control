%% id_stage2_fit.m  --  Plant identification: fit beta from stage 2 data
%
% Standalone. Run after id_stage2.m has been run for all lambda values.
% Fits beta by linear regression and saves final identified parameters.
% Also updates setup_plant.m with the identified values.

clear; clc;
out_dir = fileparts(mfilename('fullpath'));
src_dir = fullfile(out_dir, '..', 'src');

% ── Load results ──────────────────────────────────────────────────────────
load(fullfile(out_dir,'identified_stage1.mat'),'s1');
load(fullfile(out_dir,'identified_stage2.mat'),'s2');

alpha_id = s1.alpha;
gamma_id = s1.gamma;

fprintf('Stage 2 data collected:\n');
fprintf('  lambda  q_ss   b_ss   l_ss   beta_est\n');
for i=1:numel(s2.lambda)
    fprintf('  %6.1f  %5.2f  %5.2f  %5.1f  %8.4f\n', ...
        s2.lambda(i), s2.q_ss(i), s2.b_ss(i), s2.l_ss(i), s2.beta_est(i));
end

% ── Fit beta ──────────────────────────────────────────────────────────────
% Regression through origin: (l_ss - service) = beta * q_ss
service  = alpha_id * s2.b_ss + gamma_id * s2.b_ss.^2;
residual = s2.l_ss - service;
valid    = s2.q_ss > 0.2;

% Linear regression through origin
beta_id = (s2.q_ss(valid)' * residual(valid)) / (s2.q_ss(valid)' * s2.q_ss(valid));

l_fit = service + beta_id * s2.q_ss;
ss_res = sum((s2.l_ss(valid) - l_fit(valid)).^2);
ss_tot = sum((s2.l_ss(valid) - mean(s2.l_ss(valid))).^2);
r2_beta = 1 - ss_res/ss_tot;

fprintf('\n  beta  = %.4f ms/req  (R^2 = %.4f)\n', beta_id, r2_beta);

% ── Summary ───────────────────────────────────────────────────────────────
fprintf('\n╔══════════════════════════════════════════╗\n');
fprintf('║  IDENTIFIED PARAMETERS  (%s)\n', s2.model);
fprintf('╠══════════════════════════════════════════╣\n');
fprintf('║  alpha = %7.4f ms/req    (R^2=%.3f)\n', alpha_id, s1.r2);
fprintf('║  gamma = %7.4f ms/req^2  (R^2=%.3f)\n', gamma_id, s1.r2);
fprintf('║  beta  = %7.4f ms/req    (R^2=%.3f)\n', beta_id,  r2_beta);
fprintf('╠══════════════════════════════════════════╣\n');
fprintf('║  Assumed: alpha=0.1 gamma=0.8 beta=2.0  ║\n');
dt=1.0; tau=30.0; z=exp(-dt/tau);
fprintf('║  K_il(identified) = %.6f\n', (z-1)/beta_id);
fprintf('║  K_il(assumed)    = %.6f\n', (z-1)/2.0);
fprintf('╚══════════════════════════════════════════╝\n\n');

% ── Plot ──────────────────────────────────────────────────────────────────
fig = figure('Name','Stage 2 fit','Visible','off');
subplot(1,2,1);
scatter(s2.q_ss, residual, 60, s2.lambda,'filled'); colorbar;
hold on;
q_fine = linspace(0, max(s2.q_ss)*1.1, 100);
plot(q_fine, beta_id*q_fine,'r-','LineWidth',2);
xlabel('q_{ss} [req]'); ylabel('l_{ss} - service [ms]');
title(sprintf('beta fit: %.3f ms/req  R^2=%.3f', beta_id, r2_beta));
legend('Measured (colour=lambda)','beta*q'); grid on;

subplot(1,2,2);
b_fine = linspace(1,12,100);
for qi = [0 2 4 6]
    plot(b_fine, alpha_id*b_fine + gamma_id*b_fine.^2 + beta_id*qi, ...
        'DisplayName',sprintf('q=%d',qi)); hold on;
end
scatter(s2.b_ss, s2.l_ss, 60,'k','filled','DisplayName','Measured ss');
xlabel('B [req]'); ylabel('l [ms]');
title('Identified surface: l(B,q)');
legend('show','Location','northwest'); grid on;
saveas(fig, fullfile(out_dir,'id_stage2.png'));
fprintf('Plot saved: id_stage2.png\n');

% ── Save final params ─────────────────────────────────────────────────────
identified.alpha     = alpha_id;
identified.gamma     = gamma_id;
identified.beta      = beta_id;
identified.r2_stage1 = s1.r2;
identified.r2_beta   = r2_beta;
identified.model     = s2.model;
identified.timestamp = char(datetime('now'));
save(fullfile(out_dir,'identified_params.mat'),'identified');
fprintf('Final params saved: identified_params.mat\n\n');

% ── Print setup_plant.m update instructions ───────────────────────────────
fprintf('Update these lines in setup_plant.m:\n');
fprintf('  perturbed.alpha = %.4f;\n', alpha_id);
fprintf('  perturbed.gamma = %.4f;\n', gamma_id);
fprintf('  perturbed.beta  = %.4f;\n', beta_id);
