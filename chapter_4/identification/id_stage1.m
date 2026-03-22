%% id_stage1.m  --  Plant identification Stage 1: B sweep at q=0
%
% Standalone. No dependency on setup_plant.m.
% Identifies alpha and gamma from:  l(B, q=0) = alpha*B + gamma*B^2
%
% Run FIRST. Results saved to identified_stage1.mat.
% Then run id_stage2.m to identify beta.

clear; clc;
src_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'src');
addpath(src_dir);

% ── Config ────────────────────────────────────────────────────────────────
url         = 'http://localhost:11434/api/generate';
model       = 'qwen2.5:3b';
num_predict = 1;
timeout     = 15;
n_reps      = 5;
b_sweep     = [1 2 3 4 5 6 7 8 10 12];
out_dir     = fileparts(mfilename('fullpath'));

fprintf('Stage 1: B sweep at q=0  (alpha, gamma)\n');
fprintf('Model: %s   reps per B: %d\n\n', model, n_reps);

% ── Parallel pool ─────────────────────────────────────────────────────────
if isempty(gcp('nocreate'))
    parpool('local', min(16, feature('numcores')));
end
addAttachedFiles(gcp, src_dir);

% ── Warm-up ───────────────────────────────────────────────────────────────
fprintf('Warming up...\n');
wf = cell(4,1);
for i=1:4
    wf{i} = parfeval(@ollama_ttft,1,url,model,'Hello',1,timeout);
end
for i=1:4; try; fetchOutputs(wf{i}); catch; end; end
fprintf('Warm.\n\n');

% ── B sweep ───────────────────────────────────────────────────────────────
n_b      = numel(b_sweep);
l_mean_b = zeros(n_b,1);
l_std_b  = zeros(n_b,1);

for bi = 1:n_b
    b = b_sweep(bi);
    burst_means = zeros(n_reps,1);
    for rep = 1:n_reps
        futures = cell(b,1);
        for i=1:b
            futures{i} = parfeval(@ollama_ttft,1,url,model,'What is 2+2?',num_predict,timeout);
        end
        lats = zeros(1,b);
        for i=1:b
            try; lats(i) = fetchOutputs(futures{i}); catch; lats(i) = NaN; end
        end
        burst_means(rep) = mean(lats,'omitnan');
    end
    l_mean_b(bi) = mean(burst_means,'omitnan');
    l_std_b(bi)  = std(burst_means,'omitnan');
    fprintf('  B=%2d:  mean=%5.1f ms  std=%4.1f ms\n', b, l_mean_b(bi), l_std_b(bi));
end

% ── Fit alpha, gamma ──────────────────────────────────────────────────────
A        = [b_sweep(:), b_sweep(:).^2];
params   = A \ l_mean_b;
alpha_id = params(1);
gamma_id = params(2);
l_fit    = A * params;
r2       = 1 - sum((l_mean_b-l_fit).^2) / sum((l_mean_b-mean(l_mean_b)).^2);

fprintf('\n  alpha = %.4f ms/req\n', alpha_id);
fprintf('  gamma = %.4f ms/req^2\n', gamma_id);
fprintf('  R^2   = %.4f\n\n', r2);

% ── Plot ──────────────────────────────────────────────────────────────────
fig = figure('Name','Stage 1','Visible','off');
errorbar(b_sweep, l_mean_b, l_std_b, 'bo','MarkerFaceColor','b','LineWidth',1.2);
hold on;
b_fine = linspace(1, max(b_sweep), 100);
plot(b_fine, alpha_id*b_fine + gamma_id*b_fine.^2, 'r-','LineWidth',2);
xlabel('Batch size B  [req]'); ylabel('Mean TTFT  [ms]');
title(sprintf('Stage 1 — B sweep (q=0)\nalpha=%.3f  gamma=%.4f  R^2=%.3f', alpha_id, gamma_id, r2));
legend('Measured','alpha*B + gamma*B^2','Location','northwest'); grid on;
saveas(fig, fullfile(out_dir,'id_stage1.png'));
fprintf('Plot saved: id_stage1.png\n');

% ── Save ──────────────────────────────────────────────────────────────────
s1.alpha   = alpha_id;
s1.gamma   = gamma_id;
s1.r2      = r2;
s1.b_sweep = b_sweep;
s1.l_mean  = l_mean_b;
s1.l_std   = l_std_b;
s1.model   = model;
save(fullfile(out_dir,'identified_stage1.mat'),'s1');
fprintf('Saved: identified_stage1.mat\n');
fprintf('\nRun id_stage2.m next.\n');
