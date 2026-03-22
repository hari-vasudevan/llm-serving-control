%% identify_plant.m  --  Chapter 4: Direct TTFT(B) identification
%
% Standalone — no dependency on setup_plant.m or any workspace variable.
%
% WHY THIS EXISTS (lessons from Chapter 3):
%   The Chapter 3 identification assumed l = alpha*B + gamma*B^2 + beta*q.
%   In practice, the drain rule (b = q + a) keeps q≈0 at all lambda values
%   because Ollama's sequential processing means arrivals are always matched
%   immediately.  The queue term beta*q is unidentifiable.
%
%   The real plant is a direct static map:
%     l_meas[k] = f(B[k])
%   where f is a nonlinear function of concurrent batch size B, driven by
%   GPU KV-cache contention.  B requests fired in PARALLEL create real
%   contention; B sequential requests do not.
%
% WHAT THIS SCRIPT DOES:
%   Fires B requests concurrently (via parfeval) across a sweep of B values.
%   Measures mean TTFT at each B.  Fits a polynomial:
%     l(B) = c0 + c1*B + c2*B^2
%   Derives the effective linearisation gain at a chosen operating point B0:
%     beta_eff = dl/dB|_{B0} = c1 + 2*c2*B0
%   This beta_eff is the only parameter needed for the integral controller.
%
% OUTPUTS:
%   identified_params.mat     -- struct with c0, c1, c2, beta_eff, B0
%   id_ttft_curve.png         -- TTFT vs B measured + fitted curve

clear; clc;

% -------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
id_dir  = fileparts(mfilename('fullpath'));
src_dir = fullfile(id_dir, '..', 'src');
addpath(src_dir);

fprintf('[init] id_dir  = %s\n', id_dir);
fprintf('[init] src_dir = %s\n', src_dir);

if ~exist('ollama_ttft', 'file')
    error('ollama_ttft.m not found on path. Check src_dir = %s', src_dir);
end
fprintf('[init] ollama_ttft found on path\n\n');

% -------------------------------------------------------------------------
% Configuration
% -------------------------------------------------------------------------
cfg.url         = 'http://localhost:11434/api/generate';
cfg.model       = 'qwen2.5:3b';
cfg.num_predict = 1;       % TTFT only — minimise generation component
cfg.timeout     = 20;      % s per request

% B sweep — must use parfeval to fire requests CONCURRENTLY.
% Sequential requests produce no GPU contention and give flat TTFT.
cfg.b_sweep     = [1 2 3 4 5 6 7 8 10 12];
cfg.n_reps      = 5;       % repetitions per B (averaged to reduce noise)

% Operating point for linearisation — must be inside the sweep range.
% Choose where the slope is steepest (maximum control authority).
% Will be updated after the fit is computed.
cfg.B0_target   = 5;       % initial guess; refined after seeing the curve

fprintf('[config] model      = %s\n',   cfg.model);
fprintf('[config] b_sweep    = %s\n',   num2str(cfg.b_sweep));
fprintf('[config] n_reps     = %d\n',   cfg.n_reps);
fprintf('[config] B0_target  = %d\n\n', cfg.B0_target);

% -------------------------------------------------------------------------
% Check Ollama is reachable
% -------------------------------------------------------------------------
fprintf('[check] Pinging Ollama ...\n');
try
    webread('http://localhost:11434/api/tags', weboptions('Timeout', 5));
    fprintf('[check] Ollama is up.\n\n');
catch
    error('Ollama not reachable at localhost:11434. Is it running?');
end

% -------------------------------------------------------------------------
% Open parallel pool
% -------------------------------------------------------------------------
pool = gcp('nocreate');
if isempty(pool)
    fprintf('[pool] Starting parallel pool...\n');
    parpool('local', min(max(cfg.b_sweep), feature('numcores')));
end
fprintf('[pool] Attaching src dir to workers...\n');
addAttachedFiles(gcp, src_dir);
fprintf('[pool] Pool ready: %d workers.\n\n', gcp.NumWorkers);

% -------------------------------------------------------------------------
% Warm-up: load model weights into GPU before data collection
% -------------------------------------------------------------------------
fprintf('[warmup] Firing %d concurrent warm-up requests...\n', 4);
wf = cell(4, 1);
for i = 1:4
    wf{i} = parfeval(@ollama_ttft, 1, cfg.url, cfg.model, 'Hello', 1, cfg.timeout);
end
for i = 1:4
    try
        lat = fetchOutputs(wf{i});
        fprintf('  warmup %d: %.0f ms\n', i, lat);
    catch e
        fprintf('  warmup %d: error (%s)\n', i, e.message);
    end
end
fprintf('[warmup] Done.\n\n');

% =========================================================================
% CONCURRENT TTFT SWEEP
% =========================================================================
%
% Each measurement fires B requests simultaneously via parfeval.
% All B requests are submitted before any are collected, so they compete
% for GPU memory bandwidth exactly as real concurrent load would.
%
% We repeat n_reps times per B value and average to reduce noise.
% Each rep is a fresh set of B concurrent requests.

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('TTFT SWEEP: firing B requests concurrently at each B value\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

n_b      = numel(cfg.b_sweep);
l_mean_b = zeros(n_b, 1);    % mean TTFT per B
l_std_b  = zeros(n_b, 1);    % std of rep means per B
l_raw    = cell(n_b, 1);     % all individual TTFTs for each B (for inspection)

for bi = 1:n_b
    b = cfg.b_sweep(bi);
    fprintf('[sweep] B = %2d  (%d reps x %d concurrent requests)...\n', ...
        b, cfg.n_reps, b);

    rep_means = zeros(cfg.n_reps, 1);

    for r = 1:cfg.n_reps
        % Submit all B requests simultaneously
        futures = cell(b, 1);
        for i = 1:b
            futures{i} = parfeval(@ollama_ttft, 1, ...
                cfg.url, cfg.model, 'What is 2+2?', cfg.num_predict, cfg.timeout);
        end

        % Collect all results
        lats = zeros(1, b);
        for i = 1:b
            try
                lats(i) = fetchOutputs(futures{i});
            catch
                lats(i) = NaN;   % timeout — will be excluded from mean
            end
        end

        rep_means(r) = mean(lats, 'omitnan');
        fprintf('  rep %d: mean=%.1f ms  [%s]\n', r, rep_means(r), ...
            num2str(round(lats), '%d '));
    end

    l_mean_b(bi) = mean(rep_means, 'omitnan');
    l_std_b(bi)  = std(rep_means,  'omitnan');
    l_raw{bi}    = rep_means;
    fprintf('  --> B=%2d final: %.1f ± %.1f ms\n\n', b, l_mean_b(bi), l_std_b(bi));
end

% =========================================================================
% POLYNOMIAL FIT:  l(B) = c0 + c1*B + c2*B^2
% =========================================================================
fprintf('[fit] Fitting l(B) = c0 + c1*B + c2*B^2 ...\n');

% Use only clean data points (finite, positive TTFT)
valid    = isfinite(l_mean_b) & l_mean_b > 0;
B_valid  = cfg.b_sweep(valid)';
l_valid  = l_mean_b(valid);

% Design matrix for quadratic fit through data (no intercept constraint)
A_fit    = [ones(size(B_valid)), B_valid, B_valid.^2];
coeffs   = A_fit \ l_valid;       % least squares
c0       = coeffs(1);
c1       = coeffs(2);
c2       = coeffs(3);

% Goodness of fit
l_fitted = A_fit * coeffs;
ss_res   = sum((l_valid - l_fitted).^2);
ss_tot   = sum((l_valid - mean(l_valid)).^2);
r2       = 1 - ss_res / ss_tot;

fprintf('[fit] c0 = %.4f  ms         (offset)\n',       c0);
fprintf('[fit] c1 = %.4f  ms/req     (linear coeff)\n',  c1);
fprintf('[fit] c2 = %.4f  ms/req^2   (quadratic coeff)\n', c2);
fprintf('[fit] R^2 = %.4f\n\n', r2);

% =========================================================================
% LINEARISATION AT OPERATING POINT
% =========================================================================
%
% The effective plant gain at a chosen B0 is:
%   beta_eff = dl/dB|_{B0} = c1 + 2*c2*B0
%
% For the integral controller:
%   K_il = (exp(-dt/tau_out) - 1) / beta_eff
%
% We choose B0 where the curve has its steepest slope (maximum sensitivity)
% to maximise controller authority.  This is where the second derivative
% is zero if c2 > 0 (minimum of dl/dB), or we pick the midpoint of the
% sweep range.  If c2 < 0, slope is monotonically decreasing — pick the
% lower end of the operating range.

fprintf('[linearise] Computing beta_eff at each B in sweep...\n');
for bi = 1:n_b
    b     = cfg.b_sweep(bi);
    slope = c1 + 2*c2*b;
    fprintf('  B=%2d:  dl/dB = %.4f ms/req\n', b, slope);
end

% Pick B0 as the user-configured target (should be equilibrium)
B0       = cfg.B0_target;
beta_eff = c1 + 2*c2*B0;
l0       = c0 + c1*B0 + c2*B0^2;   % predicted l at equilibrium

fprintf('\n[linearise] At B0 = %d:\n', B0);
fprintf('  l(B0)     = %.2f ms   (predicted TTFT at equilibrium)\n', l0);
fprintf('  beta_eff  = %.4f ms/req  (dl/dB at B0)\n\n', beta_eff);

if beta_eff <= 0
    fprintf('[WARNING] beta_eff <= 0 at B0=%d.\n', B0);
    fprintf('  The curve is decreasing at this point — increasing B reduces TTFT.\n');
    fprintf('  This is valid physics (more parallel slots = faster when not saturated).\n');
    fprintf('  The controller sign will be inverted: K_il > 0 to reduce B when l > target.\n\n');
end

% =========================================================================
% PLOT
% =========================================================================
fig1 = figure('Name', 'TTFT curve', 'Visible', 'off');
errorbar(cfg.b_sweep, l_mean_b, l_std_b, 'bo', ...
    'MarkerFaceColor', 'b', 'LineWidth', 1.2, 'DisplayName', 'Measured');
hold on;
B_fine   = linspace(1, max(cfg.b_sweep), 300);
l_fine   = c0 + c1*B_fine + c2*B_fine.^2;
plot(B_fine, l_fine, 'r-', 'LineWidth', 2, ...
    'DisplayName', sprintf('c_0 + c_1 B + c_2 B^2  (R^2=%.3f)', r2));

% Mark operating point and tangent line
x_tan    = B0 + [-2 2];
y_tan    = l0 + beta_eff * (x_tan - B0);
plot(x_tan, y_tan, 'g--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Tangent at B_0=%d  (\\beta_{eff}=%.3f)', B0, beta_eff));
plot(B0, l0, 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g', ...
    'HandleVisibility', 'off');

xlabel('Concurrent batch size B  [requests]');
ylabel('Mean TTFT  [ms]');
title(sprintf('Chapter 4 — Identified TTFT(B) curve\nModel: %s  |  %d reps per B', ...
    cfg.model, cfg.n_reps));
legend('Location', 'best');
grid on;
ylim([0, max(l_mean_b) * 1.2]);

plot_path = fullfile(id_dir, 'id_ttft_curve.png');
saveas(fig1, plot_path);
fprintf('[plot] Saved: %s\n\n', plot_path);

% =========================================================================
% SUMMARY
% =========================================================================
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  IDENTIFIED PARAMETERS  (%s)\n', cfg.model);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Polynomial fit:  l(B) = c0 + c1*B + c2*B^2\n');
fprintf('║    c0 = %8.4f  ms\n', c0);
fprintf('║    c1 = %8.4f  ms/req\n', c1);
fprintf('║    c2 = %8.4f  ms/req^2\n', c2);
fprintf('║    R^2 = %.4f\n', r2);
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Linearisation at B0 = %d:\n', B0);
fprintf('║    l(B0)    = %.2f ms\n', l0);
fprintf('║    beta_eff = %.4f ms/req   (dl/dB at B0)\n', beta_eff);
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% =========================================================================
% SAVE
% =========================================================================
identified.c0        = c0;
identified.c1        = c1;
identified.c2        = c2;
identified.r2        = r2;
identified.B0        = B0;
identified.l0        = l0;
identified.beta_eff  = beta_eff;
identified.model     = cfg.model;
identified.timestamp = char(datetime('now'));
identified.raw.b_sweep    = cfg.b_sweep;
identified.raw.l_mean_b   = l_mean_b;
identified.raw.l_std_b    = l_std_b;

out_path = fullfile(id_dir, 'identified_params.mat');
save(out_path, 'identified');
fprintf('[save] Results saved: %s\n', out_path);
fprintf('\nPaste into setup_plant.m after loading identified_params.mat:\n');
fprintf('  perturbed.beta_eff = identified.beta_eff;  %% %.4f ms/req\n', beta_eff);
fprintf('  perturbed.B0       = identified.B0;         %% %d\n', B0);
fprintf('  perturbed.l0       = identified.l0;         %% %.2f ms\n', l0);
