%% ch2_run_and_plot.m — Run Chapter 2 Simulink and generate polished plot
% Must be run from chapter_2/src/

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% -- 1. Run setup to get workspace variables ----------------------------------
setup_plant;

% -- 2. Run the Simulink model ------------------------------------------------
mdl = 'llm_inferencing_control';
mdl_path = fullfile(fileparts(mfilename('fullpath')), '..', 'simulink_model', [mdl '.slx']);
load_system(mdl_path);
out = sim(mdl);

% -- 3. Extract logged signals ------------------------------------------------
% The To Workspace blocks log: slx_q, slx_B, slx_Lp95
% Try to get timeseries from the sim output
try
    ts_q    = out.slx_q;
    ts_B    = out.slx_B;
    ts_Lp95 = out.slx_Lp95;
catch
    % If they're in base workspace as timeseries
    ts_q    = evalin('base', 'slx_q');
    ts_B    = evalin('base', 'slx_B');
    ts_Lp95 = evalin('base', 'slx_Lp95');
end

% Extract time and data
if isa(ts_q, 'timeseries')
    t     = ts_q.Time;
    log_q = ts_q.Data(:);
    log_B = ts_B.Data(:);
    log_L = ts_Lp95.Data(:);
elseif isa(ts_q, 'Simulink.SimulationData.Dataset')
    % Dataset format
    sig_q = ts_q{1}.Values;
    sig_B = ts_B{1}.Values;
    sig_L = ts_Lp95{1}.Values;
    t     = sig_q.Time;
    log_q = sig_q.Data(:);
    log_B = sig_B.Data(:);
    log_L = sig_L.Data(:);
else
    % struct with time/signals
    t     = ts_q.time;
    log_q = ts_q.signals.values(:);
    log_B = ts_B.signals.values(:);
    log_L = ts_Lp95.signals.values(:);
end

% q_ref is constant in Chapter 2 (perturbed.q0)
log_q_ref = perturbed.q0 * ones(size(t));

% L_target
log_L_target = perturbed.L_p95_target * ones(size(t));

% -- 4. Try to get lambda (arrival rate) from logged output -------------------
has_lambda = false;
try
    if isfield(out, 'slx_lambda')
        ts_lam = out.slx_lambda;
    else
        ts_lam = evalin('base', 'slx_lambda');
    end
    if isa(ts_lam, 'timeseries')
        log_lambda = ts_lam.Data(:);
    else
        log_lambda = ts_lam.signals.values(:);
    end
    has_lambda = true;
catch
    % lambda not logged — reconstruct from step blocks
    % Step_spike_on at t=5s adds lambda_mean, Step_spike_off at t=15s subtracts it
    log_lambda = perturbed.lambda_mean * ones(size(t));
    log_lambda(t >= 5 & t < 15) = 2 * perturbed.lambda_mean;
    has_lambda = true;
end

% -- 5. Generate polished plot ------------------------------------------------
fig = figure('Position', [100 100 1100 850], 'Color', 'w');

if has_lambda
    n_panels = 4;
else
    n_panels = 3;
end

% Panel 1: Latency
ax1 = subplot(n_panels, 1, 1);
plot(t, log_L, '-', 'Color', [0 0.447 0.741], 'LineWidth', 1.5); hold on;
plot(t, log_L_target, '--', 'Color', [0.85 0.125 0.098], 'LineWidth', 1.5);
ylabel('L_{p95} [ms]', 'FontSize', 11);
legend('L_{p95}', 'L_{target}', 'Location', 'northeast', 'FontSize', 9);
title('Closed-loop cascade response: latency, queue, and batch regulation', ...
      'FontSize', 13, 'FontWeight', 'bold');
set(ax1, 'XTickLabel', [], 'FontSize', 10);
grid on; box on;

% y-axis padding
y = [log_L(:); log_L_target(:)];
y = y(isfinite(y));
if ~isempty(y)
    ymin = min(y); ymax = max(y);
    pad = max(1, 0.08 * (ymax - ymin));
    ylim(ax1, [ymin - pad, ymax + pad]);
end

% Panel 2: Queue depth
ax2 = subplot(n_panels, 1, 2);
plot(t, log_q, '-', 'Color', [0.85 0.325 0.098], 'LineWidth', 1.5); hold on;
plot(t, log_q_ref, '--', 'Color', [0.133 0.545 0.133], 'LineWidth', 1.5);
ylabel('q', 'FontSize', 11);
legend('q', 'q_{ref}', 'Location', 'northeast', 'FontSize', 9);
set(ax2, 'XTickLabel', [], 'FontSize', 10);
grid on; box on;

y = [log_q(:); log_q_ref(:)];
y = y(isfinite(y));
if ~isempty(y)
    ymin = min(y); ymax = max(y);
    pad = max(1, 0.08 * (ymax - ymin));
    ylim(ax2, [ymin - pad, ymax + pad]);
end

% Panel 3: Batch size
ax3 = subplot(n_panels, 1, 3);
plot(t, log_B, '-', 'Color', [0.494 0.184 0.556], 'LineWidth', 1.5);
ylabel('B', 'FontSize', 11);
legend('B', 'Location', 'northeast', 'FontSize', 9);
grid on; box on;

if has_lambda && n_panels == 4
    set(ax3, 'XTickLabel', [], 'FontSize', 10);
else
    xlabel('Time (s)', 'FontSize', 11);
    set(ax3, 'FontSize', 10);
end

y = log_B(:);
y = y(isfinite(y));
if ~isempty(y)
    ymin = min(y); ymax = max(y);
    pad = max(1, 0.08 * (ymax - ymin));
    ylim(ax3, [ymin - pad, ymax + pad]);
end

% Panel 4: Arrival rate (if available)
if has_lambda && n_panels == 4
    ax4 = subplot(n_panels, 1, 4);
    plot(t, log_lambda, '-', 'Color', [0 0.8 0.8], 'LineWidth', 1.5);
    ylabel('\lambda / completions', 'FontSize', 11);
    xlabel('Time (s)', 'FontSize', 11);
    legend('\lambda', 'Location', 'northeast', 'FontSize', 9);
    set(ax4, 'FontSize', 10);
    grid on; box on;

    y = log_lambda(:);
    y = y(isfinite(y));
    if ~isempty(y)
        ymin = min(y); ymax = max(y);
        pad = max(1, 0.08 * (ymax - ymin));
        ylim(ax4, [ymin - pad, ymax + pad]);
    end
end

% Link x-axes
if has_lambda && n_panels == 4
    linkaxes([ax1 ax2 ax3 ax4], 'x');
else
    linkaxes([ax1 ax2 ax3], 'x');
end

% -- 6. Export ----------------------------------------------------------------
out_png = fullfile(fileparts(mfilename('fullpath')), '..', 'ch2_closed_loop_polished.png');
exportgraphics(fig, out_png, 'Resolution', 220);
fprintf('[save] %s\n', out_png);
close(fig);
