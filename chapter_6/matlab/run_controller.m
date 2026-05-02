%% run_controller.m  --  Chapter 6: Cascade controller on Intel Mac queue server
%
% Runs the cascade controller against the real queue server.
% Each tick:
%   1. Read q_sw and l_total from /metrics
%   2. Outer loop: l_total -> q_ref
%   3. Inner loop: q_sw -> B
%   4. Send B via POST /control
%
% Disturbance schedule (segments run consecutively):
%   SEGMENTS(:) = struct with fields: ticks, lambda, L_target, label
%
% Load is injected by posting to /enqueue each tick (Poisson arrivals).
% The queue server's dispatcher handles actual firing to Ollama.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% -------------------------------------------------------------------------
% Load controller
% -------------------------------------------------------------------------
out_dir = fileparts(mfilename('fullpath'));
load(fullfile(out_dir, 'controller_params.mat'));

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 6: Cascade Controller Run                           ║\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Inner: K_q=%.4f  K_i=%.4f  B0=%d  B=[%d,%d]\n', K_q, K_i, B0, B_MIN, B_MAX);
fprintf('║  Outer: K_il=%.8f  tau_out=%.0fs\n', K_il, TAU_OUT);
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% -------------------------------------------------------------------------
% Disturbance schedule
% -------------------------------------------------------------------------
% Each segment: ticks, lambda (arrivals/tick), L_target (ms), label
SEGMENTS = struct( ...
    'ticks',    {30,  90,  40,  30,  30,  40,  30}, ...
    'lambda',   {3,   6,   3,   1,   3,   3,   3}, ...
    'L_target', {600, 600, 600, 600, 600, 400, 600}, ...
    'label',    {'Steady', 'Lambda Spike (90t)', 'Recovery', ...
                 'Lambda Drop', 'Recovery', 'Target Drop (400ms)', 'Target Restore'} ...
);

total_ticks = sum([SEGMENTS.ticks]);
fprintf('Schedule: %d ticks  dt=%.1fs  ~%.0f min\n', total_ticks, DT, total_ticks*DT/60);
for s = 1:numel(SEGMENTS)
    fprintf('  %3d ticks  lambda=%d  L_target=%d ms  [%s]\n', ...
        SEGMENTS(s).ticks, SEGMENTS(s).lambda, SEGMENTS(s).L_target, SEGMENTS(s).label);
end
fprintf('\n');

% -------------------------------------------------------------------------
% Controller state
% -------------------------------------------------------------------------
xi_l = 0;   % outer integrator
xi_q = 0;   % inner integrator

% -------------------------------------------------------------------------
% Log pre-allocation
% -------------------------------------------------------------------------
log_tick     = zeros(total_ticks, 1);
log_lambda   = zeros(total_ticks, 1);
log_L_target = zeros(total_ticks, 1);
log_q_sw     = zeros(total_ticks, 1);
log_q_ref    = zeros(total_ticks, 1);
log_B        = zeros(total_ticks, 1);
log_l_meas   = zeros(total_ticks, 1);
log_e_l      = zeros(total_ticks, 1);
log_e_q      = zeros(total_ticks, 1);
log_xi_l     = zeros(total_ticks, 1);
log_xi_q     = zeros(total_ticks, 1);
log_label    = cell(total_ticks, 1);

% -------------------------------------------------------------------------
% Reset server and set initial B
% -------------------------------------------------------------------------
server_post(SERVER, '/reset', struct());
server_post(SERVER, '/control', struct('B', B0));
pause(1);

PROMPTS = {'What is 2+2?','Name a colour.','Capital of France?', ...
           'Days in a week?','Name a planet.','Speed of light?', ...
           'Name a mammal.','10 times 10?','Colour of sky?', ...
           'Name a fruit.','Hours in a day?','5 squared?'};
prompt_idx = 1;

fprintf('%5s %3s %6s %5s %6s %3s %7s %7s %6s  label\n', ...
    'tick','lam','L_tgt','q_sw','q_ref','B','l_meas','e_l','e_q');
fprintf('%s\n', repmat('-',1,72));

tick = 0;
for si = 1:numel(SEGMENTS)
    seg = SEGMENTS(si);
    for st = 1:seg.ticks
        tick     = tick + 1;
        lam      = seg.lambda;
        L_tgt    = seg.L_target;
        t_tick   = tic;

        % ── 1. Read metrics ───────────────────────────────────────────
        m = server_get(SERVER, '/metrics');
        q_sw = m.q_sw;
        if isempty(m.l_total_mean) || isnan_safe(m.l_total_mean)
            l_meas = L_tgt;   % fallback on cold start
        else
            l_meas = m.l_total_mean;
        end

        % ── 2. Poisson arrivals → enqueue ─────────────────────────────
        a_k = poissrnd(lam);
        for ai = 1:a_k
            p = PROMPTS{mod(prompt_idx-1, numel(PROMPTS)) + 1};
            prompt_idx = prompt_idx + 1;
            server_post(SERVER, '/enqueue', struct('prompt', p));
        end

        % ── 3. Cascade control ────────────────────────────────────────

        % Outer: l_total -> q_ref
        % K_il > 0: l < target -> xi_l up -> q_ref up -> inner reduces B -> latency rises
        % K_il > 0: l > target -> xi_l down -> q_ref down -> inner increases B -> queue drains -> latency falls
        e_l      = L_tgt - l_meas;
        xi_l_sat = clamp(xi_l, xi_l_min, xi_l_max);
        q_ref    = clamp(Q0 + K_il * xi_l_sat, 0, Q_MAX);

        % Anti-windup outer: freeze if saturated AND error pushes further into bound
        at_lo_l = (q_ref <= 0)     && (e_l < 0);
        at_hi_l = (q_ref >= Q_MAX) && (e_l > 0);
        if ~(at_lo_l || at_hi_l)
            xi_l = clamp(xi_l_sat + e_l, xi_l_min, xi_l_max);
        else
            xi_l = xi_l_sat;
        end

        % Inner: q_sw -> B
        % K_q > 0: q_sw > q_ref (e_q < 0) -> dB > 0 -> B up -> drains queue
        e_q      = q_ref - q_sw;
        xi_q_sat = clamp(xi_q, xi_q_min, xi_q_max);
        dB       = -(K_q * e_q + K_i * xi_q_sat);
        B_cmd    = round(clamp(B0 + dB, B_MIN, B_MAX));

        % Anti-windup inner
        at_lo_q = (B_cmd <= B_MIN) && (e_q > 0);
        at_hi_q = (B_cmd >= B_MAX) && (e_q < 0);
        if ~(at_lo_q || at_hi_q)
            xi_q = clamp(xi_q_sat + e_q, xi_q_min, xi_q_max);
        else
            xi_q = xi_q_sat;
        end

        % ── 4. Send B to server ───────────────────────────────────────
        server_post(SERVER, '/control', struct('B', B_cmd));

        % ── 5. Log ───────────────────────────────────────────────────
        log_tick(tick)     = tick;
        log_lambda(tick)   = lam;
        log_L_target(tick) = L_tgt;
        log_q_sw(tick)     = q_sw;
        log_q_ref(tick)    = q_ref;
        log_B(tick)        = B_cmd;
        log_l_meas(tick)   = l_meas;
        log_e_l(tick)      = e_l;
        log_e_q(tick)      = e_q;
        log_xi_l(tick)     = xi_l;
        log_xi_q(tick)     = xi_q;
        log_label{tick}    = seg.label;

        fprintf('%5d %3d %6d %5d %6.1f %3d %7.1f %7.1f %6.1f  %s\n', ...
            tick, lam, L_tgt, q_sw, q_ref, B_cmd, l_meas, e_l, e_q, seg.label);

        % ── 6. Tick clock ────────────────────────────────────────────
        elapsed = toc(t_tick);
        if elapsed < DT
            pause(DT - elapsed);
        end
    end
end

% -------------------------------------------------------------------------
% Per-segment summary
% -------------------------------------------------------------------------
fprintf('\n=== Per-segment summary ===\n');
t0 = 1;
for si = 1:numel(SEGMENTS)
    seg = SEGMENTS(si);
    t1  = t0 + seg.ticks - 1;
    sl  = log_l_meas(t0:t1);
    sl  = sl(sl > 0);
    if ~isempty(sl)
        l_p95 = prctile(sl, 95);
        fprintf('  %-22s  lambda=%d  L_tgt=%d  l_mean=%.0fms  p95=%.0fms  B_mean=%.1f  q_mean=%.1f\n', ...
            seg.label, seg.lambda, seg.L_target, ...
            mean(sl), l_p95, ...
            mean(log_B(t0:t1)), mean(log_q_sw(t0:t1)));
    end
    t0 = t1 + 1;
end

% -------------------------------------------------------------------------
% Save log
% -------------------------------------------------------------------------
log = struct( ...
    'tick', log_tick, 'lambda', log_lambda, 'L_target', log_L_target, ...
    'q_sw', log_q_sw, 'q_ref', log_q_ref, 'B', log_B, ...
    'l_meas', log_l_meas, 'e_l', log_e_l, 'e_q', log_e_q, ...
    'xi_l', log_xi_l, 'xi_q', log_xi_q, 'label', {log_label});
save(fullfile(out_dir, 'run_log.mat'), 'log', 'SEGMENTS');
fprintf('\n[save] run_log.mat\n');

% -------------------------------------------------------------------------
% Plot
% -------------------------------------------------------------------------
plot_results(log, SEGMENTS, total_ticks, out_dir);

% =========================================================================
% Helpers
% =========================================================================
function m = server_get(server, path)
    cmd = sprintf('curl -s "%s%s"', server, path);
    [~, out] = system(cmd);
    m = jsondecode(strtrim(out));
end

function r = server_post(server, path, data)
    body = jsonencode(data);
    body = strrep(body, '"', '\"');
    cmd  = sprintf('curl -s -X POST "%s%s" -H "Content-Type: application/json" -d "%s"', ...
                   server, path, body);
    [~, out] = system(cmd);
    try
        r = jsondecode(strtrim(out));
    catch
        r = struct();
    end
end

function v = clamp(x, lo, hi)
    v = max(lo, min(hi, x));
end

function b = isnan_safe(x)
    b = ~isnumeric(x) || isnan(x);
end

function plot_results(log, SEGMENTS, N, out_dir)
    COLORS = {[0.94 0.94 0.94],[1 0.91 0.91],[0.91 1 0.91], ...
              [0.91 0.91 1],[1 0.97 0.91],[1 0.91 1],[0.91 0.97 1]};

    fig = figure('Visible','off','Position',[50 50 1400 1000]);

    axs = gobjects(4,1);
    for ai = 1:4
        axs(ai) = subplot(4,1,ai);
    end

    % Shade segments
    t0 = 1;
    for si = 1:numel(SEGMENTS)
        t1 = t0 + SEGMENTS(si).ticks - 1;
        c  = COLORS{mod(si-1,numel(COLORS))+1};
        for ai = 1:4
            patch(axs(ai), [t0 t1 t1 t0],[get(axs(ai),'YLim') fliplr(get(axs(ai),'YLim'))], ...
                c,'FaceAlpha',0.3,'EdgeColor','none'); hold(axs(ai),'on');
        end
        text(axs(1),(t0+t1)/2, 0.97, SEGMENTS(si).label, ...
            'Units','data','HorizontalAlignment','center', ...
            'VerticalAlignment','top','FontSize',7,'Color',[0.3 0.3 0.3], ...
            'Parent', axs(1));
        t0 = t1 + 1;
    end

    % Panel 1: latency
    plot(axs(1), log.tick, log.l_meas, 'b-', 'LineWidth', 1.2);
    stairs(axs(1), log.tick, log.L_target, 'k--', 'LineWidth', 1.5);
    ylabel(axs(1), 'l_{total} [ms]');
    legend(axs(1), 'l_{total}', 'L_{target}', 'Location', 'northeast');
    title(axs(1), 'Chapter 6 — Cascade Controller on Intel Mac Queue Server');
    grid(axs(1), 'on');

    % Panel 2: queue
    area(axs(2), log.tick, log.q_sw, 'FaceColor', [1 0.65 0], 'FaceAlpha', 0.4);
    hold(axs(2), 'on');
    stairs(axs(2), log.tick, log.q_sw,  'Color',[0.85 0.45 0], 'LineWidth', 1.2);
    stairs(axs(2), log.tick, log.q_ref, 'g--', 'LineWidth', 1.5);
    ylabel(axs(2), 'Queue [req]');
    legend(axs(2), 'q_{sw} (FIFO)', 'q_{ref} (outer cmd)', 'Location', 'northeast');
    grid(axs(2), 'on');

    % Panel 3: B and lambda
    stairs(axs(3), log.tick, log.B,      'm-',  'LineWidth', 1.5);
    hold(axs(3), 'on');
    stairs(axs(3), log.tick, log.lambda, 'k--', 'LineWidth', 1.0);
    ylabel(axs(3), 'Req / tick');
    legend(axs(3), 'B (dispatch)', '\lambda (arrivals)', 'Location', 'northeast');
    grid(axs(3), 'on');

    % Panel 4: integrator states
    yyaxis(axs(4), 'left');
    plot(axs(4), log.tick, log.xi_l, 'g-', 'LineWidth', 1.2);
    yline(axs(4), 0, 'k--', 'LineWidth', 0.7);
    ylabel(axs(4), '\xi_l (outer)');
    yyaxis(axs(4), 'right');
    plot(axs(4), log.tick, log.xi_q, 'r--', 'LineWidth', 1.0);
    ylabel(axs(4), '\xi_q (inner)');
    xlabel(axs(4), 'Tick [k]');
    legend(axs(4), '\xi_l', '\xi_q', 'Location', 'northwest');
    grid(axs(4), 'on');

    for ai = 1:4; xlim(axs(ai), [1 N]); end

    ts   = datestr(now,'HHMMSS');
    path = fullfile(out_dir, sprintf('ch6_cascade_%s.png', ts));
    saveas(fig, path);
    fprintf('[plot] %s\n', path);
end
