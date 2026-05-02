%% run_controller.m  --  Chapter 6: Cascade controller on Intel Mac queue server
%
% OPERATING PARAMETERS (derived from characterise.m measurements)
% ---------------------------------------------------------------
% Intel Mac + qwen2.5:0.5b + OLLAMA_NUM_PARALLEL=4:
%
%   TTFT(B=1) ≈  580ms    TTFT(B=2) ≈  890ms
%   TTFT(B=3) ≈ 1100ms    TTFT(B=4) ≈ 1250ms   <- B_max ceiling
%
%   lambda_ss   = 2 req/tick   (50% of B_max capacity -- stable headroom)
%   l_ss        = TTFT(B=2) ≈ 890ms  (natural operating latency)
%   L_target    = 1600ms  (1.8 × l_ss -- above equilibrium so controller
%                            has room below AND above to regulate)
%   lambda_spike= 4 req/tick   (= B_max -- stresses queue but stays drainable)
%   Target-drop = 800ms   (below l_ss -- forces controller to reduce B to 1)
%
% DISTURBANCE SCHEDULE
% --------------------
%   Steady (30t):        lambda=2   L=1600ms  -- baseline
%   Lambda Spike (60t):  lambda=4   L=1600ms  -- queue builds, B should rise
%   Recovery (30t):      lambda=2   L=1600ms  -- queue drains
%   Lambda Drop (20t):   lambda=1   L=1600ms  -- under-load, B should fall
%   Recovery (20t):      lambda=2   L=1600ms  -- return to steady
%   Target Drop (30t):   lambda=2   L=800ms   -- tighter SLA, B must drop
%   Target Restore (20t):lambda=2   L=1600ms  -- restore target
%
% Total: 210 ticks = 3.5 min

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% ── Load controller ───────────────────────────────────────────────────────
out_dir = fileparts(mfilename('fullpath'));
load(fullfile(out_dir, 'controller_params.mat'));

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 6: Cascade Controller Run                           ║\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Inner: K_q=%.4f  K_i=%.4f  B0=%d  B=[%d,%d]\n', K_q,K_i,B0,B_MIN,B_MAX);
fprintf('║  Outer: K_il=%.8f  tau_out=%.0fs\n', K_il, TAU_OUT);
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ── Disturbance schedule ──────────────────────────────────────────────────
% L_target_ss and lambda values come from characterise.m recommendations.
% If identified_params.mat has them, use those; otherwise use defaults.
L_TARGET_SS = 1600;    % default: 1.8 × TTFT(B=2)
L_TARGET_TIGHT = 800;  % default: below equilibrium to force B reduction
LAMBDA_SS    = 2;
LAMBDA_SPIKE = 4;      % = B_MAX -- maximum sustainable
LAMBDA_DROP  = 1;

if exist('L_target_recommended','var') && L_target_recommended > 0
    L_TARGET_SS    = L_target_recommended;
    L_TARGET_TIGHT = round(L_target_recommended * 0.5 / 100) * 100;
    fprintf('Using identified L_target=%.0fms  L_tight=%.0fms\n', ...
        L_TARGET_SS, L_TARGET_TIGHT);
end
if exist('lambda_ss','var');    LAMBDA_SS    = lambda_ss;    end
if exist('lambda_spike','var'); LAMBDA_SPIKE = min(B_MAX, lambda_spike); end

SEGMENTS = struct( ...
    'ticks',    {30,  60,  30,  20,  20,  30,  20}, ...
    'lambda',   {LAMBDA_SS, LAMBDA_SPIKE, LAMBDA_SS, LAMBDA_DROP, ...
                 LAMBDA_SS, LAMBDA_SS, LAMBDA_SS}, ...
    'L_target', {L_TARGET_SS, L_TARGET_SS, L_TARGET_SS, L_TARGET_SS, ...
                 L_TARGET_SS, L_TARGET_TIGHT, L_TARGET_SS}, ...
    'label',    {'Steady', 'Lambda Spike', 'Recovery', ...
                 'Lambda Drop', 'Recovery', 'Target Drop', 'Restore'} ...
);

total_ticks = sum([SEGMENTS.ticks]);
fprintf('\nSchedule: %d ticks  dt=%.1fs  (~%.0f min)\n', ...
    total_ticks, DT, total_ticks*DT/60);
fprintf('  %6s  %6s  %8s  %s\n', 'ticks','lambda','L_target','label');
for s = 1:numel(SEGMENTS)
    fprintf('  %6d  %6d  %8d  %s\n', ...
        SEGMENTS(s).ticks, SEGMENTS(s).lambda, SEGMENTS(s).L_target, SEGMENTS(s).label);
end
fprintf('\n');

% ── Controller state ──────────────────────────────────────────────────────
xi_l = 0;
xi_q = 0;

% ── Log pre-allocation ────────────────────────────────────────────────────
N   = total_ticks;
log_tick     = zeros(N,1); log_lambda   = zeros(N,1);
log_L_target = zeros(N,1); log_q_sw     = zeros(N,1);
log_q_ref    = zeros(N,1); log_B        = zeros(N,1);
log_l_meas   = zeros(N,1); log_e_l      = zeros(N,1);
log_e_q      = zeros(N,1); log_xi_l     = zeros(N,1);
log_xi_q     = zeros(N,1); log_label    = cell(N,1);

% ── Reset and prime ───────────────────────────────────────────────────────
server_post(SERVER, '/reset', struct());
server_post(SERVER, '/control', struct('B', B0));
pause(1);

PROMPTS = {'What is 2+2?','Name a colour.','Capital of France?', ...
           'Days in a week?','Name a planet.','Speed of light?', ...
           'Name a mammal.','10 times 10?','Colour of sky?', ...
           'Name a fruit.','Hours in a day?','5 squared?'};
prompt_idx = 1;

fprintf('%5s  %3s  %6s  %5s  %6s  %2s  %7s  %7s  %6s  label\n', ...
    'tick','λ','L_tgt','q_sw','q_ref','B','l_meas','e_l','e_q');
fprintf('%s\n', repmat('-', 1, 75));

% ── Main loop ─────────────────────────────────────────────────────────────
tick = 0;
for si = 1:numel(SEGMENTS)
    seg = SEGMENTS(si);
    for st = 1:seg.ticks
        tick   = tick + 1;
        lam    = seg.lambda;
        L_tgt  = seg.L_target;
        t_tick = tic;

        % 1. Observe
        m = server_get(SERVER, '/metrics');
        q_sw = m.q_sw;
        if isempty(m.l_total_mean) || isnan_safe(m.l_total_mean)
            l_meas = L_tgt;
        else
            l_meas = m.l_total_mean;
        end

        % 2. Inject Poisson arrivals
        a_k = poissrnd(lam);
        for ai = 1:a_k
            p = PROMPTS{mod(prompt_idx-1, numel(PROMPTS)) + 1};
            prompt_idx = prompt_idx + 1;
            server_post(SERVER, '/enqueue', struct('prompt', p));
        end

        % 3. Outer loop: l_total → q_ref
        %    K_il > 0:
        %      l < L_tgt (e_l > 0) → xi_l↑ → q_ref↑ → inner: B↓ → less concurrency → TTFT↑ → l↑
        %      l > L_tgt (e_l < 0) → xi_l↓ → q_ref↓ → inner: B↑ → more concurrency → queue drains → l↓
        e_l      = L_tgt - l_meas;
        xi_l_sat = clamp(xi_l, xi_l_min, xi_l_max);
        q_ref    = clamp(Q0 + K_il * xi_l_sat, 0, Q_MAX);

        % Anti-windup: freeze when saturated and error pushes further
        at_lo_l = (q_ref <= 0)     && (e_l < 0);
        at_hi_l = (q_ref >= Q_MAX) && (e_l > 0);
        if ~(at_lo_l || at_hi_l)
            xi_l = clamp(xi_l_sat + e_l, xi_l_min, xi_l_max);
        else
            xi_l = xi_l_sat;
        end

        % 4. Inner loop: q_sw → B
        %    K_q > 0: q_sw > q_ref (e_q < 0) → dB > 0 → B↑ → drains queue faster
        e_q      = q_ref - q_sw;
        xi_q_sat = clamp(xi_q, xi_q_min, xi_q_max);
        dB       = -(K_q * e_q + K_i * xi_q_sat);
        B_cmd    = round(clamp(B0 + dB, B_MIN, B_MAX));

        at_lo_q = (B_cmd <= B_MIN) && (e_q > 0);
        at_hi_q = (B_cmd >= B_MAX) && (e_q < 0);
        if ~(at_lo_q || at_hi_q)
            xi_q = clamp(xi_q_sat + e_q, xi_q_min, xi_q_max);
        else
            xi_q = xi_q_sat;
        end

        % 5. Actuate
        server_post(SERVER, '/control', struct('B', B_cmd));

        % 6. Log
        log_tick(tick)=tick; log_lambda(tick)=lam; log_L_target(tick)=L_tgt;
        log_q_sw(tick)=q_sw; log_q_ref(tick)=q_ref; log_B(tick)=B_cmd;
        log_l_meas(tick)=l_meas; log_e_l(tick)=e_l; log_e_q(tick)=e_q;
        log_xi_l(tick)=xi_l; log_xi_q(tick)=xi_q; log_label{tick}=seg.label;

        fprintf('%5d  %3d  %6d  %5d  %6.1f  %2d  %7.0f  %7.0f  %6.1f  %s\n', ...
            tick, lam, L_tgt, q_sw, q_ref, B_cmd, l_meas, e_l, e_q, seg.label);

        % 7. Tick clock
        elapsed = toc(t_tick);
        if elapsed < DT; pause(DT - elapsed); end
    end
end

% ── Per-segment summary ───────────────────────────────────────────────────
fprintf('\n%s\n', repmat('═',1,75));
fprintf('Per-segment summary:\n');
fprintf('  %-18s  %2s  %6s  %6s  %6s  %5s  %5s\n', ...
    'Segment','λ','L_tgt','l_mean','l_p95','B_avg','q_avg');
t0 = 1;
for si = 1:numel(SEGMENTS)
    seg = SEGMENTS(si);
    te  = t0 + seg.ticks - 1;
    sl  = log_l_meas(t0:te);  sl = sl(sl > 0);
    if ~isempty(sl)
        fprintf('  %-18s  %2d  %6d  %6.0f  %6.0f  %5.1f  %5.1f\n', ...
            seg.label, seg.lambda, seg.L_target, ...
            mean(sl), prctile(sl,95), ...
            mean(log_B(t0:te)), mean(log_q_sw(t0:te)));
    end
    t0 = te + 1;
end

% ── Save ──────────────────────────────────────────────────────────────────
log = struct('tick',log_tick,'lambda',log_lambda,'L_target',log_L_target, ...
    'q_sw',log_q_sw,'q_ref',log_q_ref,'B',log_B,'l_meas',log_l_meas, ...
    'e_l',log_e_l,'e_q',log_e_q,'xi_l',log_xi_l,'xi_q',log_xi_q, ...
    'label',{log_label});
save(fullfile(out_dir, 'run_log.mat'), 'log', 'SEGMENTS');
fprintf('\n[save] run_log.mat\n');

% ── Plot ──────────────────────────────────────────────────────────────────
plot_results(log, SEGMENTS, N, out_dir);

% =========================================================================
% Helpers
% =========================================================================
function m = server_get(server, path)
    [~,out] = system(sprintf('curl -s "%s%s"', server, path));
    m = jsondecode(strtrim(out));
end

function r = server_post(server, path, data)
    body = strrep(jsonencode(data), '"', '\"');
    [~,out] = system(sprintf( ...
        'curl -s -X POST "%s%s" -H "Content-Type: application/json" -d "%s"', ...
        server, path, body));
    try; r = jsondecode(strtrim(out)); catch; r = struct(); end
end

function v = clamp(x, lo, hi); v = max(lo, min(hi, x)); end
function b = isnan_safe(x);    b = ~isnumeric(x) || isnan(x); end

function plot_results(log, SEGMENTS, N, out_dir)
    COLORS = {[0.95 0.95 0.95],[1.00 0.90 0.90],[0.90 1.00 0.90], ...
              [0.90 0.90 1.00],[1.00 0.97 0.90],[1.00 0.90 1.00],[0.90 0.97 1.00]};

    fig = figure('Visible','off','Position',[50 50 1400 1000]);
    ax  = gobjects(4,1);
    for i=1:4; ax(i) = subplot(4,1,i); end

    % Shade + label segments
    t0 = 1;
    for si = 1:numel(SEGMENTS)
        te = t0 + SEGMENTS(si).ticks - 1;
        c  = COLORS{mod(si-1,numel(COLORS))+1};
        for ai=1:4
            patch(ax(ai),[t0 te te t0], ...
                [get(ax(ai),'YLim') fliplr(get(ax(ai),'YLim'))], ...
                c,'FaceAlpha',0.3,'EdgeColor','none'); hold(ax(ai),'on');
        end
        text(ax(1),(t0+te)/2, 0.97, SEGMENTS(si).label, ...
            'Units','data','HorizontalAlignment','center', ...
            'VerticalAlignment','top','FontSize',7.5,'Color',[0.2 0.2 0.2]);
        t0 = te + 1;
    end

    % Panel 1: latency
    plot(ax(1), log.tick, log.l_meas/1000, 'b-', 'LineWidth',1.3, ...
        'DisplayName','l_{total} [s]');
    stairs(ax(1), log.tick, log.L_target/1000, 'k--', 'LineWidth',1.5, ...
        'DisplayName','L_{target} [s]');
    ylabel(ax(1),'l_{total} [s]');
    legend(ax(1),'Location','northeast'); grid(ax(1),'on');
    title(ax(1),'Chapter 6 — Cascade Controller  (Intel Mac + qwen2.5:0.5b)');

    % Panel 2: queue
    area(ax(2), log.tick, log.q_sw, 'FaceColor',[1 0.65 0],'FaceAlpha',0.4);
    hold(ax(2),'on');
    stairs(ax(2), log.tick, log.q_sw,  'Color',[0.85 0.45 0],'LineWidth',1.2, ...
        'DisplayName','q_{sw} (FIFO)');
    stairs(ax(2), log.tick, log.q_ref, 'g--','LineWidth',1.5, ...
        'DisplayName','q_{ref} (outer)');
    ylabel(ax(2),'Queue [req]');
    legend(ax(2),'q_{sw}','q_{ref}','Location','northeast'); grid(ax(2),'on');

    % Panel 3: B and lambda
    stairs(ax(3), log.tick, log.B,      'm-', 'LineWidth',1.5,'DisplayName','B');
    hold(ax(3),'on');
    stairs(ax(3), log.tick, log.lambda, 'k--','LineWidth',1.0,'DisplayName','\lambda');
    ylabel(ax(3),'Req / tick');
    legend(ax(3),'Location','northeast'); grid(ax(3),'on');
    ylim(ax(3), [0 max(max(log.B), max(log.lambda))+1]);

    % Panel 4: integrators
    yyaxis(ax(4),'left');
    plot(ax(4), log.tick, log.xi_l,'g-','LineWidth',1.2,'DisplayName','\xi_l');
    yline(ax(4), 0,'k--','LineWidth',0.7);
    ylabel(ax(4),'\xi_l (outer)','Color','g');
    yyaxis(ax(4),'right');
    plot(ax(4), log.tick, log.xi_q,'r--','LineWidth',1.0,'DisplayName','\xi_q');
    ylabel(ax(4),'\xi_q (inner)','Color','r');
    xlabel(ax(4),'Tick [k]');
    l1=get(ax(4).YAxis(1),'Color'); %#ok
    legend(ax(4),'\xi_l','\xi_q','Location','northwest'); grid(ax(4),'on');

    for i=1:4; xlim(ax(i),[1 N]); end

    ts = datetime('now','Format','HHmmss');
    fn = fullfile(out_dir, sprintf('ch6_cascade_%s.png', char(ts)));
    saveas(fig, fn);
    fprintf('[plot] %s\n', fn);
end
