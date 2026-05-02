%% run_controller.m  --  Chapter 6: Cascade controller on Intel Mac queue server
%
% FIXES v3 (based on analysis of first run):
%
%   FIX 1 -- Use l_total_recent_mean (last 10 completions) instead of
%             l_total_mean (last 200). The 200-sample buffer is contaminated
%             by cold-start requests (3-5s) for the first ~100 ticks,
%             keeping l_meas > L_target and freezing the outer integrator.
%
%   FIX 2 -- Pre-warm Ollama before main loop: fire 12 requests at B=B0
%             and wait for them all to complete. This fills the 10-sample
%             recent buffer with hot measurements before tick 1.
%
%   FIX 3 -- lambda_spike = B_MAX * 2 (not = B_MAX).
%             When lambda_spike = B_MAX, arrivals = dispatch every tick,
%             so the queue never accumulates (it stays at 0).
%             With lambda_spike > B_MAX, the queue grows by
%             (lambda_spike - B_MAX) req/tick, giving the controller
%             something real to regulate.

clear; clc;
addpath(fileparts(mfilename('fullpath')));

% ── Load controller ───────────────────────────────────────────────────────
out_dir = fileparts(mfilename('fullpath'));
load(fullfile(out_dir, 'controller_params.mat'));

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 6: Cascade Controller Run  (v3)                     ║\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Inner: K_q=%.4f  K_i=%.4f  B0=%d  B=[%d,%d]\n', K_q,K_i,B0,B_MIN,B_MAX);
fprintf('║  Outer: K_il=%.8f  tau_out=%.0fs\n', K_il, TAU_OUT);
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ── Disturbance schedule ──────────────────────────────────────────────────
% Load recommended values from characterise.m if available
L_TARGET_SS    = 2400;   % default (from last characterise run)
L_TARGET_TIGHT = 1200;   % tight SLA target (50% of L_TARGET_SS)
LAMBDA_SS      = 4;      % steady-state arrival rate
LAMBDA_SPIKE   = 8;      % FIX 3: must be > B_MAX(=4) so queue builds
LAMBDA_DROP    = 1;      % under-load segment

% Override with identified values if they exist
if exist('L_target_recommended','var') && L_target_recommended > 0
    L_TARGET_SS    = L_target_recommended;
    L_TARGET_TIGHT = round(L_target_recommended * 0.5 / 100) * 100;
end
if exist('lambda_ss','var');    LAMBDA_SS = lambda_ss; end
% FIX 3: spike must exceed B_MAX -- always, regardless of characterise recommendation
LAMBDA_SPIKE = B_MAX * 2;   % 4*2 = 8: 4 arrive, 4 dispatched, 4 queue up per tick

fprintf('Operating points:\n');
fprintf('  lambda_ss    = %d  lambda_spike = %d  (> B_MAX=%d to build queue)\n', ...
    LAMBDA_SS, LAMBDA_SPIKE, B_MAX);
fprintf('  L_target_ss  = %d ms\n', L_TARGET_SS);
fprintf('  L_target_tight = %d ms\n\n', L_TARGET_TIGHT);

SEGMENTS = struct( ...
    'ticks',    {30,           60,            30,          20,          20,          30,             20}, ...
    'lambda',   {LAMBDA_SS,    LAMBDA_SPIKE,  LAMBDA_SS,   LAMBDA_DROP, LAMBDA_SS,   LAMBDA_SS,      LAMBDA_SS}, ...
    'L_target', {L_TARGET_SS,  L_TARGET_SS,   L_TARGET_SS, L_TARGET_SS, L_TARGET_SS, L_TARGET_TIGHT, L_TARGET_SS}, ...
    'label',    {'Steady',     'λ↑ Spike',    'Recovery',  'λ↓ Drop',   'Recovery',  'Target Drop',  'Restore'} ...
);

total_ticks = sum([SEGMENTS.ticks]);
fprintf('Schedule: %d ticks  dt=%.1fs  (~%.0f min)\n', total_ticks, DT, total_ticks*DT/60);
fprintf('  %6s  %6s  %8s  %s\n', 'ticks','lambda','L_target','label');
for s = 1:numel(SEGMENTS)
    fprintf('  %6d  %6d  %8d  %s\n', ...
        SEGMENTS(s).ticks, SEGMENTS(s).lambda, SEGMENTS(s).L_target, SEGMENTS(s).label);
end
fprintf('\n');

% ── FIX 2: Pre-warm before main loop ─────────────────────────────────────
fprintf('[pre-warm] Firing %d requests to fill recent buffer before tick 1...\n', 12);
server_post(SERVER, '/reset', struct());
server_post(SERVER, '/control', struct('B', B0));

PROMPTS = {'What is 2+2?','Name a colour.','Capital of France?', ...
           'Days in a week?','Name a planet.','Speed of light?', ...
           'Name a mammal.','10 times 10?','Colour of sky?', ...
           'Name a fruit.','Hours in a day?','5 squared?'};

% Enqueue 12 warm-up requests (fills the 10-sample recent buffer + 2 extra)
for i = 1:12
    server_post(SERVER, '/enqueue', struct('prompt', PROMPTS{mod(i-1,numel(PROMPTS))+1}));
end

% Wait for all 12 to complete (polls up to 120s)
fprintf('  Waiting for warm-up completions...\n');
t_wait = tic;
while true
    m = server_get(SERVER, '/metrics');
    if m.completed >= 12; break; end
    if toc(t_wait) > 120
        fprintf('  WARNING: warm-up timed out (%d/12 completed)\n', m.completed);
        break;
    end
    pause(1);
end
m = server_get(SERVER, '/metrics');
fprintf('  Warm-up done: completed=%d  l_recent=%.0fms  l_mean=%.0fms\n', ...
    m.completed, ...
    ifelse(isempty(m.l_total_recent_mean), 0, m.l_total_recent_mean), ...
    ifelse(isempty(m.l_total_mean), 0, m.l_total_mean));

% Reset metrics (clears contaminated long buffer) but keep Ollama hot
server_post(SERVER, '/reset', struct());
server_post(SERVER, '/control', struct('B', B0));
pause(1);

% Fire one more batch so recent buffer has fresh samples from tick 1
for i = 1:4
    server_post(SERVER, '/enqueue', struct('prompt', PROMPTS{i}));
end
pause(6);  % wait for them to complete before starting control loop
fprintf('[pre-warm] Complete. Starting control loop.\n\n');

% ── Controller state ──────────────────────────────────────────────────────
xi_l = 0;
xi_q = 0;

% ── Log pre-allocation ────────────────────────────────────────────────────
N = total_ticks;
log_tick=zeros(N,1); log_lambda=zeros(N,1); log_L_target=zeros(N,1);
log_q_sw=zeros(N,1); log_q_ref=zeros(N,1); log_B=zeros(N,1);
log_l_meas=zeros(N,1); log_e_l=zeros(N,1); log_e_q=zeros(N,1);
log_xi_l=zeros(N,1); log_xi_q=zeros(N,1); log_label=cell(N,1);

server_post(SERVER, '/reset', struct());
server_post(SERVER, '/control', struct('B', B0));
prompt_idx = 1;
pause(1);

fprintf('%5s  %3s  %6s  %5s  %6s  %2s  %7s  %7s  %6s  label\n', ...
    'tick','λ','L_tgt','q_sw','q_ref','B','l_meas','e_l','e_q');
fprintf('%s\n', repmat('-',1,75));

% ── Main loop ─────────────────────────────────────────────────────────────
tick = 0;
for si = 1:numel(SEGMENTS)
    seg = SEGMENTS(si);
    for st = 1:seg.ticks
        tick   = tick + 1;
        lam    = seg.lambda;
        L_tgt  = seg.L_target;
        t_tick = tic;

        % 1. Observe -- FIX 1: use l_total_recent_mean (last 10), not l_total_mean
        m    = server_get(SERVER, '/metrics');
        q_sw = m.q_sw;
        if isfield(m,'l_total_recent_mean') && ~isempty(m.l_total_recent_mean) ...
                && ~isnan_safe(m.l_total_recent_mean) && m.l_total_recent_n >= 3
            l_meas = m.l_total_recent_mean;   % hot, recent window
        elseif ~isempty(m.l_total_mean) && ~isnan_safe(m.l_total_mean)
            l_meas = m.l_total_mean;           % fallback to long window
        else
            l_meas = L_tgt;                    % last resort: use target
        end

        % 2. Poisson arrivals → FIFO
        a_k = poissrnd(lam);
        for ai = 1:a_k
            p = PROMPTS{mod(prompt_idx-1, numel(PROMPTS)) + 1};
            prompt_idx = prompt_idx + 1;
            server_post(SERVER, '/enqueue', struct('prompt', p));
        end

        % 3. Outer loop: l_total → q_ref
        e_l      = L_tgt - l_meas;
        xi_l_sat = clamp(xi_l, xi_l_min, xi_l_max);
        q_ref    = clamp(Q0 + K_il * xi_l_sat, 0, Q_MAX);

        at_lo_l = (q_ref <= 0)     && (e_l < 0);
        at_hi_l = (q_ref >= Q_MAX) && (e_l > 0);
        if ~(at_lo_l || at_hi_l)
            xi_l = clamp(xi_l_sat + e_l, xi_l_min, xi_l_max);
        else
            xi_l = xi_l_sat;
        end

        % 4. Inner loop: q_sw → B
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
            tick,lam,L_tgt,q_sw,q_ref,B_cmd,l_meas,e_l,e_q,seg.label);

        elapsed = toc(t_tick);
        if elapsed < DT; pause(DT - elapsed); end
    end
end

% ── Summary ───────────────────────────────────────────────────────────────
fprintf('\n%s\n',repmat('═',1,75));
fprintf('Per-segment summary:\n');
fprintf('  %-14s  %2s  %6s  %6s  %6s  %5s  %5s\n','Segment','λ','L_tgt','l_mean','l_p95','B_avg','q_avg');
t0 = 1;
for si = 1:numel(SEGMENTS)
    te = t0 + SEGMENTS(si).ticks - 1;
    sl = log_l_meas(t0:te); sl = sl(sl>0);
    if ~isempty(sl)
        fprintf('  %-14s  %2d  %6d  %6.0f  %6.0f  %5.1f  %5.1f\n', ...
            SEGMENTS(si).label, SEGMENTS(si).lambda, SEGMENTS(si).L_target, ...
            mean(sl), prctile(sl,95), mean(log_B(t0:te)), mean(log_q_sw(t0:te)));
    end
    t0 = te + 1;
end

% ── Save + Plot ───────────────────────────────────────────────────────────
log = struct('tick',log_tick,'lambda',log_lambda,'L_target',log_L_target, ...
    'q_sw',log_q_sw,'q_ref',log_q_ref,'B',log_B,'l_meas',log_l_meas, ...
    'e_l',log_e_l,'e_q',log_e_q,'xi_l',log_xi_l,'xi_q',log_xi_q, ...
    'label',{log_label});
save(fullfile(out_dir,'run_log.mat'),'log','SEGMENTS');
fprintf('\n[save] run_log.mat\n');
plot_results(log, SEGMENTS, N, out_dir);


% =========================================================================
function m = server_get(server, path)
    [~,out] = system(sprintf('curl -s "%s%s"', server, path));
    m = jsondecode(strtrim(out));
end

function r = server_post(server, path, data)
    body = strrep(jsonencode(data), '"', '\"');
    [~,out] = system(sprintf('curl -s -X POST "%s%s" -H "Content-Type: application/json" -d "%s"', server, path, body));
    try; r = jsondecode(strtrim(out)); catch; r = struct(); end
end

function v = clamp(x, lo, hi); v = max(lo, min(hi, x)); end
function b = isnan_safe(x);    b = ~isnumeric(x) || isnan(x); end
function v = ifelse(cond, a, b); if cond; v=a; else; v=b; end; end

function plot_results(log, SEGMENTS, N, out_dir)
    COLORS = {[0.95 0.95 0.95],[1.00 0.90 0.90],[0.90 1.00 0.90], ...
              [0.90 0.90 1.00],[1.00 0.97 0.90],[1.00 0.90 1.00],[0.90 0.97 1.00]};
    fig = figure('Visible','off','Position',[50 50 1400 1000]);
    ax  = gobjects(4,1);
    for i=1:4; ax(i) = subplot(4,1,i); end

    t0 = 1;
    for si = 1:numel(SEGMENTS)
        te = t0 + SEGMENTS(si).ticks - 1;
        c  = COLORS{mod(si-1,numel(COLORS))+1};
        for ai=1:4
            patch(ax(ai),[t0 te te t0],[get(ax(ai),'YLim') fliplr(get(ax(ai),'YLim'))], ...
                c,'FaceAlpha',0.3,'EdgeColor','none'); hold(ax(ai),'on');
        end
        text(ax(1),(t0+te)/2, 0.97, SEGMENTS(si).label, ...
            'Units','data','HorizontalAlignment','center','VerticalAlignment','top', ...
            'FontSize',7.5,'Color',[0.2 0.2 0.2]);
        t0 = te + 1;
    end

    plot(ax(1), log.tick, log.l_meas/1000, 'b-', 'LineWidth',1.3, 'DisplayName','l_{total} [s]');
    stairs(ax(1), log.tick, log.L_target/1000, 'k--', 'LineWidth',1.5, 'DisplayName','L_{target}');
    ylabel(ax(1),'l_{total} [s]'); legend(ax(1),'Location','northeast');
    title(ax(1),'Chapter 6 — Cascade Controller  (Intel Mac, v3: recent window + prewarm + spike>Bmax)');
    grid(ax(1),'on');

    area(ax(2), log.tick, log.q_sw, 'FaceColor',[1 0.65 0],'FaceAlpha',0.4);
    hold(ax(2),'on');
    stairs(ax(2), log.tick, log.q_sw,  'Color',[0.85 0.45 0],'LineWidth',1.2,'DisplayName','q_{sw}');
    stairs(ax(2), log.tick, log.q_ref, 'g--','LineWidth',1.5,'DisplayName','q_{ref}');
    ylabel(ax(2),'Queue [req]'); legend(ax(2),'q_{sw}','q_{ref}','Location','northeast');
    grid(ax(2),'on');

    stairs(ax(3), log.tick, log.B,      'm-', 'LineWidth',1.5,'DisplayName','B');
    hold(ax(3),'on');
    stairs(ax(3), log.tick, log.lambda, 'k--','LineWidth',1.0,'DisplayName','\lambda');
    yline(ax(3), B_MAX, 'r:', 'LineWidth', 1.0, 'DisplayName', 'B_{max}');
    ylabel(ax(3),'Req/tick'); legend(ax(3),'Location','northeast');
    ylim(ax(3),[0 max(max(log.lambda),max(log.B))+1]);
    grid(ax(3),'on');

    yyaxis(ax(4),'left');
    plot(ax(4), log.tick, log.xi_l,'g-','LineWidth',1.2);
    yline(ax(4),0,'k--','LineWidth',0.7);
    ylabel(ax(4),'\xi_l','Color','g');
    yyaxis(ax(4),'right');
    plot(ax(4), log.tick, log.xi_q,'r--','LineWidth',1.0);
    ylabel(ax(4),'\xi_q','Color','r');
    xlabel(ax(4),'Tick [k]');
    legend(ax(4),'\xi_l','\xi_q','Location','northwest'); grid(ax(4),'on');

    for i=1:4; xlim(ax(i),[1 N]); end
    ts = datetime('now','Format','HHmmss');
    fn = fullfile(out_dir, sprintf('ch6_cascade_%s.png',char(ts)));
    saveas(fig,fn); fprintf('[plot] %s\n',fn);
end
