%% run_controller.m  --  Chapter 6: Single-loop controller (TTFT measurement)
%
% ROOT CAUSE OF PREVIOUS FAILURES
% --------------------------------
% 1. WRONG SIGNAL: l_total = queue_wait + TTFT.  When the queue builds,
%    l_total grows without bound.  The controller responds by reducing B,
%    which makes queue_wait grow even faster (fewer dispatched per tick).
%    This is POSITIVE FEEDBACK -- the controller destabilises the system.
%
%    FIX: use ttft_recent_mean (dispatch-to-completion, last 10 requests).
%    TTFT only reflects B's effect on Ollama concurrency.  B larger -> TTFT
%    larger.  This is the stable, monotone relationship the controller needs.
%
% 2. OVERLOADED SYSTEM: lambda_ss=4, B_max=4, TTFT(B=4)=1350ms > dt=1000ms.
%    Ollama takes 1.35 ticks to serve each batch.  Queue grows by ~1/tick
%    from tick 1 regardless of controller.
%
%    FIX: lambda_ss=2, B_max=4.  At equilibrium lambda=2 < B=2-4, so Ollama
%    always keeps up.  The spike (lambda=6) briefly overloads, but once
%    lambda returns to 2 the queue drains.
%
% CONTROLLER LAW (same as before, but cleaner signal):
%   e[k]  = L_target - ttft_meas[k]
%   xi    = clamp(xi + e, xi_min, xi_max)
%   B[k]  = clamp(B0 + K_il * xi, B_min, B_max)
%   K_il > 0: TTFT > target -> xi down -> B down -> TTFT down  CORRECT

clear; clc;
addpath(fileparts(mfilename('fullpath')));

out_dir = fileparts(mfilename('fullpath'));
load(fullfile(out_dir, 'controller_params.mat'));

fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Chapter 6: Single-Loop Controller (TTFT signal)             ║\n');
fprintf('╠══════════════════════════════════════════════════════════════╣\n');
fprintf('║  Measures TTFT (dispatch→done), NOT l_total (enqueue→done)   ║\n');
fprintf('║  K_il=%.8f  z_cl=%.4f  tau_cl=%.0fs\n', K_il, z_cl, TAU_CL);
fprintf('║  B0=%d  B=[%d,%d]  beta_eff=%.1f ms/req\n', B0,B_MIN,B_MAX,beta_eff);
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ── Schedule ──────────────────────────────────────────────────────────────
% TTFT at natural operating point B=2: ~887ms
% L_target = 1.5 × TTFT(B=2) gives room on both sides
% lambda_ss = 2 (< B_max=4, system stays stable at equilibrium)
% lambda_spike = 6 (> B_max=4, creates overload; controller can't drain
%                   queue but reduces TTFT per-request via lowering B)

L_TTFT_SS    = 1400;   % target TTFT at steady state [ms] (~1.5x TTFT(B=2))
L_TTFT_TIGHT = 700;    % tighter TTFT target [ms] (~below TTFT(B=2))
LAMBDA_SS    = 2;      % FIX: was 4, now 2 -- system can keep up
LAMBDA_SPIKE = 6;      % > B_max=4, creates queue pressure
LAMBDA_DROP  = 1;

% Override with identified values if they exist and are sensible
if exist('L_target_recommended','var') && L_target_recommended > 0
    % L_target_recommended was computed from l_total; scale to TTFT-only
    % TTFT is roughly 50-60% of l_total at moderate queue depths
    L_TTFT_SS    = round(L_target_recommended * 0.55 / 100) * 100;
    L_TTFT_TIGHT = round(L_TTFT_SS * 0.5 / 100) * 100;
end
% Keep lambda fixed at sensible values regardless of characterise output
% lambda_ss=2 is the key safety margin

SEGMENTS = struct(...
    'ticks',    {30,           60,            30,         20,           20,        30,              20}, ...
    'lambda',   {LAMBDA_SS,    LAMBDA_SPIKE,  LAMBDA_SS,  LAMBDA_DROP,  LAMBDA_SS, LAMBDA_SS,       LAMBDA_SS}, ...
    'L_target', {L_TTFT_SS,    L_TTFT_SS,     L_TTFT_SS,  L_TTFT_SS,    L_TTFT_SS, L_TTFT_TIGHT,    L_TTFT_SS}, ...
    'label',    {'Steady',     'λ↑ Spike',    'Recovery', 'λ↓ Drop',    'Recovery','TTFT Target↓',  'Restore'} ...
);

N = sum([SEGMENTS.ticks]);
fprintf('Schedule: %d ticks  dt=%.0fs (~%.0f min)\n', N, DT, N*DT/60);
fprintf('  TTFT targets: steady=%dms  tight=%dms\n', L_TTFT_SS, L_TTFT_TIGHT);
fprintf('  lambda_ss=%d (< B_max=%d -- system stays stable)\n\n', LAMBDA_SS, B_MAX);
for s=1:numel(SEGMENTS)
    fprintf('  %3dt  λ=%-2d  L_TTFT=%4dms  [%s]\n', ...
        SEGMENTS(s).ticks, SEGMENTS(s).lambda, SEGMENTS(s).L_target, SEGMENTS(s).label);
end

% ── Capacity sanity check before starting ────────────────────────────────
fprintf('\n[check] Verifying system can sustain lambda_ss=%d...\n', LAMBDA_SS);
ttft_at_B_max = alpha*B_MAX + gamma*B_MAX^2;
throughput_at_Bmax = B_MAX / (ttft_at_B_max/1000);  % req/s
fprintf('  TTFT(B_max=%d) = %.0fms  throughput ≈ %.1f req/s\n', ...
    B_MAX, ttft_at_B_max, throughput_at_Bmax);
if LAMBDA_SS / DT > throughput_at_Bmax
    fprintf('  WARNING: lambda_ss=%.0f req/s > max_throughput=%.1f req/s\n', ...
        LAMBDA_SS/DT, throughput_at_Bmax);
    fprintf('  System will be overloaded. Reduce lambda_ss or increase B_max.\n');
else
    fprintf('  OK: lambda_ss=%.0f req/s < max_throughput=%.1f req/s\n', ...
        LAMBDA_SS/DT, throughput_at_Bmax);
end

% ── Pre-warm ──────────────────────────────────────────────────────────────
fprintf('\n[pre-warm] 12 requests to fill ttft_recent buffer...\n');
server_post(SERVER, '/reset', struct());
server_post(SERVER, '/control', struct('B', B0));
PROMPTS = {'What is 2+2?','Name a colour.','Capital of France?', ...
           'Days in a week?','Name a planet.','Speed of light?', ...
           'Name a mammal.','10 times 10?','Colour of sky?', ...
           'Name a fruit.','Hours in a day?','5 squared?'};
for i=1:12
    server_post(SERVER, '/enqueue', struct('prompt', PROMPTS{mod(i-1,12)+1}));
end
t_pw = tic;
while toc(t_pw) < 120
    m = server_get(SERVER, '/metrics');
    if m.completed >= 12; break; end
    pause(1);
end
m = server_get(SERVER, '/metrics');
ttft_now = get_ttft(m);
fprintf('  Done: completed=%d  ttft_recent=%.0fms\n', m.completed, ttft_now);

% Reset ring buffers; Ollama stays hot
server_post(SERVER, '/reset', struct()); pause(0.5);
server_post(SERVER, '/control', struct('B', B0));
% Fire 4 requests so buffer has fresh samples from tick 1
for i=1:4; server_post(SERVER, '/enqueue', struct('prompt', PROMPTS{i})); end
pause(DT*3);
fprintf('[pre-warm] Complete.\n\n');

% ── Controller state ──────────────────────────────────────────────────────
xi = 0;

% ── Log ───────────────────────────────────────────────────────────────────
log_tick=zeros(N,1); log_lambda=zeros(N,1); log_L_target=zeros(N,1);
log_q_sw=zeros(N,1); log_B=zeros(N,1); log_ttft=zeros(N,1);
log_e=zeros(N,1); log_xi=zeros(N,1); log_label=cell(N,1);

server_post(SERVER, '/reset', struct());
server_post(SERVER, '/control', struct('B', B0));
prompt_idx=1; pause(1);

fprintf('%5s  %3s  %6s  %5s  %2s  %7s  %7s  %7s  label\n', ...
    'tick','λ','L_ttft','q_sw','B','ttft','e','xi');
fprintf('%s\n', repmat('-',1,72));

tick=0;
for si=1:numel(SEGMENTS)
    seg=SEGMENTS(si);
    for st=1:seg.ticks
        tick=tick+1; lam=seg.lambda; L_tgt=seg.L_target; t0=tic;

        % 1. Observe TTFT (not l_total)
        m    = server_get(SERVER, '/metrics');
        q_sw = m.q_sw;
        ttft = get_ttft_or_fallback(m, L_tgt);

        % 2. Arrivals
        a_k = poissrnd(lam);
        for ai=1:a_k
            server_post(SERVER, '/enqueue', ...
                struct('prompt', PROMPTS{mod(prompt_idx-1,12)+1}));
            prompt_idx=prompt_idx+1;
        end

        % 3. Control on TTFT
        e      = L_tgt - ttft;
        xi_sat = clamp(xi, xi_min, xi_max);
        B_cmd  = round(clamp(B0 + K_il*xi_sat, B_MIN, B_MAX));

        at_min = (B_cmd<=B_MIN) && (e<0);
        at_max = (B_cmd>=B_MAX) && (e>0);
        if ~(at_min||at_max)
            xi = clamp(xi_sat+e, xi_min, xi_max);
        else
            xi = xi_sat;
        end

        % 4. Actuate
        server_post(SERVER, '/control', struct('B', B_cmd));

        % 5. Log
        log_tick(tick)=tick; log_lambda(tick)=lam; log_L_target(tick)=L_tgt;
        log_q_sw(tick)=q_sw; log_B(tick)=B_cmd; log_ttft(tick)=ttft;
        log_e(tick)=e; log_xi(tick)=xi; log_label{tick}=seg.label;

        fprintf('%5d  %3d  %6d  %5d  %2d  %7.0f  %7.0f  %7.1f  %s\n', ...
            tick,lam,L_tgt,q_sw,B_cmd,ttft,e,xi,seg.label);

        elapsed=toc(t0); if elapsed<DT; pause(DT-elapsed); end
    end
end

% ── Summary ───────────────────────────────────────────────────────────────
fprintf('\n%s\n',repmat('═',1,72));
fprintf('  %-14s  %2s  %6s  %6s  %6s  %4s  %5s\n', ...
    'Segment','λ','L_ttft','t_mean','t_p95','B','q');
t0=1;
for si=1:numel(SEGMENTS)
    te=t0+SEGMENTS(si).ticks-1;
    sl=log_ttft(t0:te); sl=sl(sl>0);
    if ~isempty(sl)
        fprintf('  %-14s  %2d  %6d  %6.0f  %6.0f  %4.1f  %5.1f\n', ...
            SEGMENTS(si).label, SEGMENTS(si).lambda, SEGMENTS(si).L_target, ...
            mean(sl), prctile(sl,95), mean(log_B(t0:te)), mean(log_q_sw(t0:te)));
    end
    t0=te+1;
end

log=struct('tick',log_tick,'lambda',log_lambda,'L_target',log_L_target,...
    'q_sw',log_q_sw,'B',log_B,'ttft',log_ttft,'e',log_e,'xi',log_xi,...
    'label',{log_label});
save(fullfile(out_dir,'run_log.mat'),'log','SEGMENTS');
fprintf('\n[save] run_log.mat\n');
plot_results(log,SEGMENTS,N,out_dir);


% =========================================================================
function m = server_get(server,path)
    [~,out]=system(sprintf('curl -s "%s%s"',server,path));
    m=jsondecode(strtrim(out));
end
function r = server_post(server,path,data)
    body=strrep(jsonencode(data),'"','\"');
    [~,out]=system(sprintf('curl -s -X POST "%s%s" -H "Content-Type: application/json" -d "%s"',server,path,body));
    try; r=jsondecode(strtrim(out)); catch; r=struct(); end
end
function v = clamp(x,lo,hi); v=max(lo,min(hi,x)); end
function t = get_ttft(m)
    % jsondecode returns [] for JSON null -- must check isscalar
    % before isnan/>, otherwise && gets non-scalar logical operand
    v = []; % temp
    if isfield(m,'ttft_recent_mean'); v = m.ttft_recent_mean; end
    if isnumeric(v) && isscalar(v) && ~isnan(v) && v > 0
        t = double(v); return;
    end
    v = [];
    if isfield(m,'ttft_mean'); v = m.ttft_mean; end
    if isnumeric(v) && isscalar(v) && ~isnan(v) && v > 0
        t = double(v); return;
    end
    t = NaN;
end
function t = get_ttft_or_fallback(m,fb)
    t=get_ttft(m); if isnan(t)||t<=0; t=fb; end
end

function plot_results(log,SEGMENTS,N,out_dir)
    COLORS={[0.95 0.95 0.95],[1.0 0.90 0.90],[0.90 1.0 0.90],...
            [0.90 0.90 1.0],[1.0 0.97 0.90],[1.0 0.90 1.0],[0.90 0.97 1.0]};
    fig=figure('Visible','off','Position',[50 50 1400 950]);
    ax=gobjects(3,1); for i=1:3; ax(i)=subplot(3,1,i); end

    t0=1;
    for si=1:numel(SEGMENTS)
        te=t0+SEGMENTS(si).ticks-1; c=COLORS{mod(si-1,numel(COLORS))+1};
        for ai=1:3
            yl=ylim(ax(ai));
            patch(ax(ai),[t0 te te t0],[yl(1) yl(1) yl(2) yl(2)],...
                c,'FaceAlpha',0.28,'EdgeColor','none'); hold(ax(ai),'on');
        end
        text(ax(1),(t0+te)/2,0.96,SEGMENTS(si).label,...
            'Units','data','HorizontalAlignment','center',...
            'VerticalAlignment','top','FontSize',8,'Color',[0.2 0.2 0.2]);
        t0=te+1;
    end

    % P1: TTFT
    plot(ax(1),log.tick,log.ttft/1000,'b-','LineWidth',1.3,...
        'DisplayName','TTFT_{recent} [s]');
    hold(ax(1),'on');
    stairs(ax(1),log.tick,log.L_target/1000,'k--','LineWidth',1.5,...
        'DisplayName','L_{TTFT target}');
    ylabel(ax(1),'TTFT [s]');
    legend(ax(1),'Location','northeast'); grid(ax(1),'on');
    title(ax(1),'Chapter 6 — Single-loop controller on TTFT  (Intel Mac)');

    % P2: B, lambda, queue
    yyaxis(ax(2),'left');
    stairs(ax(2),log.tick,log.B,'m-','LineWidth',1.8,...
        'DisplayName','B (dispatch)'); hold(ax(2),'on');
    stairs(ax(2),log.tick,log.lambda,'k--','LineWidth',1.0,...
        'DisplayName','\lambda');
    ylabel(ax(2),'B / \lambda [req/tick]','Color','m');
    ylim(ax(2),[0 max(max(log.lambda),max(log.B))+1]);
    yyaxis(ax(2),'right');
    area(ax(2),log.tick,log.q_sw,'FaceColor',[1 0.65 0],'FaceAlpha',0.4,...
        'EdgeColor','none');
    hold(ax(2),'on');
    stairs(ax(2),log.tick,log.q_sw,'Color',[0.8 0.4 0],'LineWidth',1.0,...
        'DisplayName','q_{sw}');
    ylabel(ax(2),'q_{sw} [req]','Color',[0.8 0.4 0]);
    legend(ax(2),'B','\lambda','q_{sw}','Location','northeast');
    grid(ax(2),'on');

    % P3: xi and e
    yyaxis(ax(3),'left');
    plot(ax(3),log.tick,log.xi,'g-','LineWidth',1.2);
    yline(ax(3),0,'k--','LineWidth',0.7);
    ylabel(ax(3),'\xi (integrator)','Color','g');
    yyaxis(ax(3),'right');
    plot(ax(3),log.tick,log.e/1000,'b:','LineWidth',1.0);
    yline(ax(3),0,'k--','LineWidth',0.5);
    ylabel(ax(3),'e = L_{TTFT} - ttft [s]','Color','b');
    xlabel(ax(3),'Tick [k]');
    legend(ax(3),'\xi','e','Location','northeast'); grid(ax(3),'on');

    for i=1:3; xlim(ax(i),[1 N]); end
    ts=datetime('now','Format','HHmmss');
    fn=fullfile(out_dir,sprintf('ch6_single_ttft_%s.png',char(ts)));
    saveas(fig,fn); fprintf('[plot] %s\n',fn); close(fig);
end
