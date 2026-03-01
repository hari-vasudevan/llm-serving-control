function run_simulation(perturbed, controller)
%RUN_SIMULATION  Closed-loop discrete-time simulation of the LLM plant.
%
%   Simulates the nonlinear plant under integral state-feedback.
%   A traffic spike (2× arrival rate, t = 5..15 s) tests disturbance rejection.
%   Anti-windup clamps the integral state to prevent post-spike windup.
%
%   Inputs
%     perturbed   plant parameter structure
%     controller  structure from design_controller()

dt   = perturbed.dt;
B0   = perturbed.B0;
q0   = perturbed.q0;
K_q  = controller.K_q;
K_i  = controller.K_i;

% ── Time vector & arrival rate profile ───────────────────────────────────────
T = round(30 / dt);            % total steps  (30 s simulation)
t = (0:T-1)' * dt;

% Lambda: base rate, 2× spike from t=5..15 s, then back to base
lambda = perturbed.lambda_mean * ones(T, 1);
lambda(t >= 5 & t < 15) = perturbed.lambda_mean * 2;

% Anti-windup limits for xi (integral of Δq)
%   Clamp xi so ΔB contribution stays within ±(B_max - B0) / K_i
%   (avoid integrator saturating the actuator indefinitely post-spike)
xi_max =  (perturbed.B_max - B0) / max(abs(K_i), 1e-6);
xi_min = -(B0 - perturbed.B_min) / max(abs(K_i), 1e-6);

% ── Preallocate ───────────────────────────────────────────────────────────────
q      = zeros(T, 1);
B_act  = zeros(T, 1);       % actual (saturated) batch size
L_mean = zeros(T, 1);
L_p95  = zeros(T, 1);
xi     = zeros(T, 1);       % integral of Δq (augmented state)

% ── Initial conditions — start at equilibrium ─────────────────────────────────
q(1)    = q0;
xi(1)   = 0;
B_act(1)= B0;

% ── Closed-loop loop ──────────────────────────────────────────────────────────
for k = 1:T-1

    % 1. Plant output at current state
    [~, L_mean(k), L_p95(k)] = llm_plant(q(k), B_act(k), lambda(k), perturbed);

    % 2. Deviations from equilibrium
    dq = q(k) - q0;

    % 3. Control law: ΔB = -K_q·Δq - K_i·xi
    dB     = -K_q * dq  -  K_i * xi(k);
    B_raw  = B0 + dB;

    % 4. Actuator saturation
    B_act(k+1) = min(perturbed.B_max, max(perturbed.B_min, B_raw));

    % 5. Plant state update (nonlinear)
    q(k+1) = llm_plant(q(k), B_act(k+1), lambda(k), perturbed);

    % 6. Integral state update with anti-windup
    %    Only integrate when actuator is NOT saturated
    if B_act(k+1) == B_raw          % unsaturated — normal update
        xi(k+1) = xi(k) + dq;
    else                             % saturated — freeze integrator
        xi(k+1) = xi(k);
    end
    xi(k+1) = min(xi_max, max(xi_min, xi(k+1)));   % hard clamp

end

% Fill final-step outputs
[~, L_mean(T), L_p95(T)] = llm_plant(q(T), B_act(T), lambda(T), perturbed);

% ── Performance metrics ───────────────────────────────────────────────────────
spike_idx = t >= 5  & t < 15;
post_idx  = t >= 15;

fprintf('=== Simulation Results  [%s] ===\n', upper(controller.method));
fprintf('  p95 latency:  mean = %.1f ms,   max = %.1f ms\n', ...
        mean(L_p95), max(L_p95));
fprintf('  SLA soft violations  (L_p95 > %.0f ms): %.1f%%\n', ...
        perturbed.L_p95_target, 100*mean(L_p95 > perturbed.L_p95_target));
fprintf('  SLA hard violations  (L_p95 > 200  ms): %.1f%%\n', ...
        100*mean(L_p95 > 200));
fprintf('  Queue:   mean = %.2f req,   max = %.2f req\n', mean(q), max(q));
fprintf('  Batch:   mean = %.2f req,   min = %.2f,  max = %.2f\n', ...
        mean(B_act), min(B_act), max(B_act));
fprintf('  During spike  (t=5-15 s):  mean L_p95 = %.1f ms\n', ...
        mean(L_p95(spike_idx)));
fprintf('  Post-spike    (t>15 s):    mean L_p95 = %.1f ms\n\n', ...
        mean(L_p95(post_idx)));

% ── Plot ──────────────────────────────────────────────────────────────────────
fig = figure('Name', sprintf('LLM Serving — %s', upper(controller.method)), ...
             'Position', [100 80 1000 850]);

% Row 1: arrival rate
ax1 = subplot(4,1,1);
stairs(t, lambda, 'b', 'LineWidth', 1.5);
yline(perturbed.lambda_mean,   'g--', '\lambda_{base}',  'LabelHorizontalAlignment','left');
yline(perturbed.lambda_mean*2, 'r--', '\lambda_{spike}', 'LabelHorizontalAlignment','left');
ylabel('\lambda  (req/tick)');
title('Disturbance: Arrival Rate');
grid on; xlim([0 t(end)]);

% Row 2: queue depth
ax2 = subplot(4,1,2);
plot(t, q, 'b', 'LineWidth', 1.5); hold on;
yline(q0, 'r--', sprintf('q_0 = %.1f req', q0), 'LabelHorizontalAlignment','left');
xline(5,  'k:', 'LineWidth', 1.2);
xline(15, 'k:', 'LineWidth', 1.2);
ylabel('Queue depth  (req)');
title('State: Queue Depth  q[k]');
grid on; xlim([0 t(end)]);

% Row 3: latency
ax3 = subplot(4,1,3);
plot(t, L_p95,  'r',   'LineWidth', 1.5, 'DisplayName', 'L_{p95}'); hold on;
plot(t, L_mean, 'b--', 'LineWidth', 1.0, 'DisplayName', 'L_{mean}');
yline(perturbed.L_p95_target, 'k--', 'SLA target', 'LabelHorizontalAlignment','left');
yline(200,                    'k:',  'Hard SLA',   'LabelHorizontalAlignment','left');
xline(5,'k:','LineWidth',1.2); xline(15,'k:','LineWidth',1.2);
legend('Location','northeast');
ylabel('Latency  (ms)');
title('Output: p95 and Mean Latency');
grid on; xlim([0 t(end)]);

% Row 4: batch size
ax4 = subplot(4,1,4);
stairs(t, B_act, 'g', 'LineWidth', 1.5); hold on;
yline(B0,              'r--', sprintf('B_0 = %.0f', B0), 'LabelHorizontalAlignment','left');
yline(perturbed.B_max, 'k:',  'B_{max}', 'LabelHorizontalAlignment','left');
yline(perturbed.B_min, 'k:',  'B_{min}', 'LabelHorizontalAlignment','left');
xline(5,'k:','LineWidth',1.2); xline(15,'k:','LineWidth',1.2);
ylabel('Batch size  (req)');
xlabel('Time  (s)');
title('Control Input: Batch Size  B[k]');
grid on; xlim([0 t(end)]);

linkaxes([ax1 ax2 ax3 ax4], 'x');

sgtitle(sprintf('LLM Inference Serving — Integral State-Feedback Regulator  [%s]', ...
                upper(controller.method)), 'FontSize', 12, 'FontWeight', 'bold');

out_path = fullfile('/Users/hvasudevan/Documents/MATLAB/llm_control_v2', ...
                    sprintf('results_%s.png', controller.method));
saveas(fig, out_path);
fprintf('  Figure saved: %s\n', out_path);

end
