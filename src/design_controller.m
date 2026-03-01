function controller = design_controller(perturbed, method)
%DESIGN_CONTROLLER  Integral state-feedback controller for the LLM plant.
%
%   CONTROLLER = DESIGN_CONTROLLER(PERTURBED, METHOD)
%
%   Inputs
%     perturbed  plant parameter structure (from setup_plant.m)
%     method     'lqr'  or  'pole_placement'
%
%   Returns: controller structure with fields
%     .K_q        gain on Δq          (proportional on queue deviation)
%     .K_i        gain on xi          (integral of Δq)
%     .K_aug      [K_q, K_i]
%     .method     string
%     .A_aug, .B_aug   augmented matrices
%     .poles_ol, .poles_cl
%     .B0, .q0    equilibrium (for use in run_simulation)
%
% ── Formulation ───────────────────────────────────────────────────────────────
%
%  Deviations from equilibrium:
%    Δq[k] = q[k] - q0         (queue depth deviation)
%    ΔB[k] = B[k] - B0         (batch size deviation — control input)
%
%  Linearised plant:
%    Δq[k+1] = Δq[k] - ΔB[k]              A = 1,  B_m = -1
%
%  KEY DESIGN CHOICE — integrate Δq, not ΔL_p95:
%
%    q0 is defined so that q[k]=q0 ⟺ L_p95[k]=L_p95_target.
%    Therefore regulating q → q0 is equivalent to meeting the SLA.
%    Integrating Δq avoids the sign ambiguity that arises from
%    D = dL_p95/dB > 0 (increasing B directly raises service latency),
%    which can cause a naive latency-integrating controller to wind up
%    in the wrong direction.
%
%  Augmented state:  x_aug = [Δq ; xi]  where  xi = integral of Δq
%
%    xi[k+1] = xi[k] + Δq[k]
%
%  Augmented matrices:
%    A_aug = [ 1   0 ]     B_aug = [ -1 ]
%            [ 1   1 ]             [  0 ]
%
%  Control law (regulator):
%    ΔB[k] = -K_aug · x_aug[k] = -K_q·Δq[k] - K_i·xi[k]
%
%  Actual batch size (caller applies saturation):
%    B[k] = B0 + ΔB[k]
%
%  Intuition of signs:
%    When q > q0  (Δq > 0):  K_q > 0  →  ΔB = -K_q·Δq < 0  →  B decreases
%      Wait — shouldn't we increase B to drain the queue?
%    Answer: the pole of the plant is at z=1 (integrator). Feedback gain K_q
%    places the closed-loop pole at z = 1 - K_q. For K_q ∈ (0,2) the closed
%    loop is stable. LQR / pole placement will choose K_q s.t. the plant pole
%    moves inside the unit circle. Decreasing B when q is high lets the queue
%    self-regulate because the ARRIVAL side is the disturbance — with less B,
%    the queue drains at exactly lambda (queue equation). Actually K_q<0 means
%    increase B when queue is high, which is the physically correct direction.
%    LQR will determine the correct sign automatically.

% ── Unpack ────────────────────────────────────────────────────────────────────
B0 = perturbed.B0;
q0 = perturbed.q0;

% ── Linearised plant matrices (scalar SISO) ───────────────────────────────────
A   =  1;     % queue integrator pole at z=1
B_m = -1;     % batch size drains queue

% ── Augmented system: state = [Δq ; xi=∫Δq] ──────────────────────────────────
%
%   A_aug = [A,  0] = [1, 0]     B_aug = [B_m] = [-1]
%           [1,  1]   [1, 1]             [ 0 ]   [ 0]
%
A_aug = [A,   0;
         1,   1];

B_aug = [B_m;
         0  ];

% Controllability check
Co = ctrb(A_aug, B_aug);
fprintf('=== Linearised plant: A=%g, B=%g ===\n', A, B_m);
fprintf('  Augmented system controllable: %s\n', ...
        mat2str(rank(Co) == size(A_aug,1)));

poles_ol = eig(A_aug);
fprintf('  Open-loop augmented poles: [%.4f, %.4f]\n', poles_ol(1), poles_ol(2));
fprintf('  (z=1 twice: plant integrator + queue integral augmentation)\n\n');

% ── Controller design ─────────────────────────────────────────────────────────
switch lower(method)

    % ── LQR ──────────────────────────────────────────────────────────────────
    case 'lqr'
        % Augmented state cost Q = diag([q_weight, xi_weight])
        %   q_weight  large → penalise queue deviations hard  (fast rejection)
        %   xi_weight       → penalise accumulated queue error (zero steady-state)
        % R : penalise batch-size changes (higher → smoother, more conservative)
        Q = diag([5, 0.5]);
        R = 1;

        K_aug = dlqr(A_aug, B_aug, Q, R);

        fprintf('=== LQR design ===\n');
        fprintf('  Q = diag([%.1f, %.1f]),  R = %.1f\n', Q(1,1), Q(2,2), R);

    % ── Pole placement ────────────────────────────────────────────────────────
    case 'pole_placement'
        % Desired closed-loop poles in discrete time.
        % Continuous-time bandwidth ωc (rad/s) ↔ discrete pole z = exp(-ωc·dt)
        %
        % With dt = 0.1 s:
        %   z = 0.75 ↔ ωc ≈ 2.9 rad/s  (fast — handles queue transients)
        %   z = 0.90 ↔ ωc ≈ 1.1 rad/s  (slower integral pole)
        %
        % Poles must be distinct for place() (use acker() for repeated poles).
        p_desired = [0.75, 0.90];

        K_aug = place(A_aug, B_aug, p_desired);

        fprintf('=== Pole placement design ===\n');
        fprintf('  Desired poles: z = [%.2f, %.2f]\n', p_desired(1), p_desired(2));

    otherwise
        error('design_controller: method must be ''lqr'' or ''pole_placement''');
end

% ── Closed-loop analysis ──────────────────────────────────────────────────────
A_cl     = A_aug - B_aug * K_aug;
poles_cl = eig(A_cl);

fprintf('  K_aug  = [K_q = %.4f,  K_i = %.4f]\n', K_aug(1), K_aug(2));
fprintf('  Closed-loop poles: [%.4f, %.4f]\n', poles_cl(1), poles_cl(2));
fprintf('  Stable (all |z| < 1): %s\n\n', mat2str(all(abs(poles_cl) < 1)));

% ── Pack output structure ─────────────────────────────────────────────────────
controller.K_q      = K_aug(1);
controller.K_i      = K_aug(2);
controller.K_aug    = K_aug;
controller.method   = method;
controller.A        = A;
controller.B_m      = B_m;
controller.A_aug    = A_aug;
controller.B_aug    = B_aug;
controller.poles_ol = poles_ol;
controller.poles_cl = poles_cl;
controller.B0       = B0;
controller.q0       = q0;

end
