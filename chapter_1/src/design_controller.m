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
%     .K_q        gain on Dq          (proportional on queue deviation)
%     .K_i        gain on xi          (integral of error)
%     .K_aug      [K_q, K_i]
%     .method     string
%     .A_aug, .B_aug   augmented matrices
%     .poles_ol, .poles_cl
%     .B0, .q0    equilibrium (for use in run_simulation)
%
% -- Formulation ---------------------------------------------------------------
%
%  Deviations from equilibrium:
%    Dq[k] = q[k] - q0         (queue depth deviation)
%    DB[k] = B[k] - B0         (batch size deviation -- control input)
%
%  Linearised plant:
%    Dq[k+1] = Dq[k] - DB[k]              A = 1,  B_m = -1
%
%  KEY DESIGN CHOICE -- integrate error e = q0-q, not DL_p95:
%
%    q0 is defined so that q[k]=q0 <=> L_p95[k]=L_p95_target.
%    Therefore regulating q -> q0 is equivalent to meeting the SLA.
%    Integrating Dq avoids the sign ambiguity that arises from
%    D = dL_p95/dB > 0 (increasing B directly raises service latency),
%    which can cause a naive latency-integrating controller to wind up
%    in the wrong direction.
%
%  Augmented state:  x_aug = [Dq ; xi]  where  xi = integral of error e
%
%    e[k]  = q0 - q[k]           (Franklin 8.5.1: error = r - y)
%    xi[k+1] = xi[k] + e[k]      (pure accumulation, no dt)
%            = xi[k] - Dq[k]     (since e = -Dq)
%
%  Augmented matrices (Franklin 8.5.1 -- derives directly from e = -C*Dx):
%    A_aug = [ 1   0 ]     B_aug = [ -1 ]
%            [-1   1 ]             [  0 ]
%
%  Control law (regulator):
%    DB[k] = -K_aug * x_aug[k] = -K_q*Dq[k] - K_i*xi[k]
%
%  Actual batch size (caller applies saturation):
%    B[k] = B0 + DB[k]
%
% -- Pole placement parameterisation ------------------------------------------
%
%  f = 0  (non-oscillatory): specify two independent time constants
%    perturbed.pp_tau1, perturbed.pp_tau2
%    -> z_i = exp(-dt / tau_i)   (two distinct real poles)
%
%  f > 0  (oscillatory): specify one time constant + one damped frequency
%    perturbed.pp_tau, perturbed.pp_f  [s, Hz]
%    -> s = -1/tau +/- j*2*pi*f   then   z = exp(s*dt)
%    (complex conjugate pair; tau sets envelope decay, f sets ring frequency)

% -- Unpack -------------------------------------------------------------------
B0 = perturbed.B0;
q0 = perturbed.q0;

% -- Linearised plant matrices (scalar SISO) ----------------------------------
A   =  1;     % queue integrator pole at z=1
B_m = -1;     % batch size drains queue
C =  1;
D = 0;

% -- Augmented system: state = [Dq ; xi=integral of e] -----------------------
A_aug = [A,   0;
         C,   1];
B_aug = [B_m;
         D  ];

% Controllability check
Co = ctrb(A_aug, B_aug);
fprintf('=== Linearised plant: A=%g, B=%g ===\n', A, B_m);
fprintf('  Augmented system controllable: %s\n', ...
        mat2str(rank(Co) == size(A_aug,1)));
poles_ol = eig(A_aug);
fprintf('  Open-loop augmented poles: [%.4f, %.4f]\n', poles_ol(1), poles_ol(2));
fprintf('  (z=1 twice: plant integrator + queue integral augmentation)\n\n');

% -- Controller design --------------------------------------------------------
switch lower(method)

    % -- LQR ------------------------------------------------------------------
    case 'lqr'
        % Augmented state cost Q = diag([q_weight, xi_weight])
        %   q_weight  large -> penalise queue deviations hard  (fast rejection)
        %   xi_weight       -> penalise accumulated error       (zero ss error)
        % R : penalise batch-size changes (higher -> smoother, more conservative)
        Q = diag([5, 0.5]);
        R = 1;
        K_aug = dlqr(A_aug, B_aug, Q, R);
        fprintf('=== LQR design ===\n');
        fprintf('  Q = diag([%.1f, %.1f]),  R = %.1f\n', Q(1,1), Q(2,2), R);

    % -- Pole placement -------------------------------------------------------
    case 'pole_placement'
        f  = perturbed.pp_f;
        dt = perturbed.dt;

        if f == 0
            % Non-oscillatory: two independent real poles.
            % Each pole specified by its own time constant tau_i.
            %   z_i = exp(-dt / tau_i)
            tau1 = perturbed.pp_tau1;
            tau2 = perturbed.pp_tau2;
            z1 = exp(-dt / tau1);
            z2 = exp(-dt / tau2);
            p_desired = [z1, z2];
            fprintf('=== Pole placement design ===\n');
            fprintf('  Mode: non-oscillatory\n');
            fprintf('  Pole 1:  tau = %.3f s  ->  s = %.4f,  z = %.4f\n', tau1, -1/tau1, z1);
            fprintf('  Pole 2:  tau = %.3f s  ->  s = %.4f,  z = %.4f\n', tau2, -1/tau2, z2);
        else
            % Oscillatory: complex conjugate pair.
            % tau  -> envelope decay  (real part of s = -1/tau)
            % f    -> damped ring frequency [Hz]
            %   s = -1/tau +/- j*2*pi*f
            %   z = exp(s * dt)
            tau   = perturbed.pp_tau;
            sigma = 1 / tau;
            wd    = 2 * pi * f;
            s_pole = -sigma + 1j * wd;
            z1 = exp(s_pole * dt);
            p_desired = [z1, conj(z1)];
            fprintf('=== Pole placement design ===\n');
            fprintf('  Mode: oscillatory   tau = %.3f s,  f = %.4f Hz\n', tau, f);
            fprintf('  Continuous poles: s = %.4f +/- j*%.4f\n', -sigma, wd);
            fprintf('  Discrete poles:   z = %.4f +/- j*%.4f  (|z| = %.4f)\n', ...
                    real(z1), imag(z1), abs(z1));
        end
        K_aug = acker(A_aug, B_aug, p_desired);

    otherwise
        error('design_controller: method must be ''lqr'' or ''pole_placement''');
end

% -- Closed-loop analysis -----------------------------------------------------
A_cl     = A_aug - B_aug * K_aug;
poles_cl = eig(A_cl);
fprintf('  K_aug  = [K_q = %.4f,  K_i = %.4f]\n', K_aug(1), K_aug(2));
fprintf('  Closed-loop poles: [%.4f, %.4f]\n', poles_cl(1), poles_cl(2));
fprintf('  Stable (all |z| < 1): %s\n\n', mat2str(all(abs(poles_cl) < 1)));

% -- Pack output structure ----------------------------------------------------
% NOTE: only real numeric scalars/arrays are included here.
% Simulink MATLAB Function blocks reject structs containing strings,
% complex arrays, or cell arrays when used as block parameters.
controller.K_q      = K_aug(1);        % scalar double
controller.K_i      = K_aug(2);        % scalar double
controller.K_aug    = K_aug;           % 1x2 double
controller.A        = A;               % scalar double
controller.B_m      = B_m;             % scalar double
controller.A_aug    = A_aug;           % 2x2 double
controller.B_aug    = B_aug;           % 2x1 double
controller.poles_ol = real(poles_ol);  % 2x1 double (real parts only)
controller.poles_cl = real(poles_cl);  % 2x1 double (real parts only)
controller.B0       = B0;              % scalar double
controller.q0       = q0;              % scalar double
controller.q_max    = perturbed.q_max; % scalar double
% NOTE: method string is intentionally NOT stored in the struct so that
% the struct remains Simulink-compatible (all-numeric fields only).
% run_simulation receives the method string as a separate argument.
controller.method_id = double(lower(method(1)) == 'l');  % 1=lqr, 0=pole_placement
end
