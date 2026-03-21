function controller = design_controller(perturbed, method)
%DESIGN_CONTROLLER  Cascade (inner + outer) integral state-feedback controller.
%
%   CONTROLLER = DESIGN_CONTROLLER(PERTURBED, METHOD)
%
%   Returns a nested structure:
%     controller.inner_c   — inner loop:  B -> q      (queue regulator)
%     controller.outer_c   — outer loop:  q_ref -> l_p95  (latency regulator)
%
% ═══════════════════════════════════════════════════════════════════════════════
% INNER LOOP  —  B -> q
% ═══════════════════════════════════════════════════════════════════════════════
%
%   Deviation variables:
%     dq[k] = q[k] - q0        (queue deviation)
%     dB[k] = B[k] - B0        (batch size deviation — actuator)
%
%   Linearised plant (Jacobian at equilibrium):
%     dq[k+1] = 1*dq[k] + (-1)*dB[k]     ->  A_in = 1,  B_in = -1
%
%   Error and integrator (Franklin 8.5.1):
%     e_q[k]    = q0 - q[k]  =  -dq[k]
%     xi_q[k+1] = xi_q[k] + e_q[k]       (C row = -1 in augmentation)
%
%   Augmented matrices (Franklin form):
%     A_aug_in = [1   0]    B_aug_in = [-1]
%                [-1  1]               [ 0]
%
%   Control law:
%     dB[k] = -K_aug_in * [dq[k]; xi_q[k]]
%     B[k]  =  B0 + dB[k],  clamped to [B_min, B_max]
%
% ═══════════════════════════════════════════════════════════════════════════════
% OUTER LOOP  —  q_ref -> l_p95
% ═══════════════════════════════════════════════════════════════════════════════
%
%   Simplifying assumption:  q[k] = q_ref[k]  (inner loop tracks perfectly).
%
%   Under this assumption the outer loop plant collapses to a static gain:
%     Δl_p95[k] = beta * Δq_ref[k]
%
%   There are no plant dynamics to stabilise — the only dynamics come from
%   the integrator we add for zero steady-state error.
%
%   State:   xi_l[k]   — integral of latency error
%   Error:   e_l[k]  = L_p95_target - l_p95_meas[k]
%   Update:  xi_l[k+1] = xi_l[k] + e_l[k]
%
%   Control law (integral-only):
%     dq_ref[k] = -K_il * xi_l[k]
%     q_ref[k]  =  q0 + dq_ref[k],  clamped to [0, q_max]
%
%   Closed-loop pole derivation:
%     Substituting e_l = -Δl_p95 = -beta*dq_ref = beta*K_il*xi_l:
%
%       xi_l[k+1] = xi_l[k] + e_l[k]
%                 = xi_l[k] * (1 + beta * K_il)
%
%     CL pole:  z_cl = 1 + beta * K_il
%
%     Stability:  |z_cl| < 1  ->  K_il in (-1/beta, 0)
%                                 K_il in (-0.5,  0)  for beta=2
%
%   Gain from desired time constant tau_out:
%     z_cl = exp(-dt / tau_out)
%     K_il = (z_cl - 1) / beta
%            = (exp(-dt/tau_out) - 1) / beta
%
%   Anti-windup limits (xi domain):
%     xi_l_max =  (q_max - q0) / |K_il|
%     xi_l_min = -(q0 - 0)     / |K_il|

% -- Unpack -------------------------------------------------------------------
B0    = perturbed.B0;
q0    = perturbed.q0;
dt    = perturbed.dt;
beta  = perturbed.beta;

% =============================================================================
% INNER LOOP DESIGN  (unchanged from Chapter 1)
% =============================================================================

A_in  =  1;
B_in  = -1;
C_in  =  1;
D_in  =  0;

A_aug_in = [A_in,  0;
             C_in,  1];
B_aug_in = [B_in;
             D_in];

Co_in = ctrb(A_aug_in, B_aug_in);
fprintf('=== INNER LOOP (B -> q) ===\n');
fprintf('  Controllable: %s\n', mat2str(rank(Co_in) == size(A_aug_in, 1)));
poles_ol_in = eig(A_aug_in);
fprintf('  OL poles: [%.4f, %.4f]  (double integrator)\n\n', ...
        poles_ol_in(1), poles_ol_in(2));

switch lower(method)

    case 'lqr'
        Q_in     = diag([5, 0.5]);
        R_in     = 1;
        K_aug_in = dlqr(A_aug_in, B_aug_in, Q_in, R_in);
        fprintf('  LQR: Q = diag([%.1f %.1f]),  R = %.1f\n', ...
                Q_in(1,1), Q_in(2,2), R_in);

    case 'pole_placement'
        f = perturbed.pp_f;
        if f == 0
            z1_in = exp(-dt / perturbed.pp_tau1);
            z2_in = exp(-dt / perturbed.pp_tau2);
        else
            s_p   = -1/perturbed.pp_tau + 1j*2*pi*f;
            z1_in = exp(s_p * dt);
            z2_in = conj(z1_in);
        end
        K_aug_in = acker(A_aug_in, B_aug_in, [z1_in, z2_in]);
        fprintf('  Pole placement: z = [%.4f, %.4f]\n', real(z1_in), real(z2_in));

    otherwise
        error('design_controller: method must be ''lqr'' or ''pole_placement''');
end

A_cl_in     = A_aug_in - B_aug_in * K_aug_in;
poles_cl_in = eig(A_cl_in);
fprintf('  K_aug_in = [K_q = %.4f,  K_i_q = %.4f]\n', K_aug_in(1), K_aug_in(2));
fprintf('  CL poles: [%.4f, %.4f],  stable: %s\n\n', ...
        real(poles_cl_in(1)), real(poles_cl_in(2)), ...
        mat2str(all(abs(poles_cl_in) < 1)));

K_i_in   = K_aug_in(2);
xi_q_max =  (perturbed.B_max - B0) / max(abs(K_i_in), 1e-6);
xi_q_min = -(B0 - perturbed.B_min) / max(abs(K_i_in), 1e-6);

% =============================================================================
% OUTER LOOP DESIGN  (augmented integral control, null-state static plant)
% =============================================================================
%
% Assumption:
%   Inner loop is sufficiently fast, so q[k] = q_ref[k].
%
% Outer plant in deviation variables:
%   Dl[k] = beta * Dq_ref[k]
%
% This is a null-state (memoryless) plant:
%   x[k+1] = [] x[k] + [] u[k]
%   y[k]   = beta * u[k]
%
% To use standard augmented integral design (acker / dlqr), we augment only
% with the controller integrator state xi_l:
%
%   xi_l[k+1] = xi_l[k] + e_l[k]
%             = xi_l[k] + Dl_ref[k] - Dl[k]
%
% Since Dl[k] = beta * Dq_ref[k], the augmented open-loop model is:
%
%   xi_l[k+1] = 1 * xi_l[k] + (-beta) * Dq_ref[k] + 1 * Dl_ref[k]
%
% so the standard state-space matrices for control design are:
%   A_aug_out = 1
%   B_aug_out = -beta
%
% and the control law is:
%   Dq_ref[k] = -K_aug_out * xi_l[k]
%
% giving closed-loop pole:
%   z = 1 + beta*K_aug_out
%
% Stability requires:
%   -2/beta < K_aug_out < 0

A_aug_out = 1;
B_aug_out = -beta;

Co_out = ctrb(A_aug_out, B_aug_out);
fprintf('=== OUTER LOOP (q_ref -> l_mean / l_p95 surrogate) ===\n');
fprintf('  Assumption: q[k] = q_ref[k]  (perfect inner tracking)\n');
fprintf('  Plant:      Dl = beta * Dq_ref  (static gain = %.4f ms/req)\n', beta);
fprintf('  Controllable: %s\n', mat2str(rank(Co_out) == size(A_aug_out, 1)));
poles_ol_out = eig(A_aug_out);
fprintf('  OL pole: [%.4f]  (integrator state only)\n\n', poles_ol_out(1));

switch lower(method)

    case 'lqr'
        Q_out     = 1;
        R_out     = 1;
        K_aug_out = dlqr(A_aug_out, B_aug_out, Q_out, R_out);

        A_cl_out     = A_aug_out - B_aug_out * K_aug_out;
        poles_cl_out = eig(A_cl_out);

        % Equivalent pole / time-constant summary for reporting + struct
        z_cl_out = poles_cl_out(1);
        if abs(z_cl_out) < 1 && z_cl_out > 0
            tau_out = -dt / log(z_cl_out);
        else
            tau_out = NaN;
        end

        fprintf('  LQR: Q = %.4f,  R = %.4f\n', Q_out, R_out);

    case 'pole_placement'
        tau_out  = perturbed.tau_out;
        z_cl_out = exp(-dt / tau_out);
        K_aug_out = acker(A_aug_out, B_aug_out, z_cl_out);

        A_cl_out     = A_aug_out - B_aug_out * K_aug_out;
        poles_cl_out = eig(A_cl_out);

        fprintf('  Pole placement: z = [%.4f]\n', z_cl_out);

    otherwise
        error('design_controller: method must be ''lqr'' or ''pole_placement''');
end


fprintf('  K_aug_out = [K_i_l = %.6f]\n', K_aug_out(1));
fprintf('  CL pole: [%-.4f],  stable: %s\n\n', ...
        real(poles_cl_out(1)), ...
        mat2str(all(abs(poles_cl_out) < 1)));

K_il = -K_aug_out(1);

% Anti-windup bounds for xi_l (in xi domain)
% Control law used in implementation:
%   Dq_ref = -K_aug_out * xi_l = K_il * xi_l
%   q_ref  = q0 + Dq_ref
%
% Therefore:
%   0 <= q0 + K_il*xi_l <= q_max
den = max(abs(K_il), 1e-6);
xi_l_max =  (perturbed.q_max - q0) / den;
xi_l_min = -(q0 - 0)               / den;

% =============================================================================
% PACK OUTPUT STRUCTURE
% =============================================================================

% -- Inner controller ---------------------------------------------------------
controller.inner_c.K_q      = K_aug_in(1);
controller.inner_c.K_i      = K_aug_in(2);
controller.inner_c.K_aug    = K_aug_in;
controller.inner_c.A        = A_in;
controller.inner_c.B_m      = B_in;
controller.inner_c.A_aug    = A_aug_in;
controller.inner_c.B_aug    = B_aug_in;
controller.inner_c.poles_ol = real(poles_ol_in);
controller.inner_c.poles_cl = real(poles_cl_in);
controller.inner_c.xi_q_max = xi_q_max;
controller.inner_c.xi_q_min = xi_q_min;
controller.inner_c.B0       = B0;
controller.inner_c.q0       = q0;
controller.inner_c.q_max    = perturbed.q_max;

% -- Outer controller ---------------------------------------------------------
controller.outer_c.K_il         = -K_aug_out(1);
controller.outer_c.K_aug        = K_aug_out;
controller.outer_c.A_aug        = A_aug_out;
controller.outer_c.B_aug        = B_aug_out;
controller.outer_c.poles_ol     = real(poles_ol_out);
controller.outer_c.poles_cl     = real(poles_cl_out);
controller.outer_c.z_cl         = real(z_cl_out);
controller.outer_c.tau_out      = tau_out;
controller.outer_c.xi_l_max     = xi_l_max;
controller.outer_c.xi_l_min     = xi_l_min;
controller.outer_c.q0           = q0;
controller.outer_c.q_max        = perturbed.q_max;
controller.outer_c.L_p95_target = perturbed.L_p95_target;
controller.outer_c.L_mean_target = perturbed.L_mean_target;

% Shared
controller.method_id = double(lower(method(1)) == 'l');

end
