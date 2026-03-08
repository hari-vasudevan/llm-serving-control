function controller = design_controller(perturbed, method)
%DESIGN_CONTROLLER  Direct integral state-feedback controller — Chapter 2.
%
%   Closes the loop directly on the measured l_p95 output using batch
%   size b as the control input.  No inner/outer cascade — one loop,
%   one augmented state, one gain vector.
%
%   controller = design_controller(perturbed, method)
%
%   inputs
%     perturbed  plant parameter struct (from setup_plant.m)
%     method     'lqr'  or  'pole_placement'
%
% =========================================================================
% PLANT EQUATIONS (nonlinear)
% =========================================================================
%
%   state equation  f(q, b):
%     q[k+1] = q[k] + lambda - b[k]        (arrivals replace stochastic a[k])
%
%   output equation  h(q, b):
%     l_p95[k] = alpha*b + gamma*b^2 + beta*q + 1.645*delta/sqrt(b)
%
%   control input : b[k]   (batch size)
%   state         : q[k]   (queue depth)
%   output        : l_p95[k]
%
%   note: the 1.645*delta/sqrt(b) term is the expected p95 offset above
%   the mean due to per-tick latency noise.  it is a deterministic function
%   of b — not a noise term — so it belongs in h(q,b) and contributes to
%   the jacobian of the output w.r.t. b.
%
% =========================================================================
% EQUILIBRIUM
% =========================================================================
%
%   at equilibrium q[k+1] = q[k], so f(q_eq, b_eq) = q_eq:
%
%     q_eq + lambda - b_eq = q_eq   =>   b_eq = lambda
%
%   setting h(q_eq, b_eq) = l_target and solving for q_eq:
%
%     l_target = alpha*b_eq + gamma*b_eq^2 + beta*q_eq + 1.645*delta/sqrt(b_eq)
%     q_eq     = ( l_target - alpha*b_eq - gamma*b_eq^2
%                            - 1.645*delta/sqrt(b_eq) ) / beta
%
% =========================================================================
% JACOBIAN LINEARISATION
% =========================================================================
%
%   for a nonlinear discrete system  x[k+1] = f(x,u),  y[k] = h(x,u)
%   the linearisation around (x_eq, u_eq) is:
%
%     dx[k+1] = a * dx[k]  +  b_plant * du[k]
%     dy[k]   = c * dx[k]  +  d * du[k]
%
%   where the jacobians are:
%     a       = df/dq  evaluated at (q_eq, b_eq)
%     b_plant = df/db  evaluated at (q_eq, b_eq)
%     c       = dh/dq  evaluated at (q_eq, b_eq)
%     d       = dh/db  evaluated at (q_eq, b_eq)
%
%   working through each:
%
%   a = df/dq = d/dq [q + lambda - b] = 1
%
%   b_plant = df/db = d/db [q + lambda - b] = -1
%
%   c = dh/dq = d/dq [alpha*b + gamma*b^2 + beta*q + 1.645*delta/sqrt(b)]
%             = beta
%
%   d = dh/db = d/db [alpha*b + gamma*b^2 + beta*q + 1.645*delta/sqrt(b)]
%             = alpha + 2*gamma*b_eq - 1.645*delta * (1/2) * b_eq^(-3/2)
%             = alpha + 2*gamma*lambda - 0.8225*delta / lambda^(3/2)
%
%   define:  c1 = d  (the direct feedthrough gain, units ms/req)
%
%   the positive term (alpha + 2*gamma*lambda) captures the service-time
%   penalty of increasing b: larger batches take longer to process.
%   the negative term (-0.8225*delta/lambda^(3/2)) captures the p95
%   benefit of increasing b: larger batches average out per-request noise,
%   shrinking the p95 spread.  net c1 is positive but smaller than
%   alpha + 2*gamma*lambda alone.
%
%   c1 > 0 means b directly raises l_p95 at the same time step — this is
%   the direct feedthrough d != 0 that makes the plant non-minimum phase.
%
%   linearised plant in deviation variables:
%     dq  = q - q_eq,  db = b - b_eq,  dl = l_p95 - l_target
%
%     dq[k+1] =  1 * dq[k]  +  (-1) * db[k]          ... (I)
%     dl[k]   =  c * dq[k]  +   c1  * db[k]           ... (II)
%              = beta*dq[k] + c1*db[k]
%
% =========================================================================
% INTEGRAL AUGMENTATION
% =========================================================================
%
%   to guarantee zero steady-state error in l_p95 under constant
%   disturbances, we augment the state with an integrator on latency error.
%
%   latency tracking error:
%     e[k] = 0 - dl[k] = -dl[k]          (reference is dl = 0)
%
%   integrator state  xi (greek letter xi, the latency integral):
%     xi[k+1] = xi[k] + e[k]
%             = xi[k] - dl[k]
%
%   substitute (II) into the integrator update:
%     xi[k+1] = xi[k] - beta*dq[k] - c1*db[k]         ... (III)
%
%   the term -c1*db[k] in (III) is the feedthrough from (II) propagating
%   into the integrator.  this is what makes b_aug(2) = -c1 instead of
%   zero as in chapter 1.
%
%   chapter 1 had: integrator on queue error => b_aug = [-1; 0]
%   chapter 2 has: integrator on latency error => b_aug = [-1; -c1]
%
% =========================================================================
% AUGMENTED STATE-SPACE
% =========================================================================
%
%   augmented state: x = [dq ; xi]
%
%   stacking (I) and (III):
%
%     x[k+1] = a_aug * x[k]  +  b_aug * db[k]
%
%     a_aug = [ df/dq    0  ] = [  1     0 ]
%             [-dh/dq    1  ]   [ -beta  1 ]
%
%     b_aug = [  df/db  ] = [   -1  ]
%             [ -dh/db  ]   [  -c1  ]
%
%   open-loop poles: eig(a_aug) = {1, 1}
%     - z=1 from the queue integrator (physical: queue has no natural decay)
%     - z=1 from the appended latency integrator (by design)
%
%   controllability determinant:
%     det([b_aug | a_aug*b_aug]) = det([-1, -1; -c1, beta-c1])
%                                = (-1)(beta-c1) - (-1)(-c1)
%                                = -beta
%   since beta > 0 always, the system is always controllable regardless
%   of the values of alpha, gamma, lambda, delta.
%
% =========================================================================
% CONTROL LAW
% =========================================================================
%
%   state feedback:
%     db[k] = -k_vec * x[k] = -k_q*dq[k] - k_i*xi[k]
%     b[k]  = b_eq + db[k]   clamped to [b_min, b_max]
%
%   k_vec = [k_q, k_i] designed by dlqr or acker on (a_aug, b_aug).
%
%   anti-windup: xi saturated in xi-domain.
%     at dq=0:  db = -k_i*xi  =>  b = b_eq - k_i*xi
%     b saturates at b_max when xi = (b_eq - b_max) / k_i
%     b saturates at b_min when xi = (b_eq - b_min) / k_i
%
% =========================================================================

% -- unpack parameters -----------------------------------------------------
alpha   = perturbed.alpha;
gamma_p = perturbed.gamma;    % gamma_p to avoid clash with matlab built-in
beta    = perturbed.beta;
delta   = perturbed.delta;
lambda  = perturbed.lambda_mean;
dt      = perturbed.dt;
b_min   = perturbed.B_min;
b_max   = perturbed.B_max;
q_max   = perturbed.q_max;
l_target = perturbed.L_p95_target;

% -- equilibrium -----------------------------------------------------------
% b_eq = lambda  (from queue balance: q[k+1]=q[k] => lambda = b_eq)
b_eq = lambda;

% q_eq from l_p95 = l_target at equilibrium, including the p95 offset term
q_eq = (l_target - alpha*b_eq - gamma_p*b_eq^2 ...
        - 1.645*delta/sqrt(b_eq)) / beta;

fprintf('=== equilibrium ===\n');
fprintf('  b_eq = %.4f  req/tick\n', b_eq);
fprintf('  q_eq = %.4f  requests\n', q_eq);
l_eq_check = alpha*b_eq + gamma_p*b_eq^2 + beta*q_eq + 1.645*delta/sqrt(b_eq);
fprintf('  l_p95 at equilibrium = %.2f ms  (target = %.0f ms)\n\n', ...
        l_eq_check, l_target);

% -- jacobian elements -----------------------------------------------------
%
%   a       = df/dq = 1             (queue integrator)
%   b_plant = df/db = -1            (b drains queue)
%   c       = dh/dq = beta          (queue depth raises latency linearly)
%   d = c1  = dh/db = alpha + 2*gamma*b_eq - 0.8225*delta/b_eq^1.5
%             (service time term raises latency, p95 spread term lowers it)

a_jac       =  1;
b_plant_jac = -1;
c_jac       =  beta;

% c1: direct feedthrough gain (ms per req change in b)
%   positive service-time sensitivity:  alpha + 2*gamma_p*b_eq
%   negative p95-spread sensitivity:   -0.8225*delta / b_eq^1.5
%   (0.8225 = 1.645/2, from d/db of 1.645*delta*b^(-1/2) = -0.8225*delta*b^(-3/2))
c1 = alpha + 2*gamma_p*b_eq - 0.8225*delta / (b_eq^1.5);

fprintf('=== jacobian at equilibrium ===\n');
fprintf('  a  = df/dq = %.4f  (queue is integrator)\n',      a_jac);
fprintf('  b  = df/db = %.4f  (b drains queue)\n',            b_plant_jac);
fprintf('  c  = dh/dq = %.4f  ms/req  (latency vs queue)\n',  c_jac);
fprintf('  d  = dh/db = %.4f  ms/req  (direct feedthrough)\n', c1);
fprintf('     service-time term : +%.4f  (alpha + 2*gamma*b_eq)\n', ...
        alpha + 2*gamma_p*b_eq);
fprintf('     p95-spread term   : -%.4f  (0.8225*delta/b_eq^1.5)\n\n', ...
        0.8225*delta / b_eq^1.5);

% -- augmented state-space matrices ----------------------------------------
%
%   state: x = [dq ; xi]
%
%   a_aug = [  a_jac      0  ] = [  1      0 ]
%           [-c_jac       1  ]   [ -beta   1 ]
%
%   b_aug = [  b_plant_jac ] = [  -1  ]
%           [ -d           ]   [ -c1  ]

a_aug = [a_jac,   0;
        -c_jac,   1];

b_aug = [b_plant_jac;
        -c1        ];

fprintf('=== augmented plant ===\n');
fprintf('  a_aug = [%.4f  %.4f ; %.4f  %.4f]\n', ...
        a_aug(1,1), a_aug(1,2), a_aug(2,1), a_aug(2,2));
fprintf('  b_aug = [%.4f ; %.4f]\n', b_aug(1), b_aug(2));

co = ctrb(a_aug, b_aug);
det_co = det(co);
fprintf('  controllability matrix det = %.4f  (= -beta = %.4f)\n', ...
        det_co, -beta);
fprintf('  controllable: %s\n', mat2str(rank(co) == 2));

ol_poles = eig(a_aug);
fprintf('  open-loop poles: z = [%.4f, %.4f]  (z=1 twice: queue + integrator)\n\n', ...
        ol_poles(1), ol_poles(2));

% -- controller design -----------------------------------------------------
switch lower(method)

    case 'lqr'
        % q_lqr = cost on dq  (larger => faster latency tracking, larger db swings)
        % xi_lqr = cost on xi (larger => faster integral windup removal)
        % r_lqr  = cost on db  (larger => smoother b, slower response)
        %
        % lqr naturally respects the nmp zero: pushing r_lqr too small
        % (aggressive control) causes large k_q which amplifies noise and
        % drives b into saturation.  the cost function encodes the tradeoff.
        q_lqr  = diag([perturbed.lqr_q, perturbed.lqr_xi]);
        r_lqr  = perturbed.lqr_r;
        k_vec  = dlqr(a_aug, b_aug, q_lqr, r_lqr);
        fprintf('=== lqr design ===\n');
        fprintf('  q = diag([%.2f, %.2f]),  r = %.2f\n', ...
                q_lqr(1,1), q_lqr(2,2), r_lqr);

    case 'pole_placement'
        % place poles at z = exp(-dt/tau) for each desired time constant.
        % nmp constraint: undershoot magnitude grows with bandwidth.
        % practical guideline: tau >= 1.0s keeps undershoot below ~2ms.
        tau1   = perturbed.pp_tau1;
        tau2   = perturbed.pp_tau2;
        z1     = exp(-dt / tau1);
        z2     = exp(-dt / tau2);
        k_vec  = acker(a_aug, b_aug, [z1, z2]);
        fprintf('=== pole placement design ===\n');
        fprintf('  tau = [%.3f, %.3f] s  =>  z = [%.4f, %.4f]\n', ...
                tau1, tau2, z1, z2);

    otherwise
        error('method must be ''lqr'' or ''pole_placement''');
end

% closed-loop analysis
a_cl    = a_aug - b_aug * k_vec;
cl_poles = eig(a_cl);

fprintf('  k_vec = [k_q = %.4f,  k_i = %.4f]\n', k_vec(1), k_vec(2));
fprintf('  closed-loop poles: z = [%.4f, %.4f]\n', cl_poles(1), cl_poles(2));
fprintf('  stable: %s\n\n', mat2str(all(abs(cl_poles) < 1)));

% nmp zero reminder
z_nmp = (c1 + 1) / c1;    % zero of g(z) = (c1*z - (c1+1)) / (z-1)
fprintf('=== nmp analysis ===\n');
fprintf('  open-loop transfer function zero at z = %.4f\n', z_nmp);
fprintf('  (z > 1 => non-minimum phase)\n');
tau_nmp = -dt / log(abs(z_nmp));
fprintf('  nmp time constant tau_nmp = %.3f s\n', abs(tau_nmp));
fprintf('  poles faster than tau_nmp are stable but cause undershoot\n\n');

% -- anti-windup limits for xi (saturation in xi-domain) -------------------
%
%   b = b_eq - k_q*dq - k_i*xi
%   at dq = 0:  b = b_eq - k_i*xi
%   b_max is hit when xi = (b_eq - b_max) / k_i  (sign depends on k_i sign)
%   b_min is hit when xi = (b_eq - b_min) / k_i
%
%   take the range symmetrically using |k_i|:
k_i_abs = abs(k_vec(2));
xi_max  = (b_eq - b_min) / k_i_abs;
xi_min  = (b_eq - b_max) / k_i_abs;

% -- pack output struct (all lowercase, underscore-separated) --------------
controller.k_q        = k_vec(1);
controller.k_i        = k_vec(2);
controller.k_vec      = k_vec;
controller.a_aug      = a_aug;
controller.b_aug      = b_aug;
controller.a_jac      = a_jac;
controller.b_plant_jac = b_plant_jac;
controller.c_jac      = c_jac;
controller.c1         = c1;
controller.cl_poles   = real(cl_poles);
controller.ol_poles   = real(ol_poles);
controller.b_eq       = b_eq;
controller.q_eq       = q_eq;
controller.l_target   = l_target;
controller.b_min      = b_min;
controller.b_max      = b_max;
controller.q_max      = q_max;
controller.xi_max     = xi_max;
controller.xi_min     = xi_min;
controller.z_nmp      = z_nmp;
controller.method_id  = double(lower(method(1)) == 'l');  % 1=lqr, 0=pp

end
