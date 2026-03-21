function controller = design_controller(perturbed, method)
%DESIGN_CONTROLLER  Direct integral state-feedback controller — chapter 2.
%
%   closes the loop directly on measured l_p95 using batch size b as the
%   control input.  one loop, one augmented state, one gain vector.
%
%   controller = design_controller(perturbed, method)
%
%   inputs
%     perturbed  plant parameter struct (from setup_plant.m)
%     method     'lqr'  or  'pole_placement'
%
% =========================================================================
% plant equations (nonlinear)
% =========================================================================
%
%   state equation  f(q, b):
%     q[k+1] = q[k] + lambda - b[k]        (lambda replaces stochastic a[k])
%
%   output equation  h(q, b):
%     l_p95[k] = alpha*b + gamma*b^2 + beta*q + 1.645*delta/sqrt(b)
%
%   control input : b[k]   (batch size)
%   state         : q[k]   (queue depth)
%   output        : l_p95[k]
%
%   note: 1.645*delta/sqrt(b) is the expected p95 offset above the mean.
%   it is a deterministic function of b — not a noise term — so it belongs
%   in h(q,b) and contributes to the jacobian dh/db.
%
% =========================================================================
% equilibrium
% =========================================================================
%
%   at equilibrium q[k+1] = q[k], so:
%     q_eq + lambda - b_eq = q_eq   =>   b_eq = lambda
%
%   setting h(q_eq, b_eq) = l_target and solving for q_eq:
%     q_eq = ( l_target - alpha*b_eq - gamma*b_eq^2
%                        - 1.645*delta/sqrt(b_eq) ) / beta
%
% =========================================================================
% jacobian linearisation
% =========================================================================
%
%   for a nonlinear discrete system  x[k+1] = f(x,u),  y[k] = h(x,u),
%   linearise around (q_eq, b_eq):
%
%     dq[k+1] = a * dq[k]  +  b_plant * db[k]
%     dl[k]   = c * dq[k]  +  d       * db[k]
%
%   where  dq = q - q_eq,  db = b - b_eq,  dl = l_p95 - l_target.
%
%   jacobian a  =  df/dq  =  1
%     the queue is a pure integrator: no natural decay.
%
%   jacobian b_plant  =  df/db  =  -1
%     increasing b drains the queue by one unit per tick.
%
%   jacobian c  =  dh/dq  =  beta
%     each extra request in the queue adds beta ms to latency.
%
%   jacobian d  =  dh/db  =  alpha + 2*gamma*b_eq - 0.8225*delta/b_eq^(3/2)
%     define this as c1 (direct feedthrough gain, ms/req).
%     two competing effects:
%       + (alpha + 2*gamma*b_eq): larger batches have more service time
%       - 0.8225*delta/b_eq^(3/2): larger batches shrink the p95 noise floor
%         (0.8225 = 1.645/2, from d/db of 1.645*delta*b^(-1/2))
%     net c1 > 0: service-time penalty dominates the p95 benefit.
%
%   c1 != 0 means b affects l_p95 at the same time step — direct
%   feedthrough — which is the structural source of nmp behaviour.
%
%   linearised plant:
%     dq[k+1] =      dq[k]  +  (-1) * db[k]       ... (I)
%     dl[k]   = beta*dq[k]  +   c1  * db[k]        ... (II)
%
% =========================================================================
% integral augmentation
% =========================================================================
%
%   to guarantee zero steady-state error in l_p95 under constant
%   disturbances, we augment the state with an integrator on latency error.
%
%   latency error:
%     e[k] = l_target - l_p95[k]  =  -dl[k]
%
%   integrator state  xi  (accumulates measured latency error directly):
%     xi[k+1] = xi[k] + e[k]
%             = xi[k] - dl[k]                                        ... (III)
%
%   implementation note — normal (measurement-based) integrator:
%     in the simulink block, xi is updated using the actual measured
%     l_p95, not the linearised prediction beta*dq + c1*db.  that is:
%
%       xi[k+1] = xi[k] + (l_target - l_p95_measured[k])
%
%     in the linear regime around (q_eq, b_eq), this is equivalent to
%     substituting (II) into (III).  away from equilibrium, the two
%     differ — the measurement-based update is more robust because it
%     does not depend on knowledge of beta or c1.
%
%     the cost: l_p95_measured comes from a 60-sample rolling buffer
%     (~6s window at dt=0.1s).  this introduces significant lag into
%     the integral path.  to compensate, keep k_i small by using a
%     large lqr_r or slow outer pole (pp_tau2 >> tau_nmp).
%
%   for lqr design we substitute the linearised (II) into (III) to
%   obtain the augmented state-space matrices — this gives the correct
%   optimal gains for the linear regime:
%
%     xi[k+1] = xi[k] - beta*dq[k] - c1*db[k]       ... (III expanded)
%
% =========================================================================
% augmented state-space
% =========================================================================
%
%   augmented state: x = [dq ; xi]
%
%   stacking (I) and (III expanded):
%
%     x[k+1] = a_aug * x[k]  +  b_aug * db[k]
%
%     a_aug = [ df/dq    0  ] = [  1     0 ]
%             [-dh/dq    1  ]   [ -beta  1 ]
%
%     b_aug = [  df/db  ] = [  -1  ]
%             [ -dh/db  ]   [ -c1  ]
%
%   open-loop poles: eig(a_aug) = {1, 1}
%     - z=1 from the queue integrator (physical)
%     - z=1 from the appended latency integrator (by design)
%
%   controllability:
%     det([b_aug | a_aug*b_aug]) = det([-1, -1; -c1, beta-c1]) = -beta
%     since beta > 0 always, the system is always controllable.
%
% =========================================================================
% control law
% =========================================================================
%
%   state feedback:
%     db[k] = -k_vec * x[k] = -k_q*dq[k] - k_i*xi[k]
%     b[k]  = b_eq + db[k]   clamped to [b_min, b_max]
%
%   k_vec = [k_q, k_i] designed by dlqr or acker on (a_aug, b_aug).
%
% =========================================================================
% anti-windup bounds for xi
% =========================================================================
%
%   xi accumulates latency error in units of ms (one ms per tick of error).
%   the bounds are derived from the control law, not the integrator update.
%
%   from the control law at dq = 0:
%     db = -k_i * xi   =>   b = b_eq - k_i * xi
%
%   b saturates at b_max when:   b_eq - k_i * xi = b_max
%     =>  xi = (b_eq - b_max) / k_i
%
%   b saturates at b_min when:   b_eq - k_i * xi = b_min
%     =>  xi = (b_eq - b_min) / k_i
%
%   using |k_i| to handle sign correctly:
%     xi_max = (b_eq - b_min) / |k_i|
%     xi_min = (b_eq - b_max) / |k_i|
%
%   note: xi_max and xi_min set the range over which the integral term
%   can move b.  xi beyond these bounds has no additional effect on b
%   (b is already saturated), so further accumulation only delays recovery.
%
% =========================================================================

% -- unpack parameters -----------------------------------------------------
alpha    = perturbed.alpha;
gamma_p  = perturbed.gamma;   % gamma_p avoids clash with matlab built-in
beta     = perturbed.beta;
delta    = perturbed.delta;
lambda   = perturbed.lambda_mean;
dt       = perturbed.dt;
b_min    = perturbed.B_min;
b_max    = perturbed.B_max;
q_max    = perturbed.q_max;
l_target = perturbed.L_p95_target;

% -- equilibrium -----------------------------------------------------------
b_eq = lambda;
q_eq = (l_target - alpha*b_eq - gamma_p*b_eq^2 ...
        - 1.645*delta/sqrt(b_eq)) / beta;

fprintf('=== equilibrium ===\n');
fprintf('  b_eq = %.4f  req/tick\n', b_eq);
fprintf('  q_eq = %.4f  requests\n', q_eq);
l_eq_check = alpha*b_eq + gamma_p*b_eq^2 + beta*q_eq + 1.645*delta/sqrt(b_eq);
fprintf('  l_p95 at equilibrium = %.2f ms  (target = %.0f ms)\n\n', ...
        l_eq_check, l_target);

% -- jacobian elements -----------------------------------------------------
a_jac       =  1;
b_plant_jac = -1;
c_jac       =  beta;

% c1 = dh/db: service-time penalty minus p95-spread benefit
c1 = alpha + 2*gamma_p*b_eq - 0.8225*delta / (b_eq^1.5);

fprintf('=== jacobian at equilibrium ===\n');
fprintf('  a  = df/dq = %.4f  (queue integrator)\n',            a_jac);
fprintf('  b  = df/db = %.4f  (b drains queue)\n',              b_plant_jac);
fprintf('  c  = dh/dq = %.4f  ms/req  (queue raises latency)\n', c_jac);
fprintf('  d  = dh/db = %.4f  ms/req  (direct feedthrough c1)\n', c1);
fprintf('       service-time term : +%.4f  (alpha + 2*gamma*b_eq)\n', ...
        alpha + 2*gamma_p*b_eq);
fprintf('       p95-spread term   : -%.4f  (0.8225*delta/b_eq^1.5)\n\n', ...
        0.8225*delta / b_eq^1.5);

% -- augmented state-space matrices ----------------------------------------
a_aug = [a_jac,   0;
        -c_jac,   1];

b_aug = [b_plant_jac;
        -c1        ];

fprintf('=== augmented plant ===\n');
fprintf('  a_aug = [%.4f  %.4f ; %.4f  %.4f]\n', ...
        a_aug(1,1), a_aug(1,2), a_aug(2,1), a_aug(2,2));
fprintf('  b_aug = [%.4f ; %.4f]\n', b_aug(1), b_aug(2));

co     = ctrb(a_aug, b_aug);
det_co = det(co);
fprintf('  controllability det = %.4f  (= -beta = %.4f)\n', det_co, -beta);
fprintf('  controllable: %s\n', mat2str(rank(co) == 2));

ol_poles = eig(a_aug);
fprintf('  open-loop poles: z = [%.4f, %.4f]  (z=1 twice)\n\n', ...
        ol_poles(1), ol_poles(2));

% -- controller design -----------------------------------------------------
switch lower(method)

    case 'lqr'
        % lqr_r should be large enough that the closed-loop bandwidth stays
        % well below the effective bandwidth of the 60-sample rolling buffer
        % (cutoff ~ 1/(N*dt) = 1/(60*0.1) = 0.167 Hz = 6s period).
        % if k_i is too large the integrator reacts faster than l_p95 can
        % update, causing overshoot and oscillation.
        q_lqr = diag([perturbed.lqr_q, perturbed.lqr_xi]);
        r_lqr = perturbed.lqr_r;
        k_vec = dlqr(a_aug, b_aug, q_lqr, r_lqr);
        fprintf('=== lqr design ===\n');
        fprintf('  q = diag([%.2f, %.2f]),  r = %.2f\n', ...
                q_lqr(1,1), q_lqr(2,2), r_lqr);

    case 'pole_placement'
        % dominant pole tau should be >> rolling buffer window (6s).
        % second pole slower still to support the integrator cleanly.
        tau1  = perturbed.pp_tau1;
        tau2  = perturbed.pp_tau2;
        z1    = exp(-dt / tau1);
        z2    = exp(-dt / tau2);
        k_vec = acker(a_aug, b_aug, [z1, z2]);
        fprintf('=== pole placement design ===\n');
        fprintf('  tau = [%.2f, %.2f] s  =>  z = [%.4f, %.4f]\n', ...
                tau1, tau2, z1, z2);

    otherwise
        error('method must be ''lqr'' or ''pole_placement''');
end

% -- closed-loop analysis --------------------------------------------------
a_cl     = a_aug - b_aug * k_vec;
cl_poles = eig(a_cl);

fprintf('  k_vec = [k_q = %.4f,  k_i = %.4f]\n', k_vec(1), k_vec(2));
fprintf('  closed-loop poles: z = [%.4f, %.4f]\n', cl_poles(1), cl_poles(2));
fprintf('  stable: %s\n\n', mat2str(all(abs(cl_poles) < 1)));

% -- nmp analysis ----------------------------------------------------------
% transfer function from db to dl: g(z) = (c1*z - (c1+1)) / (z-1)
% zero at z = (c1+1)/c1
z_nmp   = (c1 + 1) / c1;
tau_nmp = abs(-dt / log(abs(z_nmp)));
fprintf('=== nmp analysis ===\n');
fprintf('  nmp zero at z = %.4f  (z > 1 confirms nmp)\n', z_nmp);
fprintf('  nmp time constant tau_nmp = %.3f s\n', tau_nmp);
fprintf('  rolling buffer lag        = %.1f s  (N*dt = 60*0.1)\n', 60*dt);
fprintf('  effective bandwidth limit = max(tau_nmp, buffer_lag) = %.1f s\n\n', ...
        max(tau_nmp, 60*dt));

% -- anti-windup limits ----------------------------------------------------
% xi accumulates ms of latency error per tick.
% bounds derived from control law: b = b_eq - k_i*xi  (at dq=0)
% b_max hit when xi = (b_eq - b_max) / |k_i|
% b_min hit when xi = (b_eq - b_min) / |k_i|
k_i_abs = abs(k_vec(2));
xi_max  = (b_eq - b_min) / k_i_abs;
xi_min  = (b_eq - b_max) / k_i_abs;

fprintf('=== anti-windup bounds (xi in ms) ===\n');
fprintf('  xi_max = %.2f ms  (b saturates at b_min = %.0f)\n', xi_max, b_min);
fprintf('  xi_min = %.2f ms  (b saturates at b_max = %.0f)\n\n', xi_min, b_max);

% -- pack output struct (all lowercase, underscore-separated) --------------
controller.k_q         = k_vec(1);
controller.k_i         = k_vec(2);
controller.k_vec       = k_vec;
controller.a_aug       = a_aug;
controller.b_aug       = b_aug;
controller.a_jac       = a_jac;
controller.b_plant_jac = b_plant_jac;
controller.c_jac       = c_jac;   % = beta
controller.c1          = c1;      % = dh/db (direct feedthrough)
controller.cl_poles    = real(cl_poles);
controller.ol_poles    = real(ol_poles);
controller.b_eq        = b_eq;
controller.q_eq        = q_eq;
controller.l_target    = l_target;
controller.b_min       = b_min;
controller.b_max       = b_max;
controller.q_max       = q_max;
controller.xi_max      = xi_max;
controller.xi_min      = xi_min;
controller.z_nmp       = z_nmp;
controller.tau_nmp     = tau_nmp;
controller.method_id   = double(lower(method(1)) == 'l');  % 1=lqr, 0=pp

end
