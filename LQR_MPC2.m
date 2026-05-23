% Nonlinear UAV + suspended payload simulation with linearization-based MPC

clear;
clc;
figs = findall(0, 'Type', 'figure');
for k = 1:length(figs)
    clf(figs(k));
end

% Figure Sizing
fig.Units = 'inches';
fig.Position = [1 1 6 4];   % [l b w h]

%% ============================================================
%                     System Parameters
% =============================================================
M    = 1.5;       % UAV mass [kg]
m    = 0.3;       % payload mass [kg]
l    = 0.75;      % cable length [m]
g    = 9.81;      % gravity [m/s^2]
Jphi = 0.01;      % roll inertia [kg m^2]

% Actuator limits (on virtual inputs v1, v2, and torque tau)
f_max   = 100;    % max magnitude for v1 and v2
tau_min = -5;     % min roll torque [N*m]
tau_max =  5;     % max roll torque [N*m]

%% ============================================================
%                Simulation configuration
% =============================================================
t0         = 0;
tf         = 100;
dt_control = 0.05;         % controller update period [s]

% Initial state: [y ydot z zdot theta thetadot phi phidot]
x0 = [0; 0; 0; 0; 0.2; 0; 0.05; 0];

% Hover reference (constant)
y_ref = 2.0;   % desired horizontal position [m]
z_ref = 2.0;   % desired altitude [m]

x_ref_full = @(t) [y_ref; 0; ...
                   z_ref; 0; ...
                   0;     0; ...
                   0;     0];

%% ============================================================
%                     MPC Weights (updated)
% =============================================================
% Strong damping on payload swing
%Q = diag([ 80,   8, ...    % y, ydot
%           80,   8, ...    % z, zdot
%         1200, 200, ...    % theta, thetadot (stronger damping)
%           20,   5]);      % phi, phidot (roll lower priority)

Q = diag([ 200,  20, ...    % y, ydot (stronger position control)
           200,  20, ...    % z, zdot (stronger altitude control)
         3000, 600, ...     % theta, thetadot (very strong damping)
           20,   5]);       % phi, phidot (roll lower priority)


% Inputs: keep tau expensive to reduce thrashing
R = 2*diag([10, 10, 20]);    % v1, v2, tau

% MPC horizon
N = 40;

%% ============================================================
%              Logging and ODE settings
% =============================================================
tlog = [];
xlog = [];
ulog = [];

odeopts = odeset('RelTol',1e-6,'AbsTol',1e-8);

x     = x0;
t_sim = t0;

%% ============================================================
%                Main Simulation Loop (MPC)
% =============================================================
while t_sim < tf - 1e-9

    % Nominal input for linearization
    u_nom = [0; 0; 0];

    % Linearize about current state and nominal input
    [A,B] = linearize_fd(x,u_nom);

    % Time span for this control interval
    t_next = min(t_sim + dt_control, tf);
    xref   = x_ref_full(t_sim);

    % Compute MPC control at current state
    u_mpc = mpc_control_nobox(x, xref, A, B, Q, R, N, dt_control, f_max, tau_min, tau_max);

    % Control law: keep u constant over this interval
    u_fun = @(t_local,x_local) u_mpc;

    % Saturated dynamics wrapper
    dyn_wrap = @(tt,xx) dynamics( ...
        tt, xx, ...
        saturate_u(u_fun(tt,xx), f_max, tau_min, tau_max) ...
    );

    % Integrate nonlinear dynamics over [t_sim, t_next]
    [Tstep, Xstep] = ode45(dyn_wrap, [t_sim t_next], x, odeopts);

    % Log states
    tlog = [tlog; Tstep];
    xlog = [xlog; Xstep];

    % Log controls (piecewise constant)
    for k = 1:size(Xstep,1)
        ulog = [ulog; u_mpc.'];
    end

    % Update for next loop
    x     = Xstep(end,:).';
    t_sim = t_next;
end

%% ============================================================
%                         Plotting
% =============================================================
T = tlog;
X = xlog;
U = ulog;

Yref_vec = y_ref * ones(size(T));
Zref_vec = z_ref * ones(size(T));

% Aircraft Translational States
figure(1);
subplot(2,2,1);
plot(T, X(:,1), 'LineWidth', 1.3); hold on;
plot(T, Yref_vec, 'r--','LineWidth',1.0);
ylabel('$y \ [m]$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('Lateral Position $y$', 'interpreter', 'latex'); legend('y(t)','y_{ref}','Location','best'); grid on;

subplot(2,2,2);
plot(T, X(:,2), 'LineWidth', 1.3);
ylabel('$\dot{y} [m/s]$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('Lateral Velocity $\dot{y}$', 'interpreter', 'latex'); grid on;

subplot(2,2,3);
plot(T, X(:,3), 'LineWidth', 1.3); hold on;
plot(T, Zref_vec, 'r--','LineWidth',1.0);
ylabel('$z \ [m]$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('Vertical Position $z$', 'interpreter', 'latex'); legend('z(t)','z_{ref}','Location','best'); grid on;

subplot(2,2,4);
plot(T, X(:,4), 'LineWidth', 1.3);
ylabel('$\dot{z} [m/s]$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('Vertical Velocity $\dot{z}$', 'interpreter', 'latex'); grid on;

sgtitle('MPC - Quadrotor Translational States (y, z)');

% Aircraft Roll States
figure(2);
subplot(2,1,1);
plot(T, X(:,7), 'LineWidth', 1.3);
ylabel('$\phi \ [rad]$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('UAV Roll Angle $\phi$', 'interpreter', 'latex'); grid on;

subplot(2,1,2);
plot(T, X(:,8), 'LineWidth', 1.3);
ylabel('$\dot{\phi} [rad/s]$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('UAV Roll Rate $\dot{\phi}$', 'interpreter', 'latex'); grid on;

sgtitle('MPC - Quadrotor Roll States');

% Payload Swing States
figure(3);
subplot(2,1,1);
plot(T, X(:,5), 'LineWidth', 1.3);
ylabel('$\theta \ [rad]$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('Payload Swing Angle $\theta$', 'interpreter', 'latex'); grid on;

subplot(2,1,2);
plot(T, X(:,6), 'LineWidth', 1.3);
ylabel('$\dot{\theta} [rad/s]$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('Payload Swing Angular Rate $\dot{\theta}$', 'interpreter', 'latex'); grid on;

sgtitle('MPC - Suspended Payload Swing States');

% MPC Control Inputs
figure(4);
subplot(3,1,1);
plot(T, U(:,1), 'LineWidth', 1.3);
ylabel('$v_1$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('Control Input $v_1$', 'interpreter', 'latex'); grid on;

subplot(3,1,2);
plot(T, U(:,2), 'LineWidth', 1.3);
ylabel('$v_2$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('Control Input $v_2$', 'interpreter', 'latex'); grid on;

subplot(3,1,3);
plot(T, U(:,3), 'LineWidth', 1.3);
ylabel('$\tau \ [N \cdot m]$', 'interpreter', 'latex'); xlabel('Time [s]', 'interpreter', 'latex');
title('Control Input $\tau$', 'interpreter', 'latex'); grid on;

sgtitle('MPC Control Inputs');

final_error = norm(X(end,:)' - x_ref_full(T(end)));
fprintf("Final state error : %.4f\n", final_error);

save('uav_payload_mpc_results.mat','T','X','U');

%% ============================================================
%                   Dynamics Function
% =============================================================
function dx = dynamics(~,x,v)
    M    = 1.5;
    m    = 0.3;
    l    = 0.75;
    g    = 9.81;
    Jphi = 0.01;

    y        = x(1);
    ydot     = x(2);
    z        = x(3);
    zdot     = x(4);
    theta    = x(5);
    thetadot = x(6);
    phi      = x(7);
    phidot   = x(8);

    M11 = (M + m);
    M12 = 0;
    M13 = m*l*cos(theta);

    M21 = 0;
    M22 = (M + m);
    M23 = m*l*sin(theta);

    M31 = m*l*cos(theta);
    M32 = m*l*sin(theta);
    M33 = -m*l;

    Minv = inv([M11 M12 M13;
                M21 M22 M23;
                M31 M32 M33]);

    h1 = -m*l*thetadot^2*sin(theta);
    h2 =  m*l*thetadot^2*cos(theta);
    h3 =  m*g*l*sin(theta);
    h  = [h1; h2; h3];

    accs = Minv * (-h + [v(1); v(2); 0]);

    ydd      = accs(1);
    zdd      = accs(2);
    thetadd  = accs(3);
    phidd    = v(3)/Jphi;

    dx = [ydot;
          ydd;
          zdot;
          zdd;
          thetadot;
          thetadd;
          phidot;
          phidd];
end

%% ============================================================
%                 Linearization (Finite Difference)
% =============================================================
function [A,B] = linearize_fd(x,u)
    n  = length(x);
    m  = length(u);
    fx = dynamics(0,x,u);
    eps = 1e-6;

    A = zeros(n,n);
    B = zeros(n,m);

    for i = 1:n
        xp = x;
        xp(i) = xp(i) + eps;
        fp = dynamics(0,xp,u);
        A(:,i) = (fp - fx)/eps;
    end

    for j = 1:m
        up = u;
        up(j) = up(j) + eps;
        fp = dynamics(0,x,up);
        B(:,j) = (fp - fx)/eps;
    end
end

%% ============================================================
%                    Input Saturation
% =============================================================
function u_sat = saturate_u(u, f_max, tau_min, tau_max)
    u_sat = u;
    u_sat(1) = max(min(u_sat(1), f_max), -f_max);
    u_sat(2) = max(min(u_sat(2), f_max), -f_max);
    u_sat(3) = max(min(u_sat(3), tau_max), tau_min);
end

%% ============================================================
%                 MPC Controller
% =============================================================
function u0 = mpc_control_nobox(x, x_ref, A, B, Q, R, N, dt, f_max, tau_min, tau_max)
    nx = size(A,1);
    nu = size(B,2);

    % Discretize
    Ad = expm(A*dt);
    Bd = B*dt;

    % Prediction matrices
    [Sx, Su] = build_prediction_matrices(Ad, Bd, N);

    Qbar = kron(eye(N), Q);
    Rbar = kron(eye(N), R);

    Xref = repmat(x_ref, N, 1);

    H = Su' * Qbar * Su + Rbar;
    f = Su' * Qbar * (Sx * x - Xref);

    % Box constraints on U
    u_min = [-f_max; -f_max; tau_min];
    u_max = [ f_max;  f_max; tau_max];

    Umin = repmat(u_min, N, 1);
    Umax = repmat(u_max, N, 1);

    % Solve QP: min 0.5*U'HU + f'U s.t. Umin <= U <= Umax
    U0 = zeros(N*nu,1);
    Uopt = projected_gradient_qp(H, f, Umin, Umax, U0);

    u0 = Uopt(1:nu);
end

%% ============================================================
%          Build Prediction Matrices for MPC (Ad,Bd,N)
% =============================================================
function [Sx, Su] = build_prediction_matrices(Ad, Bd, N)
    nx = size(Ad,1);
    nu = size(Bd,2);

    Sx = zeros(N*nx, nx);
    Su = zeros(N*nx, N*nu);

    A_power = eye(nx);
    for i = 1:N
        A_power = Ad * A_power;
        Sx((i-1)*nx+1:i*nx, :) = A_power;

        for j = 1:i
            A_j = Ad^(i-j);
            Su((i-1)*nx+1:i*nx, (j-1)*nu+1:j*nu) = A_j * Bd;
        end
    end
end

%% ============================================================
%        Simple Projected Gradient Solver for Box-QP
%        min 0.5*U'HU + f'U  s.t. Umin <= U <= Umax
% =============================================================
function U = projected_gradient_qp(H, f, Umin, Umax, U0)
    max_iter = 200;
    alpha    = 1.0 / (max(eig(H)) + 1e-6);
    U        = U0;

    for k = 1:max_iter
        grad = H*U + f;
        U    = U - alpha * grad;
        U    = min(max(U, Umin), Umax);  % projection onto box
    end
end
