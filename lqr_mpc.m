% Nonlinear UAV + suspended payload simulation with LQR control
% (LQR can be viewed as the rho -> infinity limit of H-infinity)
% main.m

clear; close all; clc;

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
tf         = 20;
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
%                     Controller Weights
% =============================================================
% State weighting (positions + angles heavily penalized)
Q = diag([100,  5, ...   % y, ydot
          100,  5, ...   % z, zdot
          200, 10, ...   % theta, thetadot
           50,  5]);     % phi, phidot

r_weight = 1.0;                % input weight (moderate)
R       = r_weight * eye(3);   % input weight matrix

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
%                Main Simulation Loop
% =============================================================
while t_sim < tf - 1e-9

    % Nominal input for hover in (v1,v2,tau) coordinates
    % v1 = 0, v2 = 0, tau = 0  => f = (M+m)g, phi = 0 in original variables
    u_nom = [0; 0; 0];

    % Linearize about current state and nominal input
    [A,B] = linearize_fd(x,u_nom);

    % Solve standard LQR ARE:
    % A'P + P A - P B R^{-1} B' P + Q = 0
    [P,~,~] = care(A,B,Q,R);

    % State feedback gain: u = u_nom - K (x - x_ref)
    K = (R \ (B' * P));   % equivalent to inv(R)*B'*P

    % Time span for this control interval
    t_next = min(t_sim + dt_control, tf);
    xref   = x_ref_full(t_sim);   % [2;0;2;0;0;0;0;0]

    % Control law: u = u_nom - K (x - x_ref)
    u_fun = @(t_local,x_local) u_nom - K*(x_local - xref);

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

    % Log controls along this interval
    for k = 1:size(Xstep,1)
        xk = Xstep(k,:).';
        uk = u_fun(Tstep(k), xk);
        uk = saturate_u(uk, f_max, tau_min, tau_max);
        ulog = [ulog; uk.'];
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

% Reference vectors for plotting
Yref_vec = y_ref * ones(size(T));
Zref_vec = z_ref * ones(size(T));

%% ---------- Figure 1: Translational states (y,z and rates) ----------
figure;

subplot(2,2,1);
plot(T, X(:,1), 'LineWidth', 1.3); hold on;
plot(T, Yref_vec, 'r--','LineWidth',1.0);
ylabel('y [m]');
xlabel('time [s]');
title('Lateral Position y');
legend('y(t)','y_{ref}','Location','best');
grid on;

subplot(2,2,2);
plot(T, X(:,2), 'LineWidth', 1.3);
ylabel('\dot{y} [m/s]');
xlabel('time [s]');
title('Lateral Velocity \dot{y}');
grid on;

subplot(2,2,3);
plot(T, X(:,3), 'LineWidth', 1.3); hold on;
plot(T, Zref_vec, 'r--','LineWidth',1.0);
ylabel('z [m]');
xlabel('time [s]');
title('Vertical Position z');
legend('z(t)','z_{ref}','Location','best');
grid on;

subplot(2,2,4);
plot(T, X(:,4), 'LineWidth', 1.3);
ylabel('\dot{z} [m/s]');
xlabel('time [s]');
title('Vertical Velocity \dot{z}');
grid on;

sgtitle('Quadrotor Translational States (y,z)');

%% ---------- Figure 2: UAV attitude (roll) and rate ----------
figure;

subplot(2,1,1);
plot(T, X(:,7), 'LineWidth', 1.3);
ylabel('\phi [rad]');
xlabel('time [s]');
title('UAV Roll Angle \phi');
grid on;

subplot(2,1,2);
plot(T, X(:,8), 'LineWidth', 1.3);
ylabel('\dot{\phi} [rad/s]');
xlabel('time [s]');
title('UAV Roll Rate \dot{\phi}');
grid on;

sgtitle('Quadrotor Attitude States (Roll)');

%% ---------- Figure 3: Payload swing angle and rate ----------
figure;

subplot(2,1,1);
plot(T, X(:,5), 'LineWidth', 1.3);
ylabel('\theta [rad]');
xlabel('time [s]');
title('Payload Swing Angle \theta');
grid on;

subplot(2,1,2);
plot(T, X(:,6), 'LineWidth', 1.3);
ylabel('\dot{\theta} [rad/s]');
xlabel('time [s]');
title('Payload Swing Angular Rate \dot{\theta}');
grid on;

sgtitle('Suspended Payload Swing States');

%% ---------- Control inputs (keep as separate figure) ----------
figure;

subplot(3,1,1);
plot(T, U(:,1), 'LineWidth', 1.3);
ylabel('v_1');
xlabel('time [s]');
title('Control Input v_1 (horizontal force proxy)');
grid on;

subplot(3,1,2);
plot(T, U(:,2), 'LineWidth', 1.3);
ylabel('v_2');
xlabel('time [s]');
title('Control Input v_2 (vertical force proxy)');
grid on;

subplot(3,1,3);
plot(T, U(:,3), 'LineWidth', 1.3);
ylabel('\tau [N\cdot m]');
xlabel('time [s]');
title('Control Input \tau (roll torque)');
grid on;

sgtitle('Control Inputs');

% Final state error vs reference
final_error = norm(X(end,:)' - x_ref_full(T(end)));
fprintf("Final state error (norm): %.4f\n", final_error);

% Save data
save('uav_payload_results.mat','T','X','U');

%% ============================================================
%                   Dynamics Function
% =============================================================
function dx = dynamics(~,x,v)
    % System parameters (must match main script)
    M    = 1.5;
    m    = 0.3;
    l    = 0.75;
    g    = 9.81;
    Jphi = 0.01;

    % States
    y        = x(1);  % unused directly, but kept for completeness
    ydot     = x(2);
    z        = x(3);  
    zdot     = x(4);
    theta    = x(5);
    thetadot = x(6);
    phi      = x(7);  
    phidot   = x(8);

    % Inertia matrix M(Xm)
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

    % Coriolis/gravity vector h(Xm, Xdot_m)
    h1 = -m*l*thetadot^2*sin(theta);
    h2 =  m*l*thetadot^2*cos(theta);
    h3 =  m*g*l*sin(theta);
    h  = [h1; h2; h3];

    % Virtual forces v = [v1; v2; tau]
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

    % State Jacobian
    for i = 1:n
        xp = x;
        xp(i) = xp(i) + eps;
        fp = dynamics(0,xp,u);
        A(:,i) = (fp - fx)/eps;
    end

    % Input Jacobian
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
    % Saturate v1 (symmetric)
    u_sat(1) = max(min(u_sat(1), f_max), -f_max);
    % Saturate v2 (symmetric)
    u_sat(2) = max(min(u_sat(2), f_max), -f_max);
    % Saturate tau
    u_sat(3) = max(min(u_sat(3), tau_max), tau_min);
end
