% ========================================================================
% Koopman MPC for 3D UAV + Suspended Payload
%
% Author: Alexandre Claux & Maximus Kolavennu
% ME 569 Project
%
% This script:
% 1. Simulates nonlinear UAV-payload dynamics
% 2. Generates Koopman training data
% 3. Learns lifted linear dynamics using EDMD
% 4. Runs Koopman MPC trajectory tracking
%
% ========================================================================

clear;
clc;
close all;

%% =======================================================================
% SYSTEM PARAMETERS
% ========================================================================

M = 1.5;
m = 0.3;
l = 0.75;
g = 9.81;

Fmax = 50;
Fmin = -50;

dt = 0.05;
tf = 60;

%% =======================================================================
% INITIAL CONDITION
% ========================================================================

x0 = [0;
      0;
      0;
      0;
      1.0;
      0;
      0.05;
      0;
      0.03;
      0];

%% =======================================================================
% TRAJECTORY PARAMETERS
% ========================================================================

A_horiz = 1.5;
f_horiz = 0.1;

z0 = 1.0;
slope = 0.02;

xref_fun = @(t) compute_reference_3D(t,A_horiz,f_horiz,z0,slope);

%% =======================================================================
% MPC PARAMETERS
% ========================================================================

N = 20;

Qx = diag([200 20 ...
           200 20 ...
           200 20 ...
           3000 600 ...
           3000 600]);

R = 0.1*diag([1 1 1]);

%% =======================================================================
% TRAINING DATA GENERATION
% ========================================================================

disp('Generating training data...')

Ndata = 15000;

Xtrain = zeros(10,Ndata);
Ytrain = zeros(10,Ndata);
Utrain = zeros(3,Ndata);

x = zeros(10,1);

u_hover = [0;0;(M+m)*g];

for k = 1:Ndata

    % ---------------------------------------------------------------
    % bounded random inputs around hover
    % ---------------------------------------------------------------

    du = [
        4*(2*rand-1);
        4*(2*rand-1);
        2*(2*rand-1)
    ];

    u = u_hover + du;

    % ---------------------------------------------------------------
    % keep payload angles small
    % ---------------------------------------------------------------

    x(7) = max(min(x(7),0.35),-0.35);
    x(9) = max(min(x(9),0.35),-0.35);

    dx = dynamics_3D(x,u);

    xnext = x + dt*dx;

    % reject exploding trajectories
    if norm(xnext) > 15 || any(isnan(xnext)) || any(isinf(xnext))

        x = 0.05*randn(10,1);

        continue

    end

    Xtrain(:,k) = x;
    Ytrain(:,k) = xnext;
    Utrain(:,k) = du;

    x = xnext;

end

%% =======================================================================
% LIFT TRAINING DATA
% ========================================================================

disp('Lifting data...')

Z = [];
Znext = [];

for k = 1:Ndata

    z1 = lift_state(Xtrain(:,k));
    z2 = lift_state(Ytrain(:,k));

    Z = [Z z1];
    Znext = [Znext z2];

end

nz = size(Z,1);

%% =======================================================================
% LEARN KOOPMAN MODEL
% ========================================================================

disp('Training Koopman model...')

W = [Z;
     Utrain];

lambda = 1e-6;

AB = Znext * W' / (W*W' + lambda*eye(size(W,1)));

Akoop = AB(:,1:nz);
Bkoop = AB(:,nz+1:end);

disp('Koopman model trained.')

%% =======================================================================
% PHYSICAL STATE EXTRACTION MATRIX
% ========================================================================

nx = 10;

Ckoop = zeros(nx,nz);
Ckoop(:,1:nx) = eye(nx);

%% =======================================================================
% BUILD PREDICTION MATRICES
% ========================================================================

[Sz, Su] = build_prediction_matrices(Akoop,Bkoop,N);

%% =======================================================================
% COST MATRICES
% ========================================================================

Qbar = kron(eye(N),Qx);
Rbar = kron(eye(N),R);

Qkoop = blkdiag(kron(eye(N),Ckoop'*Qx*Ckoop));

%% =======================================================================
% SIMULATION SETUP
% ========================================================================

tlog = [];
xlog = [];
ulog = [];

x = x0;

t = 0;

odeopts = odeset('RelTol',1e-6,'AbsTol',1e-8);

%% =======================================================================
% MAIN KOOPMAN MPC LOOP
% ========================================================================

disp('Running Koopman MPC...')

while t < tf

    z = lift_state(x);

    % ---------------------------------------------------------------
    % Build reference trajectory
    % ---------------------------------------------------------------

    Xref = zeros(nx*N,1);

    for k = 0:N-1

        tref = t + k*dt;

        xr = xref_fun(tref);

        Xref(k*nx+1:(k+1)*nx) = xr;

    end

    % ---------------------------------------------------------------
    % MPC QP
    % ---------------------------------------------------------------

    H = Su' * Qkoop * Su + Rbar;
    H = 0.5*(H + H');

    f = Su' * Qkoop * (Sz*z - kron(ones(N,1),zeros(nz,1)));

    % physical tracking correction
    pred_offset = zeros(nx*N,1);

    for k = 1:N

        rowsx = (k-1)*nx+1:k*nx;
        rowsz = (k-1)*nz+1:k*nz;

        pred_offset(rowsx) = Ckoop * Sz(rowsz,:) * z;

    end

    f = Su'*Su*0;

    for k = 1:N

        rowsx = (k-1)*nx+1:k*nx;
        rowsu = (k-1)*3+1:k*3;
        rowsz = (k-1)*nz+1:k*nz;

        CzSu = Ckoop * Su(rowsz,:);

        H = H + CzSu'*Qbar(rowsx,rowsx)*CzSu;

        f = f + CzSu'*Qbar(rowsx,rowsx)*(Ckoop*Sz(rowsz,:)*z - Xref(rowsx));

    end

    Umin = Fmin*ones(3*N,1);
    Umax = Fmax*ones(3*N,1);

    U0 = zeros(3*N,1);

    Uopt = projected_gradient_qp(H,f,Umin,Umax,U0);

    u = Uopt(1:3);

    u(3) = u(3) + (M+m)*g;

    u = saturate_u_3D(u,Fmin,Fmax);

    % ---------------------------------------------------------------
    % Integrate nonlinear dynamics
    % ---------------------------------------------------------------

    dyn = @(tt,xx) dynamics_3D(xx,u);

    [Tstep,Xstep] = ode45(dyn,[t t+dt],x,odeopts);

    % log states
    tlog = [tlog; Tstep(1:end-1)];
    xlog = [xlog; Xstep(1:end-1,:)];
    
    % log controls
    urow = reshape(u,1,3);

    ulog = [ulog;
            repmat(urow,length(Tstep)-1,1)];

    x = Xstep(end,:)';

    t = t + dt;

end

tlog = [tlog; Tstep(end)];
xlog = [xlog; Xstep(end,:)];
ulog = [ulog; reshape(u,1,3)];

disp('Simulation complete.')

%% =======================================================================
% PLOTS
% ========================================================================

T = tlog;
X = xlog;
U = ulog;

Tref = linspace(0,tf,2000)';

Xref = zeros(length(Tref),10);

for k = 1:length(Tref)

    xr = xref_fun(Tref(k));

    Xref(k,:) = xr';

end

figure;
plot3(X(:,1),X(:,3),X(:,5),'b','LineWidth',1.5)
hold on
plot3(Xref(:,1),Xref(:,3),Xref(:,5),'r--','LineWidth',1.5)
grid on
xlabel('X')
ylabel('Y')
zlabel('Z')
title('3D Trajectory')
legend('Actual','Reference')

figure;

subplot(3,1,1)
plot(T,X(:,1),'b')
hold on
plot(Tref,Xref(:,1),'r--')
ylabel('X')

subplot(3,1,2)
plot(T,X(:,3),'b')
hold on
plot(Tref,Xref(:,3),'r--')
ylabel('Y')

subplot(3,1,3)
plot(T,X(:,5),'b')
hold on
plot(Tref,Xref(:,5),'r--')
ylabel('Z')
xlabel('Time')

sgtitle('Position Tracking')

figure;

subplot(2,1,1)
plot(T,X(:,7),'LineWidth',1.5)
ylabel('\theta_x')

subplot(2,1,2)
plot(T,X(:,9),'LineWidth',1.5)
ylabel('\theta_y')
xlabel('Time')

sgtitle('Payload Swing')

figure;

Nu = size(U,1);
Tu = T(1:Nu);

subplot(3,1,1)
plot(Tu,U(:,1),'LineWidth',1.5)
ylabel('F_x')
grid on

subplot(3,1,2)
plot(Tu,U(:,2),'LineWidth',1.5)
ylabel('F_y')
grid on

subplot(3,1,3)
plot(Tu,U(:,3),'LineWidth',1.5)
ylabel('F_z')
xlabel('Time')
grid on

sgtitle('Control Inputs')

%% =======================================================================
% ERROR METRICS
% ========================================================================

ex = rms(X(:,1) - interp1(Tref,Xref(:,1),T));
ey = rms(X(:,3) - interp1(Tref,Xref(:,3),T));
ez = rms(X(:,5) - interp1(Tref,Xref(:,5),T));

fprintf('\n');
fprintf('RMS Tracking Errors:\n');
fprintf('X Error: %.4f m\n',ex);
fprintf('Y Error: %.4f m\n',ey);
fprintf('Z Error: %.4f m\n',ez);

%% =======================================================================
% SAVE RESULTS
% ========================================================================

save('koopman_mpc_results.mat','T','X','U','Akoop','Bkoop')

%% =======================================================================
% REFERENCE TRAJECTORY
% ========================================================================

function xr = compute_reference_3D(t,A,f,z0,slope)

vx = 0.5;

x = vx*t;
y = A*sin(2*pi*f*t);
z = z0 + slope*t;

xdot = vx;
ydot = A*2*pi*f*cos(2*pi*f*t);
zdot = slope;

xr = [x;
      xdot;
      y;
      ydot;
      z;
      zdot;
      0;
      0;
      0;
      0];

end

%% =======================================================================
% STATE LIFTING
% ========================================================================

function z = lift_state(x)

z = [
    x;

    sin(x(7));
    cos(x(7));

    sin(x(9));
    cos(x(9))
];

end

%% =======================================================================
% NONLINEAR UAV + PAYLOAD DYNAMICS
% ========================================================================

function dx = dynamics_3D(x,u)

M = 1.5;
m = 0.1; % 67% lighter payload
l = 0.15; % 80% shorter cable
g = 9.81;

xpos = x(1);
xdot = x(2);

ypos = x(3);
ydot = x(4);

zpos = x(5);
zdot = x(6);

thx = x(7);
thx_dot = x(8);

thy = x(9);
thy_dot = x(10);

Fx = u(1);
Fy = u(2);
Fz = u(3);

Mmat = zeros(5,5);

Mmat(1,1) = M+m;
Mmat(1,4) = m*l*cos(thx)*cos(thy);

Mmat(2,2) = M+m;
Mmat(2,5) = m*l*cos(thy);

Mmat(3,3) = M+m;
Mmat(3,4) = -m*l*sin(thx)*cos(thy);
Mmat(3,5) = -m*l*cos(thx)*sin(thy);

Mmat(4,1) = Mmat(1,4);
Mmat(4,3) = Mmat(3,4);
Mmat(4,4) = m*l^2;

Mmat(5,2) = Mmat(2,5);
Mmat(5,3) = Mmat(3,5);
Mmat(5,5) = m*l^2*cos(thx)^2;

rhs = zeros(5,1);

rhs(1) = Fx;

rhs(2) = Fy;

rhs(3) = Fz - (M+m)*g;

rhs(4) = -m*g*l*sin(thx)*cos(thy);

rhs(5) = -m*g*l*cos(thx)*sin(thy);

acc = Mmat \ rhs;

xdd = acc(1);
ydd = acc(2);
zdd = acc(3);

thxdd = acc(4);
thydd = acc(5);

dx = [
    xdot;
    xdd;
    ydot;
    ydd;
    zdot;
    zdd;
    thx_dot;
    thxdd;
    thy_dot;
    thydd
];

end

%% =======================================================================
% BUILD PREDICTION MATRICES
% ========================================================================

function [Sx,Su] = build_prediction_matrices(A,B,N)

nx = size(A,1);
nu = size(B,2);

Sx = zeros(nx*N,nx);
Su = zeros(nx*N,nu*N);

for i = 1:N

    Sx((i-1)*nx+1:i*nx,:) = A^i;

    for j = 1:i

        Su((i-1)*nx+1:i*nx,...
           (j-1)*nu+1:j*nu) = A^(i-j)*B;

    end

end

end

%% =======================================================================
% INPUT SATURATION
% ========================================================================

function u = saturate_u_3D(u,umin,umax)

u = min(max(u,umin),umax);

end

%% =======================================================================
% PROJECTED GRADIENT QP SOLVER
% ========================================================================

function U = projected_gradient_qp(H,f,Umin,Umax,U0)

if norm(H,2) > 0
    alpha = 0.9/norm(H,2);
else
    alpha = 1e-3;
end

U = U0;

max_iter = 300;

for k = 1:max_iter

    grad = H*U + f;

    U = U - alpha*grad;

    U = min(max(U,Umin),Umax);

end

end

%% ============================================================
%                   SAVE EXPERIMENT DATA
% ============================================================
data.T = T;
data.X = X;
data.U = U;

data.params.M = M;
data.params.m = m;
data.params.l = l;
data.params.g = g;


data.meta.N = N;
data.meta.model = "KOOPMAN_MISMATCH";

save('exp_koopman_mismatch.mat', 'data');
