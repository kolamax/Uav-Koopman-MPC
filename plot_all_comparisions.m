% ============================================================
%  COMPARISON PLOTS: LQR vs KOOPMAN STUDY (EXP SET)
% ============================================================

clear; clc; close all;

%% ============================================================
% Load experiments
% ============================================================
exp1 = load('exp_lqr_mpc_nominal.mat').data;
exp2 = load('exp_koopman_nominal.mat').data;
exp3 = load('exp_lqr_wind.mat').data;
exp4 = load('exp_koopman_wind.mat').data;
exp5 = load('exp_koopman_mismatch.mat').data;

%% ============================================================
% Extract
% ============================================================
T1 = exp1.T; X1 = exp1.X;
T2 = exp2.T; X2 = exp2.X;
T3 = exp3.T; X3 = exp3.X;
T4 = exp4.T; X4 = exp4.X;
T5 = exp5.T; X5 = exp5.X;

%% ============================================================
% Reference trajectory
% ============================================================
T_ref = linspace(0, 60, 1000)';

A_horiz = 1.5;
f_horiz = 0.1;
z0 = 1.0;
slope = 0.02;
vx = 0.5;

Xref = vx * T_ref;
Yref = A_horiz * sin(2*pi*f_horiz*T_ref);
Zref = z0 + slope * T_ref;

%% ============================================================
% ============================================================
% EXP 1: NOMINAL LQR vs KOOPMAN
% ============================================================
%% ============================================================

% ---------------- 3D TRAJECTORY ----------------
figure('Name','Nominal 3D','Color','w');
plot3(X1(:,1),X1(:,3),X1(:,5),'b','LineWidth',1.2); hold on;
plot3(X2(:,1),X2(:,3),X2(:,5),'r','LineWidth',1.2);
plot3(Xref,Yref,Zref,'k--');
grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
title('Nominal 3D Trajectory');
legend('LQR-MPC','Koopman','Reference');

% ---------------- STATES ----------------
figure('Name','Nominal States','Color','w');
subplot(3,1,1);
plot(T1,X1(:,1),'b',T2,X2(:,1),'r',T_ref,Xref,'k--'); grid on;
ylabel('X');

subplot(3,1,2);
plot(T1,X1(:,3),'b',T2,X2(:,3),'r',T_ref,Yref,'k--'); grid on;
ylabel('Y');

subplot(3,1,3);
plot(T1,X1(:,5),'b',T2,X2(:,5),'r',T_ref,Zref,'k--'); grid on;
ylabel('Z'); xlabel('t');

% ---------------- SWING ----------------
figure('Name','Nominal Swing','Color','w');
subplot(2,1,1);
plot(T1,X1(:,7),'b',T2,X2(:,7),'r'); grid on;
ylabel('\theta_x');

subplot(2,1,2);
plot(T1,X1(:,9),'b',T2,X2(:,9),'r'); grid on;
ylabel('\theta_y'); xlabel('t');


%% ============================================================
% EXP 2: WIND LQR vs KOOPMAN
% ============================================================

figure('Name','Wind 3D','Color','w');
plot3(X3(:,1),X3(:,3),X3(:,5),'b','LineWidth',1.2); hold on;
plot3(X4(:,1),X4(:,3),X4(:,5),'r','LineWidth',1.2);
plot3(Xref,Yref,Zref,'k--');
grid on;
title('Wind Disturbance 3D');
legend('LQR Wind','Koopman Wind','Reference');

figure('Name','Wind States','Color','w');
subplot(3,1,1);
plot(T3,X3(:,1),'b',T4,X4(:,1),'r',T_ref,Xref,'k--'); grid on;
ylabel('X');

subplot(3,1,2);
plot(T3,X3(:,3),'b',T4,X4(:,3),'r',T_ref,Yref,'k--'); grid on;
ylabel('Y');

subplot(3,1,3);
plot(T3,X3(:,5),'b',T4,X4(:,5),'r',T_ref,Zref,'k--'); grid on;
ylabel('Z'); xlabel('t');

figure('Name','Wind Swing','Color','w');
subplot(2,1,1);
plot(T3,X3(:,7),'b',T4,X4(:,7),'r'); grid on;
ylabel('\theta_x');

subplot(2,1,2);
plot(T3,X3(:,9),'b',T4,X4(:,9),'r'); grid on;
ylabel('\theta_y'); xlabel('t');


%% ============================================================
% EXP 3: KOOPMAN NOMINAL vs MISMATCH
% ============================================================

figure('Name','Koopman Robust 3D','Color','w');
plot3(X2(:,1),X2(:,3),X2(:,5),'b','LineWidth',1.2); hold on;
plot3(X5(:,1),X5(:,3),X5(:,5),'r','LineWidth',1.2);
plot3(Xref,Yref,Zref,'k--');
grid on;
title('Koopman Robustness (Nominal vs Mismatch)');
legend('Nominal','Mismatch','Reference');

figure('Name','Koopman Robust States','Color','w');
subplot(3,1,1);
plot(T2,X2(:,1),'b',T5,X5(:,1),'r',T_ref,Xref,'k--'); grid on;
ylabel('X');

subplot(3,1,2);
plot(T2,X2(:,3),'b',T5,X5(:,3),'r',T_ref,Yref,'k--'); grid on;
ylabel('Y');

subplot(3,1,3);
plot(T2,X2(:,5),'b',T5,X5(:,5),'r',T_ref,Zref,'k--'); grid on;
ylabel('Z'); xlabel('t');

figure('Name','Koopman Robust Swing','Color','w');
subplot(2,1,1);
plot(T2,X2(:,7),'b',T5,X5(:,7),'r'); grid on;
ylabel('\theta_x');

subplot(2,1,2);
plot(T2,X2(:,9),'b',T5,X5(:,9),'r'); grid on;
ylabel('\theta_y'); xlabel('t');

%% ============================================================
% ERROR METRICS
% ============================================================
rmse = @(X,T) sqrt(mean((X(:,1) - interp1(T_ref, Xref, T)).^2));

fprintf('\n===== RMSE (X tracking) =====\n');
fprintf('LQR nominal      : %.4f\n', rmse(X1,T1));
fprintf('Koopman nominal  : %.4f\n', rmse(X2,T2));
fprintf('LQR wind         : %.4f\n', rmse(X3,T3));
fprintf('Koopman wind     : %.4f\n', rmse(X4,T4));
fprintf('Koopman mismatch : %.4f\n', rmse(X5,T5));
