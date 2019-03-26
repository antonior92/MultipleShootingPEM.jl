%% ########## Controler design for pendulum ########## %%
% Constants
g = 9.8; l = 0.3; k = 2.0; m = 3.0; delta = 0.01;

%% Define linear version of:
% x1[k+1] = x1[k] + delta*x2[k]
% x2[k+1] = delta*g/l*sin(x1[k]) + (1-delta*k/m)*x2[k] + delta*1/m*u[k]

A = [1, delta; -delta*g/l (1-delta*k/m)];
B = [0; delta*1/m];
C = [1, 0];
D = [0];
Ts = delta;
sys = ss(A, B, C, D, Ts);
% Convert to transfer function
sys = tf(sys);

%% Define controler and analyse it
K = 3582;
z = [0.9457, 0.9092];
p = [0.0273, 1.0000];
ctrl = zpk(z, p, K, Ts);

sisotool(sys, ctrl)
