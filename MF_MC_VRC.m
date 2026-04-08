
%% Memory function and memory capacity calculation for VRC with iid input
clear; clc; close all;
% Parameters
dt = 1e-2;           % Time step for Runge-Kutta
t_end = 7000;        % Runtime
T = 1;               % Choosing every T points
Nodes = 30;          % Number of reservoir nodes

% plotting parameter
lable_fs=35;
title_fs=15;
Mf_ens=[];MC_ens=[];Emat=[];
U_ens=[];
% Prepare RK4 time vector
t_RK4 = 0:dt:t_end; % RK4 time vector
n_RK4 = length(t_RK4);
T_step = 1:T:(t_end / dt);
count=1;
for ens=1:count
    ens
U=rand(700001,1); %iid input
U=U';
dim_rc = 4; % RC_dimension
w0_in = 2; % input weights amplitude
W_in = w0_in*(2 * rand(Nodes, 1) - 1); % Input weights (-1 to 1 for 3D Lorenz)
degree = 0.4; 

%coupled reservoir
% eig_rho=0.8;% Sparsity degree for reservoir weights
% A = sprandsym(Nodes, degree); % Reservoir adjacency matrix
% alpha = -0.8;
% eig_A = eig(A);
% A = A + (alpha-max(real(eig_A)))*eye(Nodes);
% eig_A = eig(A);
% A = (eig_rho/max(abs(eig_A)))*A;
% eig_A_new = eig(A);
% Emat=[Emat;eig_A_new];

%uncoupled reservoir

degree = 0.4; 
eig_rho=-0.8;
alpha_max = eig_rho;         % maximum alpha
alphas = rand(Nodes,1) * eig_rho;   % random alphas in (0, 0.9]
alphas(1) = alpha_max;            % explicitly set first entry = 0.9
A = diag(alphas);                 % build diagonal reservoir matrix
eig_A_new = eig(A);

% Initialize RC variables
Xt0 = zeros(dim_rc*Nodes, 1);   % Initial condition
Xt1 = zeros(dim_rc*Nodes, 1);   % Placeholder for updated state

RC_state_mat_all1 = ones(length(T_step), Nodes); % RC State matrix by adding bias term to the last column
x_me = ones(length(T_step), Nodes); % RC State matrix by adding bias term to the last column


func_name = 'SSCFHN'; % tanhODE, PolyODE, FHN
% fun = @(x,u) PolyODE(x, u, A, W_in);
% fun = @(x,u) tanhODE(x, u, A, W_in);
% fun = @(x,u) FHN(x, u, A, W_in,Nodes);
fun = @(x,u) SSCFHN(x, u, A, W_in,Nodes);


K=1; % scaling coefficient
% Reservoir Computing with RK4 Integration
for i = 1:length(T_step)
    
    U_t = U(:, T_step(i)); % Input at current time step

    k1 = fun(Xt0, U_t);
    k2 = fun(Xt0 + 0.5 * dt * k1, U_t);
    k3 = fun(Xt0 + 0.5 * dt * k2, U_t);
    k4 = fun(Xt0 + dt * k3, U_t);

    Xt1 = Xt0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4); % Update state

    % Store the state in the full state matrix
    % RC_state_mat_all1(i, :) = K*Xt1(Nodes+1:end)';
    % RC_state_mat_all1(i, :) = K*Xt1(1:Nodes)';
    RC_state_mat_all1(i, :) = K*Xt1(2*Nodes+1:3*Nodes)';
    x_me(i, :)= Xt1(1:1*Nodes)';


    Xt0 = Xt1; % Update state for the next iteration
end

% RC_state_mat_all2 =RC_state_mat_all;
RC_state_mat_all2= RC_state_mat_all1.^2;
RC_state_mat_all=[RC_state_mat_all1,RC_state_mat_all2];
%RC_state_mat_all1(2:2:end,:)=RC_state_mat_all1(2:2:end,:).^2;


%% Training and testing data length
dl_Tr=24999;dl_Ts=4999;ts_L=5000;

% Training data  
data=U(10000:39999);  % we choose 30000 points for our analysis from the iid data and split into training and testing
training_RCdata = RC_state_mat_all(10001:40000,:); % rc states considered for delayed past input 

X_train = training_RCdata(2:dl_Tr+1,:);   % (T_eff x N)
U_train = data(1:dl_Tr); % (T_eff x 1)
T_eff_train = size(X_train,1);

X_test = training_RCdata(dl_Tr+2:dl_Tr+ts_L+1,:);    % (T_eff x N)
U_test = data(dl_Tr+1:dl_Tr+ts_L);   % (T_eff x 1)
T_eff_test = size(X_test,1);
tau_max=200;
% -------- Train & Evaluate Memory function --------
Mf_tau = zeros(tau_max+1,1);
ridge=10^-9;
for tau = 0:tau_max
    if tau >= T_eff_train || tau >= T_eff_test
        break;
    end

    % Training alignment
    X_tau_train = X_train(1+tau:end,:);
    Y_train     = U_train(1:end-tau);

    % Train weights (ridge regression)
    Wout = (X_tau_train' * X_tau_train + ridge * eye(2*Nodes)) \ ...
           (X_tau_train' * Y_train');

    % Testing alignment
    X_tau_test = X_test(1+tau:end, :);
    Y_test     = U_test(1:end-tau);

    % Predict on test data
    yhat = X_tau_test * Wout;

    % Compute R^2
    Yc = Y_test - mean(Y_test);
    yhc = yhat - mean(yhat);
    denom = (Yc* Yc') * (yhc' * yhc);
    if denom <= eps
        R2 = 0;
    else
        R2 = (Yc * yhc)^2 / denom;
%         R2 = max(0, min(1, real(R2)));
    end

    Mf_tau(tau+1) = R2;
end
Mf_ens=[Mf_ens,Mf_tau]; %memory function across ensemble
end
MC_ens=sum(Mf_ens,1);   %memory capacity across ensemble
save MF_30_L_uc_random.mat Mf_ens -mat
save MC_30_L_uc_random.mat MC_ens -mat
 