%% Continuous Reservoir Computing with RK4 Discretization for chaotic Foodchain system
clear; clc; close all;

% Parameters
dt = 1e-2;         % Time step for Runge-Kutta
t_end = 7000;      % Runtime
T = 1;             % Choosing every T points
%Nodes = 30;       % Number of reservoir nodes
lambda = 10^(-5);  % Regularization parameter
Nodes = 30;
% plotting parameter
lable_fs=35;
title_fs=15;


% Prepare RK4 time vector
t_RK4 = 0:dt:t_end; % RK4 time vector
n_RK4 = length(t_RK4);

% Training/Validation/Testing specifications
Ktr = 100000;        % Training time
start_tr = 50000;   % Burn-in for training
Kval = 5000;        % Validation time
start_val = 10000;  % Burn-in for validation
auto_len = 10000;   % Multistep prediction length

% Select every T data point
T_step = 1:T:(t_end / dt);


DD=[];DE=[];e_tr=[];e_val=[];e_ts=[];
LLE1=[];KL1=[];

% Generate Lorenz input data
dim =3;
%Xt1 = zeros(dim_rc*Nodes, 1);   % Placeholder for updated state
w0_in = 4; % input weights amplitude
attempts = 0;  %different realizations
nens=1; %
for ens=1:nens
    ens
    U=[];
k1=0.98;
data=foodchain_sim(k1);
U1=data';

  U=U1;
      Xt0=zeros(4*Nodes,1);
      Xt1=zeros(4*Nodes,1);
      W_in = w0_in*(2 * rand(Nodes, dim) - 1); % Input weights (-1 to 1 for 3D Lorenz)
  
%% uncoupled reservoir
eig_rho=-0.7;
alpha_max = eig_rho;         % maximum alpha
alphas = rand(Nodes,1) * eig_rho;   % random alphas in (0, 0.9]
alphas(1) = alpha_max;            % explicitly set first entry = 0.9
A = diag(alphas); 

% A = eye(Nodes);  %special case all diagonal entries same
% alpha = -0.7;
% A = alpha * A;
% max(eig(A));

%% coupled reservoir
%  eig_rho=0.2;
% degree = 0.4;               % Sparsity degree for reservoir weights
% A = sprandsym(Nodes, degree); % Reservoir adjacency matrix
% alpha = -0.8;
% eig_A = eig(A);
% A = A + (alpha-max(real(eig_A)))*eye(Nodes);
% eig_A = eig(A);
% A = (eig_rho/max(abs(eig_A)))*A;
% eig_A_new = eig(A);

%%
RC_state_mat_all1 = ones(length(T_step), Nodes); % RC State matrix by adding bias term to the last column
x_me = ones(length(T_step), 4*Nodes); % RC State matrix by adding bias term to the last column


func_name = 'SSCFHN'; % tanhODE, PolyODE, FHN
% fun = @(x,u) PolyODE(x, u, A, W_in);
% fun = @(x,u) tanhODE(x, u, A, W_in);
% fun = @(x,u) FHN(x, u, A, W_in,Nodes);
fun = @(x,u) SSCFHN(x, u, A, W_in,Nodes);

U_t=[];
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
    x_me(i, :)= Xt1(1:4*Nodes)';


    Xt0 = Xt1; % Update state for the next iteration
end

% RC_state_mat_all2 =RC_state_mat_all;
RC_state_mat_all2= RC_state_mat_all1.^2;
RC_state_mat_all=[RC_state_mat_all1,RC_state_mat_all2];


%% Training

% Training data  
training_RCdata = RC_state_mat_all(start_tr:(start_tr+Ktr-1),:);

% output corresponding to the training data
target_training = zeros(dim,Ktr);
for i=1:Ktr
    target_training(:,i)=U(:,T_step(start_tr+i)); %true output
end

% Finding reservoir output weights using ridge regression
W_out = (training_RCdata' * training_RCdata + lambda * eye(2*Nodes)) \ (training_RCdata' * target_training');


% Calculating Training Error
predictions_training = training_RCdata*W_out;
NRMSE_tr = mean(vecnorm((target_training - predictions_training')./std(target_training,0,2),2,1)); % Normalized training Error

% % Plotting Training
figure
plot(target_training(1,:),'b','DisplayName', 'True Lorenz X',LineWidth=3);
hold on
plot(predictions_training(:,1),'r', 'DisplayName', 'Predicted Lorenz X',LineWidth=2)
set(gca,'LineWidth',2,'FontSize',18,'FontWeight','normal');
title('Open-Loop (Trainig) Predictions',FontSize=title_fs);
xlabel('Time','Interpreter','latex','FontSize',lable_fs)
ylabel('u','Interpreter','latex','FontSize',lable_fs)
xlim([0 5000])


%% Validation: Onestep prediction setup

% Validation data extraction 
Validation_RCdata = RC_state_mat_all((start_val+2*Ktr):(start_val+2*Ktr+Kval-1),:);

% output corresponding to the Validation data
target_val=zeros(dim,Kval);
for i=1:Kval
    target_val(:,i)=U(:,T_step(start_val+2*Ktr+i));
end

% Calculating Validation Error
predictions_val = Validation_RCdata*W_out;
NRMSE_val = mean(vecnorm((target_val- predictions_val')./std(target_val,0,2),2,1)); % Normalized training Error

% Plotting Validation
figure
plot(target_val(1,:),'b','DisplayName', 'True Lorenz X',LineWidth=3);
hold on
plot(predictions_val(:,1),'r', 'DisplayName', 'Predicted Lorenz X',LineWidth=2)
set(gca,'LineWidth',2,'FontSize',18,'FontWeight','normal');
title('Open-Loop (Validation) Predictions',FontSize=title_fs);
xlabel('Time','Interpreter','latex','FontSize',lable_fs)
ylabel('u','Interpreter','latex','FontSize',lable_fs)
xlim([0 500])

%% Testing: Multistep prediction setup
G=1;
auto_len=10000;
Lor_in=U(:,start_val+2*Ktr+Kval+1)';
auto_Lor=zeros(auto_len,dim);

% Xt0 = Xt1;
Xt0=x_me(start_val+2*Ktr+Kval+1,:)';
for i = 1:auto_len

    U_t = Lor_in'; % Input at current time step

    k1 = fun(Xt0, U_t);
    k2 = fun(Xt0 + 0.5 * dt * k1, U_t);
    k3 = fun(Xt0 + 0.5 * dt * k2, U_t);
    k4 = fun(Xt0 + dt * k3, U_t);

    Xt1 = Xt0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4); % Update state

    % output
    Xt12 = Xt1.^2;
    Lor_in=[Xt1(2*Nodes+1:3*Nodes);Xt12(2*Nodes+1:3*Nodes)]'*W_out;
    auto_Lor(i,:)=Lor_in';
    Xt0 = Xt1; % Update state for the next iteration
end
% output corresponding to the testing data
target_ts=zeros(dim,auto_len);
for i=1:auto_len
    target_ts(:,i)=U(:,start_val+2*Ktr+Kval+i+1);
end
NRMSE_ts = mean(vecnorm((target_ts - auto_Lor')./std(target_ts,0,2),2,1)); % Normalized training Error
target_ts1=target_ts';
DD=[];
DD=D_value(target_ts1,auto_Lor,10000);
DE=[DE;DD]



lle_target = Lyap(target_ts1);
lle_RC = Lyap(auto_Lor);

LLE = [lle_target, lle_RC];
LLE1=[LLE1;LLE]

[kl_div,hellinger_dist,tv_div] = div_Rc(target_ts1(end-1000:end,:),auto_Lor(end-1000:end,:));
KL1=[KL1;kl_div]

e_tr=[e_tr;NRMSE_tr]
e_val=[e_val;NRMSE_val]

end
  save('LLE_u_30.mat','LLE1')
  save('KL_u_30.mat',"KL1")
  save('metric1_f_u_30.mat','e_tr')
  save('metric2_f_u_30.mat',"e_val")
  save('metric3_f_u_30.mat',"DE")
% 
% 

% Plotting Testing result
figure(Position=[436,459,560,420])
plot(target_ts(1,:),'b','DisplayName', 'True Lorenz X',LineWidth=3);
hold on
plot(auto_Lor(:,1),'r', 'DisplayName', 'Predicted Lorenz X',LineWidth=2)
set(gca,'LineWidth',2,'FontSize',18,'FontWeight','normal');
title('Close-Loop (testing) Predictions',FontSize=title_fs);
xlabel('Time','Interpreter','latex','FontSize',lable_fs)
ylabel('u','Interpreter','latex','FontSize',lable_fs)
xlim([0 5000])

%%
figure
plot(target_ts(1,:),target_ts(2,:),'.b','DisplayName', 'True Lorenz');
hold on
plot(auto_Lor(:,1),auto_Lor(:,2),'.r', 'DisplayName', 'Predicted Lorenz')
set(gca,'LineWidth',2,'FontSize',18,'FontWeight','normal');
title('Close-Loop (testing) Predictions',FontSize=title_fs);
xlabel('Time','Interpreter','latex','FontSize',lable_fs)
ylabel('u','Interpreter','latex','FontSize',lable_fs)
% 
