%% Continuous Reservoir Computing with RK4 Discretization for Lorenz system
clear; clc; close all;
rng shuffle
% Parameters
dt = 1e-2;           % Time step for Runge-Kutta
t_end = 7000;        % Runtime
T = 1;               % Choosing every T points
Nodes = 10;          % Number of reservoir nodes
lambda = 1e-4;       % Regularization parameter

% plotting parameter
lable_fs=35;
title_fs=15;

DD=[];DE=[];e_tr=[];e_val=[];e_ts=[];
LLE1=[];KL1=[];
% Prepare RK4 time vector
t_RK4 = 0:dt:t_end; % RK4 time vector
n_RK4 = length(t_RK4);

% Training/Validation/Testing specifications
Ktr = 10000;        % Training time
start_tr = 10000;   % Burn-in for training
Kval = 5000;        % Validation time
start_val = 10000;  % Burn-in for validation
auto_len = 30000;   % Multistep prediction length

% Select every T data point
T_step = 1:T:(t_end / dt);


% Generate Lorenz input data
dim =3;
nens=1;  % different realizations

for ens=1:nens
   ens
U1= func_generate_data_lorenz(700000);
U2=normalize(U1(:,1));
U3=normalize(U1(:,2));
U4=normalize(U1(:,3));
U5=[];
U5=[U2,U3,U4];
U=U5';



dim_rc = 4; % RC_dimension
% Initialize RC variables
Xt0 = zeros(dim_rc*Nodes, 1);   % Initial condition
Xt1 = zeros(dim_rc*Nodes, 1);   % Placeholder for updated state
w0_in = 2; % input weights amplitude
W_in = w0_in*(2 * rand(Nodes, 3) - 1); % Input weights (-1 to 1 for 3D Lorenz)


%% uncoupled reservoir
eig_rho=-0.5;
alpha_max = eig_rho;         % maximum alpha
alphas = rand(Nodes,1) * eig_rho;   % random alphas in (0, 0.9]
alphas(1) = alpha_max;            % explicitly set first entry = 0.9
A = diag(alphas); 

% A = eye(Nodes);   %special case all diagonal entries same
% alpha = -0.5;
% A = alpha * A;
% % max(eig(A))

%% coupled reservoir
% degree = 0.4;            % Sparsity degree for reservoir weights
% A = sprandsym(Nodes, degree); % Reservoir adjacency matrix
% alpha = -0.4;
% eig_A = eig(A);
% A = A + (alpha-max(real(eig_A)))*eye(Nodes);
% eig_A = eig(A);
% A = (0.9/max(abs(eig_A)))*A;
% eig_A_new = eig(A);

%%
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

% Plotting Training
figure
plot(target_training(1,:),'b','DisplayName', 'True Lorenz X',LineWidth=3);
hold on
plot(predictions_training(:,1),'r', 'DisplayName', 'Predicted Lorenz X',LineWidth=2)
set(gca,'LineWidth',2,'FontSize',18,'FontWeight','normal');
title('Open-Loop (Trainig) Predictions',FontSize=title_fs);
xlabel('Time','Interpreter','latex','FontSize',lable_fs)
ylabel('u','Interpreter','latex','FontSize',lable_fs)
xlim([4000 5000])
% 
% 
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
NRMSE_val = mean(vecnorm((target_val - predictions_val')./std(target_val,0,2),2,1)) % Normalized validation Error


% Plotting Validation
figure
plot(target_val(1,:),'b','DisplayName', 'True Lorenz X',LineWidth=3);
hold on
plot(predictions_val(:,1),'r', 'DisplayName', 'Predicted Lorenz X',LineWidth=2)
set(gca,'LineWidth',2,'FontSize',18,'FontWeight','normal');
title('Open-Loop (Validation) Predictions',FontSize=title_fs);
xlabel('Time','Interpreter','latex','FontSize',lable_fs)
ylabel('u','Interpreter','latex','FontSize',lable_fs)
xlim([4000 5000])

%% Testing: Multistep prediction setup
G=1;
auto_len=30000;
Lor_in=U(:,start_val+2*Ktr+Kval+1)';
auto_Lor=zeros(auto_len,dim);

Xt0 = Xt1;
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
 NRMSE_ts = mean(vecnorm((target_ts - auto_Lor')./std(target_training,0,2),2,1)); % Normalized training Error
     

target_ts1=target_ts';
DD=[];
DD=D_value(target_ts1,auto_Lor,30000);
DE=[DE;DD]


lle_target = Lyap(target_ts1);
lle_RC = Lyap(auto_Lor);

LLE = [lle_target, lle_RC];
LLE1=[LLE1;LLE]

[kl_div,hellinger_dist,tv_div] = div_Rc(target_ts1(end-10000:end,:),auto_Lor(end-10000:end,:));
KL1=[KL1;kl_div]


 e_tr=[e_tr;NRMSE_tr]
 e_val=[e_val;NRMSE_val]
 e_ts=[e_ts;NRMSE_ts]

end
 save('error_tr_u_10.mat','e_tr')
 save('error_val_10.mat',"e_val")
 save('error_ts_10.mat',"e_ts")
 save('LLE_u_10.mat','LLE1')
 save('KL_u_10.mat',"KL1")
 save('DE_u_10.mat',"DE")
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
xlim([4000 5000])
xlim([1 5000])

%%
figure
plot(target_ts(1,:),target_ts(2,:),'.b','DisplayName', 'True Lorenz');
hold on
plot(auto_Lor(:,1),auto_Lor(:,2),'.r', 'DisplayName', 'Predicted Lorenz')
set(gca,'LineWidth',2,'FontSize',18,'FontWeight','normal');
title('Close-Loop (testing) Predictions',FontSize=title_fs);
xlabel('Time','Interpreter','latex','FontSize',lable_fs)
ylabel('u','Interpreter','latex','FontSize',lable_fs)
