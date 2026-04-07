function [ts_train] = func_generate_data_lorenz(data_len)

lorenz_sigma = 10;
lorenz_rho = 29; % Lya 0.85
lorenz_beta = 8/3;
lorenz_params = [lorenz_sigma lorenz_rho lorenz_beta];

dt=0.01;
tspan=0:dt:(data_len)*dt;

x0 = [ 28*rand(1)-14; 30*rand(1)-15; 20*rand(1)];
[~,ts_train] = ode4(@(t,x) func_lorenz(t,x,lorenz_params), tspan, x0);

end

