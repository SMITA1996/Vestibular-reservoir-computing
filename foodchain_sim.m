function U1=foodchain_sim(k)
    k = 0.98; yc = 2.009; yp = 2.876;
    xc = 0.4; xp = 0.08;
    r0 = 0.16129; c0 = 0.5;

    dt = 0.1;
    Tmax = 80000;  % total simulation length
    transient = 3000; % discard initial transients
    
    % Time vector 
    t_all = 0:dt:(Tmax*10);
    t_all_sub = t_all(1:10:end); % equivalent to [::10]

    % Initialize
    Testing_data = {};
    z_dim_valid = false;
    attempts = 0;

    while ~z_dim_valid && attempts < 500
        fprintf('%d\n', attempts);

        % Random initial conditions
        x0 = [0.4*rand + 0.6, ...
              0.4*rand + 0.15, ...
              0.5*rand + 0.3];

        % Solve ODE system
        [~, sol] = ode45(@(t, x) func_foodchain(t, x, k, yc, yp, xc, xp, r0, c0), t_all, x0);

        % Subsample like Python [::10]
        sol_sub = sol(1:10:end, :);

        % Check condition on predator (x3)
        if min(sol_sub(transient+1:end, 3)) > 0.5
            z_dim_valid = true;
            Testing_data{end+1} = sol_sub(transient+1:end, :);
        else
            attempts = attempts + 1;
        end
    end
    U1=Testing_data{1};
    
