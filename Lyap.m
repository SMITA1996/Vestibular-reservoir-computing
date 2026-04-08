function lambda = Lyap(X)
X = X(:,1); % If multidimensional, take first component

%% Parameters
tau = 10;             % Embedding delay
m = 5;                % Embedding dimension
fs = 1;               % Sampling frequency (can set to 1 if unit time step)
max_t = 100;          % Max time steps to track divergence
min_separation = 50;  % Minimum temporal separation between neighbors

%% Step 1: Time-delay embedding
N = length(X) - (m - 1)*tau;
Y = zeros(N, m);
for i = 1:m
    Y(:,i) = X((1:N) + (i-1)*tau);
end

%% Step 2: Find nearest neighbors
neighbor_idx = zeros(N,1);
for i = 1:N
    dists = sqrt(sum((Y - Y(i,:)).^2, 2));
    lower_bound = max(1, i - min_separation);
    upper_bound = min(N, i + min_separation);
    dists(lower_bound : upper_bound) = inf;
    [~, idx] = min(dists);
    neighbor_idx(i) = idx;
end

%% Step 3: Calculate divergence over time
divergence = NaN(N, max_t);  % Use NaN to ignore invalid rows in mean()

for i = 1:N
    for j = 1:max_t
        if (i + j <= N) && (neighbor_idx(i) + j <= N)
            dist1 = Y(i + j, :) - Y(neighbor_idx(i) + j, :);
            divergence(i, j) = norm(dist1);
        end
    end
end

%% Step 4: Average and linear fit
mean_div = mean(divergence, 1, 'omitnan');
log_div = log(mean_div);

% Fit linear region (choose the linear part carefully!)
t = (1:max_t) / fs;
fit_range = 10:50; % Change based on your data
p = polyfit(t(fit_range), log_div(fit_range), 1);
lambda = p(1); % Largest Lyapunov Exponent


% figure;
% plot(t, log_div, 'b', 'LineWidth', 1.5); hold on;
% plot(t(fit_range), polyval(p, t(fit_range)), 'r--', 'LineWidth', 2);
% xlabel('Time');
% ylabel('log divergence');
% title(['Largest Lyapunov Exponent = ', num2str(lambda, '%.4f')]);
% legend('log(divergence)', 'linear fit');
% grid on;

end