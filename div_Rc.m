function [kl_div,hellinger_dist,tv_div] = untitled14(target_ts,auto_Lor)
% p = GT, q = KAN
x1 = target_ts(:, 1);
y1 = target_ts(:, 2);
z1 = target_ts(:, 3);
x2 = auto_Lor(:, 1);
y2 = auto_Lor(:, 2);
z2 = auto_Lor(:, 3);


% % p =KAn, q = rand
% x1 = y_pred_ms(:, 1);
% y1 = y_pred_ms(:, 2);
% z1 = y_pred_ms(:, 3);
% x2 = randn(1000, 1);
% y2 = randn(1000, 1);
% z2 = randn(1000, 1);


% Grid points for evaluation
grid_size = 100;  % Reduced grid size for 3D
x_grid = linspace(-5, 5, grid_size);
y_grid = linspace(-5, 5, grid_size);
z_grid = linspace(-5, 5, grid_size);
[X, Y, Z] = ndgrid(x_grid, y_grid, z_grid);

% Estimate PDFs using the custom KDE in 3D
bandwidth = 0.5;  % Adjust bandwidth for 3D
pdf1 = kde_estimation_3d([x1, y1, z1], X, Y, Z, bandwidth);
pdf2 = kde_estimation_3d([x2, y2, z2], X, Y, Z, bandwidth);

% Normalize PDFs
pdf1 = pdf1 / sum(pdf1(:));
pdf2 = pdf2 / sum(pdf2(:));

kl_div = kl_divergence_3d(pdf1, pdf2);
% fprintf('KL Divergence: %.4f\n', kl_div);

hellinger_dist = hellinger_distance_3d(pdf1, pdf2);
% fprintf('Hellinger Distance: %.4f\n', hellinger_dist);

tv_div = tv_divergence_3d(pdf1, pdf2);
% fprintf('Total Variation Divergence: %.4f\n', tv_div);


% Custom KDE for 3D data
function pdf = kde_estimation_3d(data, X, Y, Z, bandwidth)
    num_points = length(X(:));
    pdf = zeros(size(X));
    for i = 1:num_points
        dist_sq = sum(((data - [X(i), Y(i), Z(i)]) / bandwidth).^2, 2);
        pdf(i) = sum(exp(-0.5 * dist_sq));
    end
    pdf = pdf / (num_points * (bandwidth * sqrt(2*pi))^3);
end


% KL Divergence for 3D
function kl_div = kl_divergence_3d(p, q)
    kl_div = sum(p(:) .* log(p(:) ./ q(:)), 'omitnan');
end



% Hellinger Distance for 3D
function hellinger_dist = hellinger_distance_3d(p, q)
    hellinger_dist = sqrt(sum((sqrt(p(:)) - sqrt(q(:))).^2)) / sqrt(2);
end



% Total Variation (TV) Divergence for 3D
function tv_div = tv_divergence_3d(p, q)
    tv_div = 0.5 * sum(abs(p(:) - q(:)));
end
end