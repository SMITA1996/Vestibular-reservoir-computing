function  DV = D_value(test_real_y,test_pred_y, testing_length)
% we use DV (deviation value) to measure the long term prediction precision
% define the boundary of the lattice
max_value = 1.5;
min_value = -1.5;

xlim_set = [min_value, max_value];
ylim_set = [min_value, max_value];

lower_ = xlim_set(1);
dv_dt = 0.05;
matrix_size = ceil((xlim_set(2)-xlim_set(1))/dv_dt);

% create the lattice
real_matrix=zeros(matrix_size+1, matrix_size+1);
pred_matrix=zeros(matrix_size+1, matrix_size+1);

real_points = [test_real_y(:, 1), test_real_y(:, 3)];
pred_points = [test_pred_y(:, 1), test_pred_y(:, 3)];

% limit real_points and pred_points in the range of [min_value, max_value]
real_points = max(min(real_points, max_value), min_value);
pred_points = max(min(pred_points, max_value), min_value);

% count the number of points in each box
for pt = 1:length(real_points)
    real_x = floor(abs((real_points(pt, 1)-lower_) / dv_dt));
    real_y = floor(abs((real_points(pt, 2)-lower_) / dv_dt));
    pred_x = floor(abs((pred_points(pt, 1)-lower_) / dv_dt));
    pred_y = floor(abs((pred_points(pt, 2)-lower_) / dv_dt));

    if real_x == 0
        real_x = 1;
    end
    if real_y == 0
        real_y = 1;
    end
    if pred_x == 0
        pred_x = 1;
    end
    if pred_y == 0
        pred_y = 1;
    end

    real_matrix(real_x, real_y) = real_matrix(real_x, real_y)+1;
    pred_matrix(pred_x, pred_y) = pred_matrix(pred_x, pred_y)+1;
end

real_matrix = (reshape(real_matrix, 1, []))./testing_length;
pred_matrix = (reshape(pred_matrix, 1, []))./testing_length;

DV = sum(sqrt((real_matrix-pred_matrix).^2));
end