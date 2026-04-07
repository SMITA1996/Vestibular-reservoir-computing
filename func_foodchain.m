function dx = func_foodchain(~, x, k, yc, yp, xc, xp, r0, c0)
    dx = zeros(3,1);
    dx(1) = x(1) * (1 - x(1)/k) - xc*yc*x(2)*x(1)/(x(1)+r0);
    dx(2) = xc*x(2) * (yc*x(1)/(x(1)+r0) - 1) - xp*yp*x(3)*x(2)/(x(2)+c0);
    dx(3) = xp*x(3) * (yp*x(2)/(x(2)+c0) - 1);
end
