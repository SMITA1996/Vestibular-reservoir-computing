function [dx] = SSCFHN(X,U_in,A,W_in,n)

%FHN SSC model
x=X(1:n);
y=X(n+1:2*n);
u=X(2*n+1:3*n);
v=X(3*n+1:4*n);



a = 0.7;
b = 2;

R1 = 6.5;
R2 = 1;
tau1 = 20;
tau2 = 20;
tau = 20
m=2;
c=12;
k=50;



dx(1:n,1) = tau2*(y);
dx(n+1:2*n,1) = tau2*(A*x + R2*(W_in*U_in) +(- c * y - k * x) / m);
dx(2*n+1:3*n,1) = tau1*((1-4*1.2)*u-v-u.^3./3 + R1*(x));
dx(3*n+1:4*n,1) = tau1*(u+a-b*v)./tau1;



end
