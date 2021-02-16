function [x,w]=gaussquad(a,b,n)


%gauss-legendre quadrature
beta=.5./sqrt(1-(2*(1:n-1)').^(-2));
T=diag(beta,1)+diag(beta,-1);
[V,D]=eig(T);
x=diag(D);[x,i]=sort(x);
w=2*V(1,i).^2;
x=((b-a)*x+a+b)/2;x=x(:);
w=w*(b+a)/2;

%int=w*(W.*fun(x));
%int=w*(W.*feval(fun,x(:))');
