function Pfinal = costfun(Pval, x, y, weights, biases, fun, eta, Niter, beta)
% cost function - computes the difference between given output and
%                 trained output (parameter for lsqnonlin to min MSE)
% Pzero - initial random paremeters of neural network
% x and y - input-output training data
% weights and biases - structure of neural network, see rundnn for
%                      details
% fun - choice of activation function
% returns a vector of diffferences for each training point

n = size(x, 1);

ni = 0;
w1 = zeros(weights(1,1), weights(1,2));
w1(:) = Pval((ni +1):(ni + prod(weights(1,:))));
ni = ni + prod(weights(1,:));
b1 = Pval((ni +1):(ni + biases(1)));
ni = ni + biases(1);
w2 = zeros(weights(2,1), weights(2,2));
w2(:) = Pval((ni +1):(ni + prod(weights(2,:))));
ni = ni + prod(weights(2,:));
b2 = Pval((ni +1):(ni + biases(2)));
ni = ni + biases(2);
costmat = [];

%%%%%%%% gradient descent
for counter = 1:Niter
    for j = 1:size(x)
        xx = x(j);
        
        %%%%% BC's
        au0 = activate(0, w1, b1, fun);
        u0 = w2 * au0 + b2;
        au1 = activate(1, w1, b1, fun);
        u1 = w2 * au1 + b2;
        
        % activate each layer
        a = activate(xx, w1, b1, fun);
        u = w2 * a + b2;
        gradu = w2 * (w1 .* a .* (1-a));
        
        % Backward pass
        %delta2 = 1/2 * (gradu .^ 2) - y*u;
        %delta2 = (gradu .* gradu * (w1' * (1 - 2*a)) - y);
        %delta2 = abs(gradu * 6 * w2 * (w1 .* a) -y);
        %delta2 = -y;
        delta1 = a .* (1-a) .* (gradu * (w2 * w1) .* (1-2*a) - w2'*y);
        %delta1 = a .* (1-a)  .* (w2' * delta2);
        deltaw2 = gradu .* w1 .* a .* (1-a) - y * a + beta * (u0 * au0 + u1 * au1);
        deltab2 = -y + beta*(u0 + u1);
        deltaw1 = delta1 * xx + beta * (u1 *  w2' .* au1 .* (1-au1));
        deltab1 = delta1 + beta * (u0 * w2' .* au0 .* (1-au0) + u1 * w2' .* au1 .* (1-au1));
        %delta1 = a .* (1-a) .* (gradu * (w2 * w1) .* (1-2*a) - w2'*y);
        %deltaw2 = gradu .* w1 .* a .* (1-a) - y * a;     
        %deltab2 = -y;
        %deltaw1 = a .* (1-a) .* (gradu * w2' .* (1 + (w1 * xx) .* (1-2*a)) - y *xx);
        %deltab1 = a .* (1-a) .* (gradu * w2 * w1 .* (1-2*a) - y * w2');
        
        % Gradient step
        w1 = w1 - eta*deltaw1;
        w2 = w2 - eta*deltaw2';
        b1 = b1 - eta*deltab1;
        b2 = b2 - eta*deltab2;
        %costvec(j) = 1/2 * (gradu .^ 2) - y*u;
    end
    %costmat = [costmat; costvec];
end

%%%%%%%%% pack weights and biases
Pfinal = [w1(:); b1(:); w2(:); b2(:)];

end