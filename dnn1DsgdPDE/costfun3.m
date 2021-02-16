function Pfinal = costfun(Pval, x, y, weights, biases, fun, eta, Niter)
% cost function - computes the difference between given output and
%                 trained output (parameter for lsqnonlin to min MSE)
% Pzero - initial random paremeters of neural network
% x and y - input-output training data
% weights and biases - structure of neural network, see rundnn for
%                      details
% fun - choice of activation function
% returns a vector of diffferences for each training point

n = size(x, 1);
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
m1 = 0; m2 = 0; m3 = 0; m4 = 0;
v1 = 0; v2 = 0; v3 = 0; v4 = 0;

%%%%%%%% gradient descent
for counter = 1:Niter
    for j = 1:size(x)
        xx = x(j);
        ni = 0;
        
        % activate each layer
        for i = 1:(size(weights)-1)
            w1 = zeros(weights(i,1), weights(i,2));
            w1(:) = Pval((ni +1):(ni + prod(weights(i,:))));
            ni = ni + prod(weights(i,:));
            b1 = Pval((ni +1):(ni + biases(i)));
            ni = ni + biases(i);
            a = activate(xx, w1, b1, fun);
        end
        
        % compute trained value
        w2 = zeros(weights(i+1,1), weights(i+1,2));
        w2(:) = Pval((ni +1):(ni + prod(weights(i+1,:))));
        ni = ni + prod(weights(i+1,:));
        b2 = Pval((ni +1):(ni + biases(i+1)));
        u = w2 * a + b2;
        gradu = w2 * (w1 .* a .* (1-a));
        
        % Backward pass
        %delta2 = 1/2 * (gradu .^ 2) - y*u;
        %delta2 = (gradu .* gradu * (w1' * (1 - 2*a)) - y);
        %delta2 = abs(gradu * 6 * w2 * (w1 .* a) -y);
        %delta2 = -y;
        delta1 = a .* (1-a) .* (gradu * (w2 * w1) .* (1-2*a) - w2'*y);
        deltaw2 = gradu .* w1 .* a .* (1-a) - y * a;       
        deltab2 = -y;
        deltaw1 = delta1 * xx;
        deltab1 = delta1;
        
        m1 = beta1 * m1 + (1-beta1) * deltaw1;
        m2 = beta1 * m2 + (1-beta1) * deltab1;
        m3 = beta1 * m3 + (1-beta1) * deltaw2;
        m4 = beta1 * m4 + (1-beta1) * deltab2;
        
        v1 = beta2 * v1 + (1-beta2) * (deltaw1 .* deltaw1);
        v2 = beta2 * v2 + (1-beta2) * (deltab1 .* deltab1);
        v3 = beta2 * v3 + (1-beta2) * (deltaw2 .* deltaw2);
        v4 = beta2 * v4 + (1-beta2) * (deltab2 .* deltab2);
        
        mhat1 = m1 / (1 - beta1 ^ counter);
        mhat2 = m2 / (1 - beta1 ^ counter);
        mhat3 = m3 / (1 - beta1 ^ counter);
        mhat4 = m4 / (1 - beta1 ^ counter);
        
        vhat1 = v1 / (1 - beta2 ^ counter);
        vhat2 = v2 / (1 - beta2 ^ counter);
        vhat3 = v3 / (1 - beta2 ^ counter);
        vhat4 = v4 / (1 - beta2 ^ counter);
        
        % Gradient step
        w1 = w1 - eta*mhat1 / (sqrt(vhat1) + epsilon);
        w2 = w2 - eta*mhat3 / (sqrt(vhat3) + epsilon);
        b1 = b1 - eta*mhat2 / (sqrt(vhat2) + epsilon);
        b2 = b2 - eta*mhat4 / (sqrt(vhat4) + epsilon);
    end
end

%%%%%%%%% pack weights and biases
Pfinal = [w1(:); b1(:); w2(:); b2(:)];

end