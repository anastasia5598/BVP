function Pfinal = costfun2(Pval, x, y, weights, biases, fun, eta, Niter)
% cost function - computes the difference between given output and
%                 trained output (parameter for lsqnonlin to min MSE)
% Pzero - initial random paremeters of neural network
% x and y - input-output training data
% weights and biases - structure of neural network, see rundnn for
%                      details
% fun - choice of activation function
% returns a vector of diffferences for each training point

n = size(x, 1);

w1 = zeros(weights(1,1), weights(1,2));
w2 = zeros(weights(2,1), weights(2,2));
w3 = zeros(weights(3,1), weights(3, 2));
ni = 0;
w1(:) = Pval((ni +1):(ni + prod(weights(1,:))));
ni = ni + prod(weights(1,:));
b1 = Pval((ni +1):(ni + biases(1)));
ni = ni + biases(1);
w2(:) = Pval((ni +1):(ni + prod(weights(2,:))));
ni = ni + prod(weights(2,:));
b2 = Pval((ni +1):(ni + biases(2)));
ni = ni + biases(2);
w3(:) = Pval((ni +1):(ni + prod(weights(3,:))));
ni = ni + prod(weights(3,:));
b3 = Pval((ni +1):(ni + biases(3)));

beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
m1 = 0; m2 = 0; m3 = 0; m4 = 0; m5 = 0; m6 = 0;
v1 = 0; v2 = 0; v3 = 0; v4 = 0; v5 = 0; v6 = 0;

%%%%%%%% gradient descent
for counter = 1:Niter
    for j = 1:size(x)
        a1 = x(j);
        
        %%%%%%%% Forward pass
        % activate each layer
        z1 = w1 * a1 + b1;
        a2 = activate(a1, w1, b1, fun);
        z2 = w2 * a2 + b2;
        a3 = activate(a2, w2, b2, fun);
        % compute trained value
        u = w3 * a3 + b3;
        gradu = w3 * (w2 * (w1 .* 3 .* (z2 .^ 2)) .* 3 .* (z1 .^ 2));
        
        %%%%%%%% Backward pass
        %deltaw3 = gradu * (w2 * (w1 .* a2 .* (1-a2)) .* a3 .* (1-a3)) - y * a3;
        %deltab3 = -y;
        delta3 = 1/2 * (gradu .^ 2) - y*u;
        deltaw3 = delta3 * a3';
        deltab3 = delta3;
        %delta2 = a3 .* (1-a3) .* (gradu *  w3 * w2 * w1 .* a2 .* (1-a2) .* (1 - 2*a3) - y * w3');
        delta2 = 3 .* (z2 .^ 2)  .* (w3' * delta3);
        deltaw2 = delta2 * a2';
        deltab2 = delta2;
        delta1 = 3 .* (z1 .^ 2) .* (w1' * delta2);
        deltaw1 = delta1 * a1;
        deltab1 = delta1;

        m1 = beta1 * m1 + (1-beta1) * deltaw1;
        m2 = beta1 * m2 + (1-beta1) * deltab1;
        m3 = beta1 * m3 + (1-beta1) * deltaw2;
        m4 = beta1 * m4 + (1-beta1) * deltab2;
        m5 = beta1 * m5 + (1-beta1) * deltaw3;
        m6 = beta1 * m6 + (1-beta1) * deltab3;
        
        v1 = beta2 * v1 + (1-beta2) * (deltaw1 .* deltaw1);
        v2 = beta2 * v2 + (1-beta2) * (deltab1 .* deltab1);
        v3 = beta2 * v3 + (1-beta2) * (deltaw2 .* deltaw2);
        v4 = beta2 * v4 + (1-beta2) * (deltab2 .* deltab2);
        v5 = beta2 * v5 + (1-beta2) * (deltaw3 .* deltaw3);
        v6 = beta2 * v6 + (1-beta2) * (deltab3 .* deltab3);
        
        mhat1 = m1 / (1 - beta1 ^ counter);
        mhat2 = m2 / (1 - beta1 ^ counter);
        mhat3 = m3 / (1 - beta1 ^ counter);
        mhat4 = m4 / (1 - beta1 ^ counter);
        mhat5 = m5 / (1 - beta1 ^ counter);
        mhat6 = m6 / (1 - beta1 ^ counter);
        
        vhat1 = v1 / (1 - beta2 ^ counter);
        vhat2 = v2 / (1 - beta2 ^ counter);
        vhat3 = v3 / (1 - beta2 ^ counter);
        vhat4 = v4 / (1 - beta2 ^ counter);
        vhat5 = v5 / (1 - beta2 ^ counter);
        vhat6 = v6 / (1 - beta2 ^ counter);
        
        % Gradient step
        w1 = w1 - eta*mhat1 / (sqrt(vhat1) + epsilon);
        w2 = w2 - eta*mhat3 / (sqrt(vhat3) + epsilon);
        w3 = w3 - eta*mhat5 / (sqrt(vhat5) + epsilon);
        b1 = b1 - eta*mhat2 / (sqrt(vhat2) + epsilon);
        b2 = b2 - eta*mhat4 / (sqrt(vhat4) + epsilon);
        b2 = b2 - eta*mhat6 / (sqrt(vhat6) + epsilon);
    end
end

%%%%%%%%% pack weights and biases
Pfinal = [w1(:); b1(:); w2(:); b2(:); w3(:); b3(:)];

end