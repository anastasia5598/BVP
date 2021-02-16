function Pfinal = costfunsigm(Pval, x, y, weights, biases, fun, eta, Niter)
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

%%%%%%%% gradient descent
for counter = 1:Niter
    for j = 1:size(x)
        a1 = x(j);
        
        %%%%%%%% Forward pass
        % activate each layer
        a2 = activate(a1, w1, b1, fun);
        a3 = activate(a2, w2, b2, fun);
        % compute trained value
        u = w3 * a3 + b3;
        gradu = w3 * (w2 * (w1 .* a2 .* (1-a2)) .* a3 .* (1-a3));
        
        %%%%%%%% Backward pass
        deltaw3 = 1/2 * (gradu .^ 2) - y*u;
        %deltaw3 = gradu * (w2 * (w1 .* a2 .* (1-a2)) .* a3 .* (1-a3)) - y * a3;
        deltab3 = -y;
        delta2 = a3 .* (1-a3) .* (gradu *  w3 * w2 * w1 .* a2 .* (1-a2) .* (1 - 2*a3) - y * w3');
        delta1 = a2 .* (1-a2) .* (w2' * delta2);

        % Gradient step
        w1 = w1 - eta*delta1*a1;
        w2 = w2 - eta*delta2*a2';
        w3 = w3 - eta*deltaw3';
        b1 = b1 - eta*delta1;
        b2 = b2 - eta*delta2;
        b3 = b3 - eta*deltab3;
    end
end

%%%%%%%%% pack weights and biases
Pfinal = [w1(:); b1(:); w2(:); b2(:); w3(:); b3(:)];

end