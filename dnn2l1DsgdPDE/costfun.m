function Pfinal = costfun(Pval, beta, x, y, weights, biases, fun, eta, Niter)
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
        %%%%%%%% change this
        s = x(j);
        x0 = 0; x1 = 1;   
        for i=1:(size(w1) -1)
            s = [s; 0];
            x0 = [x0; 0];
            x1 = [x1; 0];
        end
        
        %%%%%%%% BC's
        a1u0 = activate(x0, w1, b1, fun);
        a2u0 = activate(a1u0, w2, b2, fun);
        u0 = w3 * (a2u0 + x0) + b3;
        
        a1u1 = activate(x1, w1, b1, fun);
        a2u1 = activate(a1u1, w2, b2, fun);
        u1 = w3 * (a2u1 + x1) + b3;
        
        %%%%%%%% Forward pass
        % activate each layer
        a1 = activate(s, w1, b1, fun);
        a2 = activate(a1, w2, b2, fun);
        % compute trained value
        u = w3 * (a2 + s) + b3;
        grads = 1;
        for i=1:(size(w1) -1)
            grads = [grads; 0];
        end
        product = a2 .* (1-a2) .* (w2 * ((a1 .* (1-a1)) .* (w1 * grads)));
        gradu = w3 * (product + grads);
        
        %%%%%%%% Backward pass
        %delta3 = 1/2 * (gradu .^ 2) - y*u;
        deltaw3u = gradu * (product + grads) - y * (a2 + s);
        deltaw3u0 = u0 * (a2u0 + x0);
        deltaw3u1 = u1 * (a2u1 + x1);
        deltaw3 = deltaw3u + beta * (deltaw3u0 + deltaw3u1); 
        
        deltab3u = -y;
        deltab3u0 = u0;
        deltab3u1 = u1;
        deltab3 = deltab3u + beta * (deltab3u0 + deltab3u1);
        
        delta21 = gradu * w3' .* product .* (1-2*a2);
        delta22 =  w3' .* a2 .* (1-a2) * y;
        delta2 = delta21 - delta22;
        delta2u0 = u0 .* w3' .* a2u0 .* (1-a2u0);
        delta2u1 = u1 .* w3' .* a2u1 .* (1-a2u1);
        deltaw2 = delta2 * a1' + beta * (delta2u0 * a1u0' + delta2u1 * a1u1');
        deltab2 = delta2 + beta * (delta2u0 + delta2u1);
        
        delta12 = delta22 .* (w2' * (a1 .* (1-a1)));
        delta11 = delta21 .* (w2' * (a1 .* (1-a1))) + gradu * w3' .* product .* (1 -2 *a1);
        delta1 = delta11 - delta12;
        delta1u0 = u0 .* delta2u0 .* (w2' * (a1u0 .* (1-a1u0)));        
        delta1u1 = u1 .* delta2u1 .* (w2' * (a1u1 .* (1-a1u1)));
        deltaw1 = delta1 * s' + beta * (delta1u0 * x0' + delta1u1 * x1');
        deltab1 = delta1 + beta * (delta1u0 + delta1u1);
        
        % Gradient step
        w1 = w1 - eta*deltaw1;
        w2 = w2 - eta*deltaw2;
        w3 = w3 - eta*deltaw3';
        b1 = b1 - eta*deltab1;
        b2 = b2 - eta*deltab2;
        b3 = b3 - eta*deltab3;
    end
end

%%%%%%%%% pack weights and biases
Pfinal = [w1(:); b1(:); w2(:); b2(:); w3(:); b3(:)];

end