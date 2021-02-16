function y = evalDNN(x, Pfinal, weights, biases, fun)
    % evalDNN - evaluates the value of the approximated function by the
    %           (deep) neural network at a given point / set of points
    % x - point / set of points at which to evaluate DNN
    % Pfinal - final parameters of DNN (after training network
    % weights and biases - structure of neural network, see rundnn for
    %                      details
    % returns the value(s) of DNN at given point(s)
    
    %%%%%%%% evaluate DNN at x
    for i = 1 : size(x)
        res = x(i);
        y = x(i);
        for j=1: (weights(1,1) -1)
            res = [res; 0];
            y = [y; 0];
        end
        ni = 0;
        
        % activate network
        for j = 1 : (size(weights)-1)
            wi = zeros(weights(j,1), weights(j,2));
            wi(:) = Pfinal((ni +1):(ni + prod(weights(j,:))));
            ni = ni + prod(weights(j,:));
            bi = Pfinal((ni +1):(ni + biases(j)));
            ni = ni + biases(j);
            y = activate(y, wi, bi, fun);
        end
        
        % compute DNN(x)
        wi = zeros(weights(j+1,1), weights(j+1,2));
        wi(:) = Pfinal((ni +1):(ni + prod(weights(j+1,:))));
        ni = ni + prod(weights(j+1,:));
        bi = Pfinal((ni +1):(ni + biases(j+1)));
        z(i) = wi * (y + res)+ bi;
    end
    
    y = z';
end