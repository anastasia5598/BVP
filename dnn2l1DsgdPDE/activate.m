function y = activate(x,W,b,fun)
    % activation function
    % x - input
    % W - weights matrix
    % b - biases vector
    % fun - choice of activation function
    % returns the activated value(s)
    
    % weighted input
    z = W * x + b;
    
    if fun == 1
        % sigmoid
        y = 1 ./ (1 + exp(-z));
    elseif fun == 2
        % tanh
        y = max(tanh(z),0);
    elseif fun == 3
        % ReLU
        y = max(z,0);
    elseif fun == 4
        % identity
        y = z;
    elseif fun == 5
        % cubic
        %y = max(z.^3,0);
        y = z .^ 3;
    elseif fun == 6
        % absolut value
        y = abs(z);
    elseif fun == 7
        % softmax
        y = log(1 + exp(z));
    end
end