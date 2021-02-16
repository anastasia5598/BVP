function y = grad(fun, a)
    
    if fun == 1
        % sigmoid
        y = a .* (1-a);
    elseif fun == 2
        % tanh
        y = 1-a.^2;
    elseif fun == 3
        % ReLU
        y = 1;
    elseif fun == 4
        % identity
        y = 1;
    elseif fun == 5
        % cubic
        y = 3 * a.^2;
    elseif fun == 6
        % absolut value
        y = 1;
    elseif fun == 7
        % softmax
        y = exp(a) .* (1 ./ (1 + exp(a)));
    end
end