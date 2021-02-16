function [y, grad] = activate(z,fun)
    % activation function
    % z - weighted input
    % fun - choice of activation function
    % returns the activated value(s)
    
    if fun == 1
        %% sigmoid
        y = 1 ./ (1 + exp(-z));
        grad = y .* (1-y);  
    elseif fun == 2
        %% tanh
        y = max(tanh(z),0);
        grad = 1-y.^2;   
    elseif fun == 3
        %% ReLU 
        y = max(z,0);
        grad = 1;
    elseif fun == 4
        %% identity
        y = z;
        grad = 1;
    elseif fun == 5
        %% cubic
        y = max(z.^3,0);
        grad = 3 * z .^ 2;
    elseif fun == 6
        %% absolut value
        y = abs(z);
        grad = 1;
    elseif fun == 7
        %% softmax
        y = log(1 + exp(z));
        grad = 1 - (1 ./ (exp(z) + 1));
    end
end