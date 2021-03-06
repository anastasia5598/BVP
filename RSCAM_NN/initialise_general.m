function initialise_general

    % Learning rate and no. iterations
    eps = 0.01;
    NSteps = 10000;
    
    % No. neurons in hidden layer
    m = 3;
    
    x = linspace(0, 1, 100);
    
    % Function to approximate
    g = cos(x);
    
    % Inital weights and biases
    % Note: There is 1 hidden layer with m nodes
    W1 = randn(m, 1);
    W2 = randn(1, m);
    B1 = randn(m, 1);
    B2 = randn(1, 1);
    
    % Update parameters for final weights and biases
    [W1, W2, B1, B2] = update_params(W1, W2, B1, B2, x, g, NSteps, eps);
    
    % Run final approximation
    approx = network_layer(x, W1, W2, B1, B2);
    
    % Plot against original function
    figure(1)
    plot(x, approx)
    hold on
    plot(x, g)
    hold off
    
end

function x_out = network_layer(x, W1, W2, B1, B2)
    % Computation of hidden layer
    % W, B are weights and biases for the hidden layer and output
    
    x_next = zeros(length(x), 1);
    x_out = zeros(length(x), 1);
    
    for idx = 1:length(x)
        z = W1*x(idx) + B1;

        % Activation (sigmoid function)
        x_next = 1./(1 + exp(-z));

        % Compute final output
        x_out(idx) = W2*x_next + B2; 
    end
end

function [W1, W2, B1, B2] = update_params(W1, W2, B1, B2, x, g, NSteps, eps)

    % Loop through the number of iterations
    for N=1:NSteps
        
        % Perform updates for each point x
        for idx=1:length(x)
            
            % Activation function
            a = 1./(1 + exp(-(W1*x(idx) + B1)));

            % C = (f-g)^2
            % dC/dP = a(f-g)*df/dP
            
            % Partial derivatives of f
            df_dw1 = W2*a.*(1-a)*x(idx);
            df_db1 = W2*a.*(1-a);
            df_dw2 = a;
            df_db2 = 1;

            % Partial derivatives of cost function
            dC_dw1 = 2*(network_layer(x(idx), W1, W2, B1, B2) - g(idx))*df_dw1;
            dC_db1 = 2*(network_layer(x(idx), W1, W2, B1, B2) - g(idx))*df_db1;
            dC_dw2 = 2*(network_layer(x(idx), W1, W2, B1, B2) - g(idx))*df_dw2;
            dC_db2 = 2*(network_layer(x(idx), W1, W2, B1, B2) - g(idx))*df_db2;

            % Updating weights and biases
            W1 = W1 - eps*dC_dw1;
            B1 = B1 - eps*dC_db1;
            W2 = W2 - (eps*dC_dw2)';
            B2 = B2 - eps*dC_db2;
        end
    end
    
end


