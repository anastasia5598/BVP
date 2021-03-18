function initialise_DRM

    % Learning rate and no. iterations
    eps = 5e-3;
    NSteps = 1e5;
    beta = 40;
    
    % No. neurons in hidden layer
    m = 10;
    
    x = linspace(0, 1, 10);
    
    % Function to approximate
    % d^2f_dx^2 = g
    g = ones(100, 1);
    sol = @(x) 1/2 * (x - x .^2);
    
    % Inital weights and biases
    % Note: There is 1 hidden layer with m nodes
    W1 = 0.5*randn(m, 1);
    W2 = 0.5*randn(1, m);
    B1 = 0.5*randn(m, 1);
    B2 = 0.5*randn(1, 1);
    
    % Update parameters for final weights and biases
    [W1, W2, B1, B2] = update_params(W1, W2, B1, B2, x, g, NSteps, eps, beta);
    
    % Run final approximation
    x = linspace(0, 1, 10000);
    approx = network_layer(x, W1, W2, B1, B2);
    
    % Plot against original function
    figure(1)
    plot(x, approx, 'r--')
    hold on
    plot(x, sol(x))
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

function [W1, W2, B1, B2] = update_params(W1, W2, B1, B2, x, g, NSteps, eps, beta)

    for N=1:NSteps
        for idx=1:length(x)
        
            % BC's - at 0 and 1
            au0 = 1./(1 + exp(-(W1*0 + B1)));
            u0 = W2 * au0 + B2;
            au1 = 1./(1 + exp(-(W1*1+ B1)));
            u1 = W2 * au1 + B2;
        
            % Activation function
            a = 1./(1 + exp(-(W1*x(idx) + B1)));
            u = W2 * a + B2;
            
            % partial derivative of C wrt u
            gradu = W2 * (W1 .* a .* (1-a));
            
            % delta - partial derivative of C wrt z
            delta1 = a .* (1-a) .* (gradu * (W2 * W1) .* (1-2*a) - W2'*g(idx));

            % Partial derivatives of cost function
            dC_dw1 = delta1 * x(idx) + beta * (u1 *  W2' .* au1 .* (1-au1));
            dC_db1 = delta1 + beta * (u0 * W2' .* au0 .* (1-au0) + u1 * W2' .* au1 .* (1-au1));
            dC_dw2 = gradu .* W1 .* a .* (1-a) - g(idx) * a + beta * (u0 * au0 + u1 * au1);
            dC_db2 = -g(idx) + beta*(u0 + u1);
            
            % Updating weights and biases
            W1 = W1 - eps*dC_dw1;
            B1 = B1 - eps*dC_db1;
            W2 = W2 - eps*dC_dw2';
            B2 = B2 - eps*dC_db2;
        end
    end
    
end


