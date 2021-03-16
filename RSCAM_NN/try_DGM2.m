function try_DGM2

    % Learning rate and no. iterations
    eps = 0.01;
    NSteps = 10000;
    
    % No. neurons in hidden layer
    m = 1;
    
    x = linspace(0, 1, 100);
    
    % Function to approximate
    % df_dx = g
    % g = x 
    g = ones(100, 1);
    
    % Soln: f = x
    h = 0.5*(x + x.^2);
    
    % Inital weights and biases
    % Note: There is 1 hidden layer with m nodes
    W1 = randn(m, 1);
    W2 = randn(1, m);
    B1 = randn(m, 1);
    B2 = randn(1, 1);
    
    % Update parameters for final weights and biases
    [W1, W2, B1, B2] = update_params(W1, W2, B1, B2, x, g, h, NSteps, eps);
    
    % Run final approximation
    approx = network_layer(x, W1, W2, B1, B2);
    
    % Plot against original function
    figure(1)
    plot(x, approx)
    hold on
    plot(x, h)
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

function [W1, W2, B1, B2] = update_params(W1, W2, B1, B2, x, g, h, NSteps, eps)

    % Loop through the number of iterations
    for N=1:NSteps
        
        % Perform updates for each point x
        for idx=1:length(x)
            
            % Activation function
            z = W1*x(idx) + B1;
            a = 1./(1 + exp(-z));
            
            
            da_dx = W1.*a.*(1-a);
            da_dw1 = a.*(1-a)*x(idx);
            da_db1 = a.*(1-a);
            
            df_dx = W2 * (W1 .* a .* (1-a));
            df_dx2 = W2 * (W1 .* (1 - 2*a) .* W1 .* a .* (1 - a));
            
            % C = (f''-g)^2
            % dC/dP = a (f''-g) * df''/dP
            % f = W2*a(W1*x + B1) + B2
            
            % Partial derivatives of f
            
            df_dx2_dw1 = W2 * ((1 - 2*a) .* W1 .* a .* (1-a)) + W2 * (W1 .* (1 - 2*a) .* a .* (1-a)) - 2*x(idx) * W2 * (W1 .* a .* (1-a) .* W1 .* a .* (1-a)) + x(idx) * W2 *(W1 .*(1-2*a) .* W1 .* a .* (1-a) .* (1 - a)) - x(idx) * W2 * (W1 .* (1 - 2*a) .* W1 .* a .* a .* (1-a)); 
            df_dx2_db1 = - 2* W2 * (W1 .* a * (1 - a)) + W2 * (W1 .* (1 - 2*a) .* W1 .* a .* (1 - a) .* (1 - a)) - W2 * (W1 .* (1 - 2*a) .*a .* (1 - a));
            df_dx2_dw2 = W1 .* (1 - 2*a) .* W1 .* a .* (1-a);
            df_dx2_db2 = 0;
            
            dC1_dw1 = 2*(df_dx2 - g(idx)).*df_dx2_dw1;
            dC1_db1 = 2*(df_dx2 - g(idx)).*df_dx2_db1;
            dC1_dw2 = 2*(df_dx2 - g(idx)).*df_dx2_dw2;
            dC1_db2 = 2*(df_dx2 - g(idx)).*df_dx2_db2;
            

            % C2 = (f_0 - h_0)^2
            % dC2_dP = 2*(f_0 - h_0)*df/dP(0) 
            a_0 = 1./(1 + exp(-(W1*x(1) + B1)));
            
            df_dw1_0 = W2*a_0.*(1-a_0)*x(1);
            df_db1_0 = W2*a_0.*(1-a_0);
            df_dw2_0 = a_0;
            df_db2_0 = 1;
            
            a_1 = 1./(1 + exp(-(W1*1 + B1)));
            
            df_dw1_1 = W2*a_1.*(1-a_1)*x(end);
            df_db1_1 = W2*a_1.*(1-a_1);
            df_dw2_1 = a_1;
            df_db2_1 = 1;
            
            % Boundary Conditions for x = 0
            f_0 = network_layer(x(1), W1, W2, B1, B2);
            dC2_dw1 = 2*df_dw1_0*(f_0 - h(1));
            dC2_db1 = 2*df_db1_0*(f_0 - h(1));
            dC2_dw2 = 2*df_dw2_0*(f_0 - h(1));
            dC2_db2 = 2*df_db2_0*(f_0 - h(1));
            
            % Boundary Conditions for x = 1
            f_end = network_layer(x(end), W1, W2, B1, B2);
            dC3_dw1 = 2*df_dw1_1*(f_end - h(end));
            dC3_db1 = 2*df_db1_1*(f_end - h(end));
            dC3_dw2 = 2*df_dw2_1*(f_end - h(end));
            dC3_db2 = 2*df_db2_1*(f_end - h(end));
            
            % Add loss functions 
            dC_dw1 = dC1_dw1 + dC2_dw1 + dC3_dw1;
            dC_db1 = dC1_db1 + dC2_db1 + dC3_db1;
            dC_dw2 = dC1_dw2 + dC2_dw2 + dC3_dw2;
            dC_db2 = dC1_db2 + dC2_db2 + dC3_db2;
            
            % Updating weights and biases
            W1 = W1 - eps*dC_dw1;
            B1 = B1 - eps*dC_db1;
            W2 = W2 - (eps*dC_dw2)';
            B2 = B2 - eps*dC_db2;
        end
    end
    
end

