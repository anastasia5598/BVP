function initialise_DGM

    eps = 0.01;
    NSteps = 10000;
    
    x = linspace(0, 1, 100);
    % g is function
    g = cos(x);
    
    W = randn(2, 1);
    B = randn(2, 1);
    
    [W, B] = update_params(W, B, x, g, NSteps, eps);
    
    approx = network_layer(x, W, B);
    
    figure(1)
    plot(x, approx)
    hold on
    plot(x, g)
    hold off
    
end

function x_out = network_layer(x, W, B)
    % Computation of hidden layer
    % W, B are weights and biases for the hidden layer and output
    
    x_next = zeros(length(x), 1);
    x_out = zeros(length(x), 1);
    
    for idx = 1:length(x)
        z = W(1)*x(idx) + B(1);

        % Activation function: sigmoid 
        x_next(idx) = 1/(1 + exp(-z));

        % Compute final output
        x_out(idx) = W(2)*x_next(idx) + B(2); 
    end
end

function [W, B] = update_params(W, B, x, g, NSteps, eps)

    for N=1:NSteps
        for idx=1:length(x)
            % Activation function
            a = 1/(1 + exp(-(W(1)*x(idx) + B(1))));

            df_dw1 = W(2)*a*(1-a)*x(idx);
            df_db1 = W(2)*a*(1-a);
            df_dw2 = a;
            df_db2 = 1;

            % C = (f-g)^2
            % dC/dP = a(f-g)*df/dP

            dC_dw1 = 2*(network_layer(x(idx), W, B) - g(idx))*df_dw1;
            dC_db1 = 2*(network_layer(x(idx), W, B) - g(idx))*df_db1;
            dC_dw2 = 2*(network_layer(x(idx), W, B) - g(idx))*df_dw2;
            dC_db2 = 2*(network_layer(x(idx), W, B) - g(idx))*df_db2;

            W1 = W(1) - eps*dC_dw1;
            B1 = B(1) - eps*dC_db1;
            W2 = W(2) - eps*dC_dw2;
            B2 = B(2) - eps*dC_db2;

            W(1) = W1;
            W(2) = W2;
            B(1) = B1;
            B(2) = B2;
        end
    end
    
end


