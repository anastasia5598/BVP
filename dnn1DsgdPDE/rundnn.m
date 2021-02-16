function rundnn
    % runndnn - trains a neural netowrk and compares it with the training
    %           data
    % choices - function to approximate, interval on which to approximate
    %           the function, number of training points, number of points 
    %           at which to evaluate approximation, activation function, 
    %           structure of neural network, learning rate, number of
    %           iterations
    % produces graph of the two functions (trianing data and neural
    % network approximation) and graph of the error between the two
    
    %%%%%%%%%%% choose function to approximate
    %f = @(x) besselj(0,x);
    %f=@(x) exp(x);
    %f=@erf;
    %f=@(x) sin(x).*exp(-x);
    %f=@(x) log(abs(x)+1e-1);
    %f=@(x) x.^3+x.^2-x-1;
    %f=@(x) 1+x+x.*x;
    %f = @(x) 1 ./ (1 + 25 * x.^2);
    %f = @(x) ((x.^4)/4) - ((x.^2)/8);
    f = @(x) 1; sol = @(x) 1/2 * (x - x .^2);
    
    %%%%%%%%%%% choose interval (a,b) and number of points
    a = 0; b = 1; 
    % ntraining - number of training points
    ntraining = 10; 
    % nplot - number of points at which to evaluate the functions
    nplot = 10000;
    
    %%%%%%%%%%% training data
    x = linspace(a,b,ntraining)';
    y = f(x);

    %%%%%%%%%%% choice of activation function
    % 1 - sigmoid; 2 - tanh; 3 - ReLU; 
    % 4 - identity; 5 - cubic; 6 - absolute value; 7 - softmax
    fun = 1;
    
    %%%%%%%%%%% choice of learning rate
    eta = 5e-3;
    beta = 21;
    %%%%%%%%%%% choice of number of iterations
    Niter = 1e5;
    
    %%%%%%%%%%%% choice of neural network structure
    % choose number of layers by specifying number of neurons at each layer
    layers = [4];
    % arrange dnn layout in matrix form, i.e.
        % ith row specifies the dimensions of the weights matrix at the ith
        % layer
    weights = [[layers(:);1] [1;layers(:)]];
        % ith row specifies the dimensions of the biases vector at the ith
        % layer
    biases = [layers(:);1];
    
    %%%%%%%%%%% train network
    params = dnn(x,y, weights, biases, fun, eta, Niter, beta);
    
    %%%%%%%%%%% compare functions
    xx = linspace(a,b, nplot)';
    yy = evalDNN(xx, params, weights, biases, fun);
    sol = sol(xx);
    
    %%%%%%%%%%%% plot the function and its DNN approximation
    figure(1); clf
    plot(xx, sol);
    hold on
    plot(xx, yy, 'r--');
    hold off
    grid on
    %legend('f(x)', 'DNN(x)');
    
    %%%%%%%%%%% plot the error graph between the two functions
    %figure(2);
    %semilogy(xx, abs(yy - f(xx)));
    %grid on
    %legend('|f(x) - DNN(x)|');
    
    maxError = max(abs(sol-yy));
    maxError
    minError = min(abs(sol-yy));
    minError
end