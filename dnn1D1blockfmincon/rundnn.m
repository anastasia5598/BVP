function rundnn
    %% linear BVP's and solution
    f = @(x) 2; sol = @(x) (x - x .^ 2);
    %f = @(x) 6 * x; sol = @(x) x - x .^ 3;
    %f = @(x) exp(x); sol = @(x) - exp(x) - (1 - exp(1)) .* x + 1;
    %f = @(x) sin(x); sol = @(x) sin(x);
    %f = @(x) 12 * x .^ 2; sol = @(x) x - x.^4;
    %f = @(x) 20 * x .^ 3; sol = @(x) x - x.^5;
    %f = @(x) 2+6*x+12*x.^2+20*x.^3; sol = @(x) 4*x-x.^2-x.^3-x.^4-x.^5;
    
    %% nonlinear BVP's and solution
    %%%%%%% -u'' + u = f
    %f = @(x) x-x.^2+2; sol = @(x) x-x.^2;
    %f = @(x) 7*x-x.^3; sol = @(x) x- x.^3;
    %f = @(x)-(1-exp(1)).*x+1; sol = @(x) - exp(x) - (1 - exp(1)) .* x + 1;
    %f = @(x) 2*sin(x); sol = @(x) sin(x);
    
    %%%%%%% -u'' + u^2 = f
    %f = @(x) x.^4-2*x.^3+x.^2+2; sol = @(x) x-x.^2;
    %f = @(x) x.^6-2*x.^4+x.^2+6*x; sol = @(x) x-x.^3;
    %f = @(x) (sin(x)).^2 + sin(x); sol = @(x) sin(x);
    
    %%%%%%% -u'' + cos(u) = f
    %f = @(x) cos(x-x.^2) + 2; sol = @(x) x-x.^2;
    %f = @(x) cos(x-x.^3) + 6*x; sol = @(x) x-x.^3;
    %f = @(x) cos(-exp(x)-(1-exp(1)).*x+1)+exp(x); sol = @(x)-exp(x)-(1-exp(1)).*x+1;
    %f = @(x) cos(sin(x)) + sin(x); sol = @(x) sin(x);
    
    %%%%%%% -u'' + u + u^2 + cos(u) = f
    %f = @(x) x.^4-2*x.^3+x+cos(x-x.^2)+2; sol = @(x) x-x.^2;
    
    %% choose interval (a,b) and number of points
    a = 0; b = 1; 
    ntraining = 100; 
    nplot = 10000;
    
    %% choice of activation function
    % 1 - sigmoid; 2 - tanh; 3 - ReLU; 
    % 4 - identity; 5 - cubic; 6 - absolute value; 7 - softmax
    fun = 1;
    
    %% training data
    x = linspace(a,b,ntraining)';
    %[x,w]=gaussquad(a,b,ntraining);
    y = f(x);

    %% choice of neural network structure
    % choose number of blocks and number of neurons
    blocks = 1;
    m = 10;
    
    %% compute number of parameters 
    % nP - number of parameters
    nP = m+1;
    for i = 1:blocks
        nP = nP + 2 * (m * m + m);
    end
    
    %% generate initial parameters
    %rng(5000);
    Pzero = randn(nP, 1);
    %Pzero = ones(nP,1);

    %% train network
    options = optimoptions('fmincon');
    options.StepTolerance = 1e-8;
    options.OptimalityTolerance = 1e-8;
    options.MaxFunctionEvaluations=5e4;
    options.MaxIterations = 1e5;
    
    cost = @(Pzero) costfun(Pzero, x, y, m, fun);
    bcon = @(Pzero) bconditions(Pzero, x, m, fun);
    
    [params,fval,exitlfag,output] = fmincon(cost,Pzero,[],[],[],[],[],[],bcon,options);
    %[wb,fval,exitflag,output] = fmincon(@costv,wb,[],[],[],[],[],[],@bcon);
    params
    output,fval
    
    %% compare functions
    xx = linspace(a,b, nplot)';
    yy = evalDNN(xx, params, m, fun);
    sol = sol(xx);
    maxError = max(abs(sol - yy))
    
    %% plot the function and its DNN approximation
    figure(1); clf
    plot(xx, sol);
    hold on
    plot(xx, yy, 'r--');
    hold off
    grid on
    %legend('f(x)', 'DNN(x)');
    
    %% plot the error graph between the two functions
    figure(2);
    plot(xx, abs(yy - sol));
    grid on
    %legend('|f(x) - DNN(x)|');
end