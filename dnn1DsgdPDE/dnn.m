function Pfinal = dnn(x,y, weights, biases, fun, eta, Niter, beta)
    % dnn - trains deep neural network
    % x and y - input-output training set
    % weights and biases - the structure of the neural network, see rundnn
    %                      for details
    % fun - the choice of activation function
    % returns the final (trained) parameteres of the neural network
    
    %%%%%%%%% compute number of parameters 
    % nP - number of parameters
    nP = 0;
    for i = 1:size(weights)
        nP = nP + prod(weights(i,:)) + biases(i);
    end
    
    % generate initial random parameters
    rng(5000);
    Pzero = 0.5 * randn(nP, 1);
    
    % train network / adjust parameters using gradient descent
    Pfinal = costfun(Pzero, x, y, weights, biases, fun, eta, Niter, beta);

end