function y = evalDNN(x, params, m, fun)
    %% unpack weights and biases
    ni = 0;
    w1 = zeros(m, m);
    w1(:) = params((ni +1):(ni + m * m));
    ni = ni + m*m;
    b1 = params((ni +1):(ni + m));
    ni = ni + m;
    w2 = zeros(m, m);
    w2(:) = params((ni +1):(ni + m * m));
    ni = ni + m*m;
    b2 = params((ni +1):(ni + m));
    ni = ni + m;
    
    % initial parameters for final linear combination
    w3 = zeros(1, m);
    w3(:) = params((ni +1):(ni + m));
    ni = ni + m;
    b3 = params(ni +1);
    
    %% evaluate DNN at each x
    for i = 1 : size(x)
        y = [x(i); zeros(m-1,1)];
        z1 = w1 * y + b1;
        a1 = activate(z1, fun);
        z2 = w2 * a1 + b2;
        a2 = activate(z2, fun);
        z(i) = w3 * (a2 + y) + b3;
    end
    y = z';
end