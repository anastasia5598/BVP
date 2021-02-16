function costvec = costfun(params, x, y, m, fun)
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
    
    %% integral approximation
    for j = 1:size(x,1)
        s = [x(j,:)'; zeros(m-2,1)];
        grads = [ones(2, 1); zeros(m-2,1)];

        %% Activate each layer
        z1 = w1 * s + b1;
        [a1, grada1] = activate(z1, fun);
        z2 = w2 * a1 + b2;
        [a2, grada2] = activate(z2, fun);
        u = w3 * (a2 + s) + b3;
        uprime = w3 * ((grada2 .* (w2 * (grada1 .* (w1 * grads)))) + grads);
        
        %% compute cost at each training point
        costvec(j) = 1/2 * uprime^2 - y * u;
        %costvec(j) = 1/2 * uprime^2 + 1/2\3*u^3 - y(j)*u;
        %costvec(j) = 1/2 * uprime^2 + 1/3 * u ^ 3 + 1/2 * u^2 + sin(u) - y(j)*u;
    end
    %% compute overall cost
    costvec = sum(costvec)/size(x,1);
end