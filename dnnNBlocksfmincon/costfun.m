function costvec = costfun(params, x, y, m, b, fun)
    d = size(x,2);
    
    %% integral approximation
    for j = 1:size(x,1)
        s = [x(j,:)'; zeros(m-d,1)];
        grads = [ones(d, 1); zeros(m-d,1)];
        
        f = s; gradf = grads;
        %% Activate each layer
        ni=0;
        for i = 1:b
            %% unpack weights and biases
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
            
            %% activate
            z1 = w1 * f + b1;
            [a1, grada1] = activate(z1, fun);
            z2 = w2 * a1 + b2;
            [a2, grada2] = activate(z2, fun);
            gradf = grada2 .* (w2 * (grada1 .* (w1 * gradf))) + gradf;
            f = a2 + f;
        end
        
        %% initial parameters for final linear combination
        wf = zeros(1, m);
        wf(:) = params((ni +1):(ni + m));
        ni = ni + m;
        bf = params(ni +1);
        
        %% final linear combination
        u = wf * f + bf; 
        uprime = wf * gradf;
        
        %% compute cost at each training point
        costvec(j) = 1/2 * uprime^2 - y * u;
        %costvec(j) = 1/2 * uprime^2 + 1/3*u^3 - y(j)*u;
        %costvec(j) = 1/2 * uprime^2 + 1/3 * u ^ 3 + 1/2 * u^2 + sin(u) - y(j)*u;
    end
    %% compute overall cost
    costvec = sum(costvec)/size(x,1);
end