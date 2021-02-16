function [bci, bc] = bconditions(params, x, m, fun)
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
    
    %% impose boundary conditions
    for j = 1:size(x)
        s = [x(j,:)'; zeros(m-2,1)];
        
        %% Activate each layer
        z1 = w1 * s + b1;
        a1 = activate(z1, fun);
        z2 = w2 * a1 + b2;
        a2 = activate(z2, fun);
        u = w3 * (a2 + s) + b3;
        
        %% compute cost at each training point
        bc(j) = u;
    end
    bci = [];
    bc = bc';
end