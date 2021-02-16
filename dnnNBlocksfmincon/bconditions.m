function [bci, bc] = bconditions(params, xboundary, m, b, fun)
    d = size(xboundary, 2);
    
    %% impose boundary conditions
    for j = 1:size(xboundary,1)
        s = [xboundary(j,:)'; zeros(m-d, 1)];
        
        f = s;
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
            a1 = activate(z1, fun);
            z2 = w2 * a1 + b2;
            a2 = activate(z2, fun);
            f = a2 + f;
        end
        
        %% initial parameters for final linear combination
        wf = zeros(1, m);
        wf(:) = params((ni +1):(ni + m));
        ni = ni + m;
        bf = params(ni +1);
        
        %% final linear combination
        u = wf * f + bf; 
        
        %% compute cost at each training point
        bc(j) = u;
    end
    
    bci = [];
    bc = bc';
end