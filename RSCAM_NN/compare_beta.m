function compare_beta

    % Learning rate, no. nodes and no. iterations
    eps = 1e-3;
    m = 10;
    NSteps = 10000;
    
    % Regularisation
    beta = [0.5 ,1, 20, 40];

    
    x = linspace(0, 1, 10000);

    
    err_DRM = zeros(10000, length(beta));
    err_DGM = zeros(10000, length(beta));
    for i = 1:length(beta)
        err_DRM(:, i) = initialise_DRM(m, NSteps, eps, beta(i), false);
        err_DGM(:, i) = initialise_DGM(m, NSteps, eps, beta(i), false);
    end
    
    figure(1)
    plot(x, err_DRM, 'LineWidth', 2.5)
    title('DRM: Error with varying regularisation')
    legend('\beta = 0.5' ,'\beta = 1', '\beta = 20', '\beta = 40')
    xlabel('x')
    ylabel('Error')
    
    figure(2)
    plot(x, err_DGM, 'LineWidth', 2.5)
    title('DGM: Error with varying regularisation')
    legend('\beta = 0.5' ,'\beta = 1', '\beta = 20', '\beta = 40')
    xlabel('x')
    ylabel('Error')

end