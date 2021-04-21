function compare_eps

    % Nodes, Regularisation and no. iterations
    m = 10;
    beta = 30;
    NSteps = 1e4;

    % Learning rate
    eps = [1e-1, 1e-3, 1e-5, 1e-7];  
    
    x = linspace(0, 1, 10000);    
    
    err_DRM = zeros(10000, length(eps));
    err_DGM = zeros(10000, length(eps));
    for i = 1:length(eps)
        err_DRM(:, i) = initialise_DRM(m, NSteps, eps(i), beta, false);
        err_DGM(:, i) = initialise_DGM(m, NSteps, eps(i), beta, false);
    end
    
    figure(1)
    plot(x, err_DGM, 'LineWidth',2.5)
    title('DGM: Error with varying learning rate')
    xlabel('x')
    ylabel('Error')
    legend('\epsilon = 1e-1', '\epsilon = 1e-3', '\epsilon = 1e-5', '\epsilon = 1e-7')
    
    figure(2)
    plot(x, err_DRM, 'LineWidth',2.5)
    title('DRM: Error with varying learning rate')
    xlabel('x')
    ylabel('Error')
    legend('\epsilon = 1e-1', '\epsilon = 1e-3', '\epsilon = 1e-5', '\epsilon = 1e-7')
end