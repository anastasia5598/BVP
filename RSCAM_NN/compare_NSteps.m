function compare_NSteps

    % Nodes, Regularisation and Learning Rate
    m = 10;
    beta = 30;
    eps = 1e-3; 

    % No. Iterations
    NSteps = [10, 100, 1e4, 1e6];
    
    x = linspace(0, 1, 10000);    
    
    err_DRM = zeros(10000, length(NSteps));
    err_DGM = zeros(10000, length(NSteps));
    for i = 1:length(NSteps)
        err_DRM(:, i) = initialise_DRM(m, NSteps(i), eps, beta, false);
        err_DGM(:, i) = initialise_DGM(m, NSteps(i), eps, beta, false);
    end
    
    figure(1)
    plot(x, err_DGM, 'LineWidth',2.5)
    title('DGM: Error with varying NSteps')
    xlabel('x')
    ylabel('Error')
    legend('NSteps = 10', 'NSteps = 100', 'NSteps = 1e4', 'NSteps = 1e6')
    
    figure(2)
    plot(x, err_DRM, 'LineWidth',2.5)
    title('DRM: Error with varying NSteps')
    xlabel('x')
    ylabel('Error')
    legend('NSteps = 10', 'NSteps = 100', 'NSteps = 1e4', 'NSteps = 1e6')
end