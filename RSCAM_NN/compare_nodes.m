function compare_nodes

    % Learning rate and no. iterations
    eps = 1e-3;
    beta = 30;

    m = [8, 10, 12];
    NSteps = 1e4;
    x = linspace(0, 1, 10000);    
    
    err_DRM = zeros(10000, length(m));
    err_DGM = zeros(10000, length(m));
    for i = 1:length(m)
        err_DRM(:, i) = initialise_DRM(m(i), NSteps, eps, beta, false);
        err_DGM(:, i) = initialise_DGM(m(i), NSteps, eps, beta, false);
    end
    
    figure(1)
    plot(x, err_DGM, 'LineWidth',2.5)
    title('DGM: Error with varying number of nodes')
    xlabel('x')
    ylabel('Error')
    legend('m = 8', 'm = 10', 'm = 12')
    
    figure(2)
    plot(x, err_DRM, 'LineWidth',2.5)
    title('DRM: Error with varying number of nodes')
    xlabel('x')
    ylabel('Error')
    legend('m = 8', 'm = 10', 'm = 12')
end