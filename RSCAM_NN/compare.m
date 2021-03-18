function compare

    % Get errors from both functions
    err_DRM = initialise_DRM;
    err_DGM = initialise_DGM;

    x = linspace(0, 1, 10000);
    
    % Plot errors on
    figure(1)
    plot(x, err_DRM, 'r-')
    hold on 
    plot(x, err_DGM, 'b')
    hold off
    
    title('Error of approximate solutions')
    legend('DRM', 'DGM')
    xlabel('x')
    ylabel('Error')

end