function run_DGM
    
    eps = 0.001;
    beta = 30;
    
    m = 16;
    NSteps = 10000;
    plot = true;
    
    initialise_DGM(m, NSteps, eps, beta, plot);
    
end
    
    