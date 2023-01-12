from bayesian.methods import train_SVI, pred_SVI, train_MCMC, pred_MCMC
from frequentist.methods import train_QR, pred_QR


def inference(config, model, guide, X_train, Y_train, X_test, Y_test, num_samples):
    if config.inference == "svi":
        ### TODO: implement SVI
        diagnostics = train_SVI(model, guide, X_train, Y_train, config.lr, config.num_iterations)
        predictive, diagnostics = pred_SVI()
    elif config.inference == "mcmc":
        mcmc, diagnostics = train_MCMC(model, X_train, Y_train, num_samples)
        predictive, diagnostics = pred_MCMC(model, mcmc, X_test, Y_test, config.plot, diagnostics)
    elif config.inference == "q_regr":
        diagnostics = train_QR(model, X_train, Y_train, config.quantile, config.lr, config.num_iterations)
        _ = pred_QR(model, X_test, Y_test)
    else:
        raise ValueError(f"{config.inference} method not implemented.")
    
    return mcmc, predictive, diagnostics
