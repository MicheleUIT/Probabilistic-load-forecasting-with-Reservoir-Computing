from inference.bayesian.methods import train_SVI, pred_SVI, train_MCMC, pred_MCMC, train_DO, pred_DO
from inference.frequentist.methods import train_QR, pred_QR


def inference(config, model, guide, X_train, Y_train, X_val, Y_val, X_test, Y_test, quantiles=None):
    if config.inference == "svi":
        diagnostics = train_SVI(model, guide, X_train, Y_train, config.lr, config.num_iterations)
        predictive, diagnostics = pred_SVI(model, guide, X_val, Y_val, X_test, Y_test, config.num_samples, config.plot, config.sweep, diagnostics, quantiles)
        return predictive, diagnostics

    elif config.inference == "mcmc" or config.inference == "ssvs":
        samples, diagnostics = train_MCMC(model, X_train, Y_train, config.num_chains, config.num_samples)
        predictive, diagnostics = pred_MCMC(model, samples, X_val, Y_val, X_test, Y_test, config.plot, config.sweep, diagnostics, config.inference)
        return predictive, diagnostics

    elif config.inference == "q_regr":
        diagnostics = train_QR(model, X_train, Y_train, X_val, Y_val, config.lr, config.num_iterations, quantiles)
        predictive, diagnostics = pred_QR(model, X_val, Y_val, X_test, Y_test, config.plot, config.sweep, diagnostics, quantiles)
        return predictive, diagnostics
    
    elif config.inference == "dropout":
        diagnostics = train_DO(model, X_train, Y_train, X_val, Y_val, config.lr, config.num_iterations)
        predictive, diagnostics = pred_DO(model, X_val, Y_val, X_test, Y_test, config.num_samples, config.plot, config.sweep, diagnostics, quantiles)
        return predictive, diagnostics

    else:
        raise ValueError(f"{config.inference} method not implemented.")
    
    
