from inference.bayesian.methods import train_SVI, pred_SVI, train_MCMC, pred_MCMC
from inference.frequentist.methods import train_QR, pred_QR


def inference(config, model, guide, X_train, Y_train, X_test, Y_test, num_samples, inference_name, quantiles=None):
    if config.inference == "svi":
        diagnostics = train_SVI(model, guide, X_train, Y_train, config.lr, config.num_iterations)
        predictive, diagnostics = pred_SVI(model, guide, X_test, Y_test, num_samples, config.plot, diagnostics)
        return predictive, diagnostics

    elif config.inference == "mcmc":
        mcmc, diagnostics = train_MCMC(model, X_train, Y_train, num_samples)
        predictive, diagnostics = pred_MCMC(model, mcmc, X_test, Y_test, config.plot, diagnostics, inference_name)
        return mcmc, predictive, diagnostics

    elif config.inference == "q_regr":
        diagnostics = train_QR(model, X_train, Y_train, config.lr, config.num_iterations, quantiles)
        predictive, diagnostics = pred_QR(model, X_test, Y_test, config.plot, diagnostics, quantiles)
        return predictive, diagnostics

    else:
        raise ValueError(f"{config.inference} method not implemented.")
    
    
