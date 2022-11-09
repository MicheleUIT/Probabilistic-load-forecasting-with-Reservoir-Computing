import numpy as np

from time import process_time
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS


def inference(config, model, guide, X_train, Y_train, X_test, Y_test):
    if config.inference == "svi":
        ### TODO: implement SVI
        train_SVI()
        pred_SVI()
    elif config.inference == "mcmc":
        mcmc, diagnostics = train_MCMC(model, X_train, Y_train)
        pred_MCMC(model, mcmc, X_test, diagnostics)
    elif config.inference == "q_regr":
        raise ValueError(f"{config.inference} method not implemented.")
    else:
        raise ValueError(f"{config.inference} method not implemented.")




#########################################
#   Stochastic Variational Inference    #
#########################################

def train_SVI(model, guide, X, Y, lr=0.03, num_iterations=120):
    # should I change also optimizer and loss?
    optim = Adam({"lr": lr})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    num_iterations = num_iterations

    # pyro.clear_param_store() # do we need to clear the param store first?
    start_time = process_time()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(X, Y)
        if j % 20 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / Y.shape[0]))
    train_time = process_time() - start_time

    guide.requires_grad_(False)


def pred_SVI(model, guide, X, num_samples):
    predictive = Predictive(model, guide=guide, num_samples=num_samples)(x=X, y=None)
    # return quantiles, times, calibration diagnostic, (what else?), as dictionary



#########################################
#       Monte Carlo Markov Chain        #
#########################################


def train_MCMC(model, X, Y):
    # pyro.clear_param_store() # do we need to clear the param store first?
    nuts_kernel = NUTS(model)

    # define a hook to log the acceptance rate at each iteration
    ### ISSUE: Does it work if num_chains > 1 ? I should check on the server
    acc_rate = []
    def acc_rate_hook(kernel, params, stage, i):
        acc_rate.append(kernel._mean_accept_prob)

    mcmc = MCMC(nuts_kernel, num_samples=3000, warmup_steps=1000, num_chains=1, hook_fn=acc_rate_hook)

    # run the MCMC and compute the training time
    start_time = process_time()
    mcmc.run(X, Y)
    train_time = process_time() - start_time

    diagnostics = {
        "acceptance_rate": acc_rate,
        "train_time": train_time
    }

    return mcmc, diagnostics


def pred_MCMC(model, mcmc, X, diagnostics):

    ### TODO: Compute other diagnostics related to convergence with 'samples'
    samples = mcmc.get_samples()

    # Perform inference
    predictive = Predictive(model, samples)(x=X, y=None) # num_samples?

    # Quantiles
    target_interval = 0.95  # draw and compute the 95% confidence interval
    q_low, q_hi = np.quantile(predictive["obs"].cpu().numpy().squeeze(), [(1-target_interval)/2, 1-(1-target_interval)/2], axis=0) # 40-quantile


    # return quantiles, times, calibration diagnostic, (what else?), as dictionary
