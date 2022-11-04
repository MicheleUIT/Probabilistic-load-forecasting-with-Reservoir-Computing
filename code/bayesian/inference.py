from time import process_time
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS


def inference(config):
    if config.inference == "svi":
        train_SVI()
        pred_SVI()
    elif config.inference == "mcmc":
        train_MCMC()
        pred_MCMC()
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

def train_MCMC(model, guide, X, Y):
    # pyro.clear_param_store() # do we need to clear the param store first?
    nuts_kernel = NUTS(model)

    mcmc = MCMC(nuts_kernel, num_samples=3000, warmup_steps=1000, num_chains=1)

    start_time = process_time()
    mcmc.run(X, Y)
    train_time = process_time() - start_time


def pred_MCMC(mcmc, X):
    samples = mcmc.get_samples()
    # return quantiles, times, calibration diagnostic, (what else?), as dictionary
