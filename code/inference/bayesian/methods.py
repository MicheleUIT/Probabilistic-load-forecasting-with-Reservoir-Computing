import numpy as np

from time import process_time
from pyro import clear_param_store
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from pyro.ops.stats import autocorrelation
from pyro.contrib.forecast.evaluate import eval_crps

from inference.bayesian.utils import check_calibration, check_convergence



#########################################
#   Stochastic Variational Inference    #
#########################################

def train_SVI(model, guide, X, Y, lr=0.03, num_iterations=120):
    # NOTE: should I change also optimizer and loss?
    optim = Adam({"lr": lr})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    # Clear the param store first, if it was already used
    clear_param_store()
    
    start_time = process_time()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(X, Y)
        if j % 20 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / Y.shape[0]))
    train_time = process_time() - start_time

    guide.requires_grad_(False)

    diagnostics = {
        "train_time": train_time,
    }

    return diagnostics


def pred_SVI(model, guide, X, num_samples, diagnostics):
    predictive = Predictive(model, guide=guide, num_samples=num_samples)(x=X, y=None)

    # diagnostics[] = 

    # return quantiles, times, calibration diagnostic, (what else?), as dictionary
    return predictive, diagnostics



#########################################
#       Monte Carlo Markov Chain        #
#########################################


def train_MCMC(model, X, Y, num_samples):
    ### FIXME: It seems that if the step size is too small the computation
    # time gets very big, even if the acc. prob is high. Why is that?

    # Clear the param store first, if it was already used
    # NOTE: This shouldn't be necessary for MCMC
    clear_param_store()

    # Use NUTS kernel
    nuts_kernel = NUTS(model)

    # Define a hook to log the acceptance rate and step size at each iteration
    ### NOTE: Does it work if num_chains > 1 ? I should check on the server
    step_size = []
    acc_rate = []
    def acc_rate_hook(kernel, params, stage, i):
        step_size.append(kernel.step_size) # Log step_size
        ### NOTE: _mean_accept_prob contains the acceptance probability
        # averaged over the time step n
        acc_rate.append(kernel._mean_accept_prob) # Log acceptance rate

    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=0, num_chains=1, hook_fn=acc_rate_hook)

    # Run the MCMC and compute the training time
    start_time = process_time()
    mcmc.run(X, Y)
    train_time = process_time() - start_time
    print(f"MCMC run time: {train_time/60} minutes.")

    # Get the split Gelman-Rubin factor and the effective sample size for each parameter
    eff_sample = {}
    GR_factor = {}
    for n, v in list(mcmc.diagnostics().items())[:-2]:
        eff_sample = v['n_eff']
        GR_factor[n] = v['r_hat']

    # Save diagnostics in dict
    diagnostics = {
        "step_size": np.asarray(step_size),
        "acceptance_rate": np.asarray(acc_rate),
        "train_time": train_time,
        "effective_sample_size": eff_sample,
        "split_gelman_rubin": GR_factor
    }

    return mcmc, diagnostics


def pred_MCMC(model, mcmc, X, Y, plot, diagnostics):

    samples = mcmc.get_samples()

    # Compute autocorrelation
    autocorr = {}
    for k, v in samples.items():
        autocorr[k] = autocorrelation(v)
    diagnostics["autocorrelation"] = autocorr

    # Find when it converged
    acc_rate = diagnostics["acceptance_rate"]
    warmup = check_convergence(samples, acc_rate, plot)
    print(f"MCMC converged at {warmup} steps.")

    ### TODO: Cut samples at warmup computed above
    # I probably need to loop through all samples and cut them since it's a dict
    # Perform inference
    predictive = Predictive(model, samples)(x=X, y=None) # num_samples?

    # Quantiles
    target_interval = 0.95  # draw and compute the 95% confidence interval
    q_low, q_hi = np.quantile(predictive["obs"].cpu().numpy().squeeze(), [(1-target_interval)/2, 1-(1-target_interval)/2], axis=0) # 40-quantile
    diagnostics["quantiles"] = [q_low, q_hi]

    # Compute calibration error
    diagnostics["cal_error"] = check_calibration(predictive, Y, folder="mcmc", plot=plot)

    # Continuous ranked probability score
    crps = eval_crps(predictive['obs'], Y.squeeze())
    diagnostics["crps"] = crps

    return predictive, diagnostics