import numpy as np

from time import process_time
from pyro import clear_param_store
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from pyro.ops.stats import autocorrelation
from pyro.contrib.forecast.evaluate import eval_crps
from tqdm import trange

from inference.bayesian.utils import check_calibration, check_convergence, acceptance_rate



#########################################
#   Stochastic Variational Inference    #
#########################################

def train_SVI(model, guide, X, Y, lr=0.03, num_iterations=120):
    
    optim = Adam({"lr": lr})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    # Clear the param store first, if it was already used
    clear_param_store()
    
    start_time = process_time()

    with trange(num_iterations) as t:
        for j in t:
            # calculate the loss and take a gradient step
            loss = svi.step(X, Y)

            # display progress bar
            t.set_description(f"Epoch {j+1}")
            t.set_postfix({"loss":loss / Y.shape[0]})

    train_time = process_time() - start_time

    guide.requires_grad_(False)

    diagnostics = {
        "train_time": train_time,
        "final_loss": loss / Y.shape[0]
    }

    return diagnostics


def pred_SVI(model, guide, X, Y, num_samples, plot, diagnostics):

    # Perform inference
    predictive = Predictive(model, guide=guide, num_samples=num_samples)(x=X, y=None)

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



#########################################
#       Monte Carlo Markov Chain        #
#########################################


def train_MCMC(model, X, Y, num_chains, num_samples):

    # Define a hook to log the acceptance rate and step size at each iteration
    step_size = []
    acc_rate = []
    def acc_rate_hook(kernel, params, stage, i):
        step_size.append(kernel.step_size) # Log step_size
        # _mean_accept_prob contains the acceptance probability
        # averaged over the time step n
        acc_rate.append(kernel._mean_accept_prob) # Log acceptance rate

    # Using num_chains>1 on the CPU affects memory performances,
    # on GPU the multiprocessing doesn't seem to work properly
    # so we run multiple chains sequentially

    samples = []
    train_time = []

    for n in range(num_chains):
        # Use NUTS kernel
        nuts_kernel = NUTS(model)

        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=0, num_chains=1, hook_fn=acc_rate_hook)

        # Run the MCMC and compute the training time
        start_time = process_time()
        mcmc.run(X, Y)
        train_time.append(process_time() - start_time)

        samples.append(mcmc.get_samples())
    
    step_size = np.asarray(step_size).reshape((num_chains,num_samples))
    acc_rate = acceptance_rate(np.asarray(acc_rate).reshape((num_chains,num_samples)))

    # Compute autocorrelation for each chain and each parameter separately
    autocorrs = []
    for chain in samples:
        autocorr = {}
        for k, v in chain.items():
            autocorr[k] = autocorrelation(v)

    # Save diagnostics in dict
    diagnostics = {
        "step_size": step_size,
        "acceptance_rate": acc_rate,
        "train_time": np.asarray(train_time).mean(), # NOTE: should I change the definition of training time?
        "autocorrelation": autocorrs
    }

    return samples, diagnostics


def pred_MCMC(model, samples, X, Y, plot, diagnostics, inference_name):

    # Find when it converged
    acc_rate = diagnostics["acceptance_rate"]
    warmup, samples, GR_factors, ess = check_convergence(samples, acc_rate, inference_name, plot)
    print(f"MCMC converged at {warmup} steps.")

    diagnostics["gelman_rubin"] = GR_factors
    diagnostics["effective_sample_size"] = ess

    # Perform inference
    predictive = Predictive(model, samples)(x=X, y=None)

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