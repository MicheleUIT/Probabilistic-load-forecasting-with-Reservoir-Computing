import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from time import process_time
from pyro import clear_param_store
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from utils import check_calibration


def inference(config, model, guide, X_train, Y_train, X_test, Y_test):
    if config.inference == "svi":
        ### TODO: implement SVI
        train_SVI()
        predictive, diagnostics = pred_SVI()
    elif config.inference == "mcmc":
        mcmc, diagnostics = train_MCMC(model, X_train, Y_train)
        predictive, diagnostics = pred_MCMC(model, mcmc, X_test, Y_test, diagnostics)
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

    diagnostics = {}

    # return quantiles, times, calibration diagnostic, (what else?), as dictionary
    return predictive, diagnostics



#########################################
#       Monte Carlo Markov Chain        #
#########################################


def train_MCMC(model, X, Y):
    # Clear the param store first, if it was already used
    clear_param_store()

    # Use NUTS kernel
    nuts_kernel = NUTS(model)

    # define a hook to log the acceptance rate and step size at each iteration
    ### NOTE: Does it work if num_chains > 1 ? I should check on the server
    step_size = []
    acc_rate = []
    def acc_rate_hook(kernel, params, stage, i):
        step_size.append(kernel.step_size) # Log step_size
        ### NOTE: _mean_accept_prob contains the acceptance probability
        # averaged over the time step n
        acc_rate.append(kernel._mean_accept_prob) # Log acceptance rate

    mcmc = MCMC(nuts_kernel, num_samples=300, warmup_steps=0, num_chains=1, hook_fn=acc_rate_hook)

    # run the MCMC and compute the training time
    start_time = process_time()
    ### FIXME: # It seems that if the step size is too small the computation
    # time gets very big, even if the acc. prob is high. Why is that?
    mcmc.run(X, Y)
    train_time = process_time() - start_time
    print(f"MCMC run time: {train_time/60} minutes.")

    diagnostics = {
        "step_size": np.asarray(step_size),
        "acceptance_rate": np.asarray(acc_rate),
        "train_time": train_time
    }

    return mcmc, diagnostics


def pred_MCMC(model, mcmc, X, Y, diagnostics):

    ### TODO: Compute other diagnostics related to convergence with 'samples'
    # Do I need to compute also the inference time?
    samples = mcmc.get_samples()

    # Find when it converged
    acc_rate = diagnostics["acceptance_rate"]
    warmup = convergence_check(samples, acc_rate)

    ### TODO: Cut samples at warmup computed above
    # I probably need to loop through all samples and cut them since it's a dict
    # Perform inference
    predictive = Predictive(model, samples)(x=X, y=None) # num_samples?

    # Quantiles
    target_interval = 0.95  # draw and compute the 95% confidence interval
    q_low, q_hi = np.quantile(predictive["obs"].cpu().numpy().squeeze(), [(1-target_interval)/2, 1-(1-target_interval)/2], axis=0) # 40-quantile

    # Compute calibration error
    diagnostics["cal_error"] = check_calibration(predictive, Y, folder="mcmc")

    ### TODO: return quantiles(?), times, (what else?), as dictionary

    return predictive, diagnostics


def convergence_check(samples, acc_rate):
    conv = []
    for name, param in samples.items():
        conv.append(trace_plot(param, name))
    
    conv.append(trace_plot(acc_rate, "acceptance_rate"))
    
    return max(conv)


def trace_plot(variable, name, plot=False):
    # Compute a moving average of the rate of change of ´variable´
    r = np.diff(variable)
    av_r = np.convolve(r, np.ones(10)/10, mode='valid')
    x = np.asarray(range(len(av_r)))

    # Change color when ´av_r´ goes below 5%
    cond = np.abs(av_r/(np.max(av_r)-np.min(av_r)))<0.05
    col = np.where(cond, 'r', 'b')

    # Threshold for convergence set where ´av_r´ is consistently under 5%
    t = len(cond) - np.argmin(np.flip(cond))

    # Create the two plots
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, figsize=(15,10))
        ax1.set_title(f"{name} trace plot")
        ax1.grid()
        ax1.plot(variable)
        ax1.vlines(t, ymin=np.min(variable), ymax=np.max(variable), colors='g', linestyles='dashed', label="Convergence point")
        ax2.set_title("Moving average of the rate of change")
        ax2.grid()
        ax2.scatter(x, av_r, color=col)
        ax2.vlines(t, ymin=np.min(av_r), ymax=np.max(av_r), colors='g', linestyles='dashed', label="Convergence point")
        
        # Save plots
        save_path = f'./results/plots/mcmc/'
        Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
        plt.savefig(f'{save_path}{name}.png')
        plt.clf()

    return t