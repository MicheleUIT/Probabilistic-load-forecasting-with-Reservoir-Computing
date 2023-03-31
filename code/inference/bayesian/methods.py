import torch

import numpy as np

from time import process_time
from pyro import clear_param_store
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from pyro.ops.stats import autocorrelation
from pyro.contrib.forecast.evaluate import eval_crps
from tqdm import trange

from inference.bayesian.utils import check_convergence, acceptance_rate, calibrate, compute_coverage_len, num_eval_crps
from inference.early_stopping import EarlyStopping



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


def pred_SVI(model, guide, X_val, Y_val, X_test, Y_test, num_samples, plot, sweep, diagnostics, quantiles):

    # Use validation set for hyperparameters tuning
    if sweep:
        X, Y = X_val, Y_val
        X2, Y2 = X_test, Y_test
    else:
        X, Y = X_test, Y_test
        X2, Y2 = X_val, Y_val

    # Perform inference
    start_time = process_time()
    predictive = Predictive(model, guide=guide, num_samples=num_samples)(x=X, y=None)
    inference_time = process_time() - start_time
    diagnostics["inference_time"] = inference_time

    # Compute calibration error
    predictive2 = Predictive(model, guide=guide, num_samples=num_samples)(x=X2, y=None)
    # Calibrate
    cal_error, new_cal_error, new_quantiles = calibrate(predictive["obs"].cpu().numpy().squeeze(), 
                                                        predictive2["obs"].cpu().numpy().squeeze(), 
                                                        Y, Y2, quantiles, folder="svi", plot=plot)
    diagnostics["cal_error"] = cal_error
    diagnostics["new_cal_error"] = new_cal_error

    # Width at 0.95 quantile
    q_low, q_hi = np.quantile(predictive["obs"].cpu().numpy().squeeze(), [quantiles[2], quantiles[-2]], axis=0) # 40-quantile
    diagnostics["width"] = np.mean(q_hi - q_low)
    # After calibration
    new_q_low, new_q_hi = np.quantile(predictive["obs"].cpu().numpy().squeeze(), [new_quantiles[2], new_quantiles[-2]], axis=0) # 40-quantile
    diagnostics["new_width"] = np.mean(new_q_hi - new_q_low)

    # Check coverage with 95% quantiles
    coverage, avg_length = compute_coverage_len(Y.cpu().numpy(), q_low, q_hi)
    diagnostics["coverage"] = coverage
    diagnostics["avg_length"] = avg_length
    # Re-compute after calibration
    coverage, avg_length = compute_coverage_len(Y.cpu().numpy(), new_q_low, new_q_hi)
    diagnostics["new_coverage"] = coverage
    diagnostics["new_avg_length"] = avg_length

    # Mean Squared Error wrt the median
    median = np.quantile(predictive["obs"].cpu().numpy().squeeze(), quantiles[int(len(quantiles)/2)], axis=0) # median
    mse = np.mean((median-Y.cpu().numpy())**2)
    diagnostics["mse"] = mse
    # After calibration
    median = np.quantile(predictive["obs"].cpu().numpy().squeeze(), new_quantiles[int(len(new_quantiles)/2)], axis=0) # median
    mse = np.mean((median-Y.cpu().numpy())**2)
    diagnostics["new_mse"] = mse

    # Empirical continuous ranked probability score
    e_crps = eval_crps(predictive['obs'], Y.squeeze())
    diagnostics["e_crps"] = e_crps
    # Numerical continuous ranked probability score
    tau = np.quantile(predictive["obs"].cpu().numpy().squeeze(), quantiles, axis=0)
    n_crps = num_eval_crps(quantiles, tau, Y.cpu().squeeze().numpy())
    diagnostics["crps"] = n_crps
    # Compute CRPS after calibration
    tau = np.quantile(predictive["obs"].cpu().numpy().squeeze(), new_quantiles, axis=0)
    new_n_crps = num_eval_crps(new_quantiles, tau, Y.cpu().squeeze().numpy())
    diagnostics["new_crps"] = new_n_crps

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


def pred_MCMC(model, samples, X_val, Y_val, X_test, Y_test, plot, sweep, diagnostics, inference_name):

    # Use validation set for hyperparameters tuning
    if sweep:
        X, Y = X_val, Y_val
        X2, Y2 = X_test, Y_test
    else:
        X, Y = X_test, Y_test
        X2, Y2 = X_val, Y_val

    # Find when it converged
    acc_rate = diagnostics["acceptance_rate"]
    _, warmup, samples, GR_factors, ess = check_convergence(samples, acc_rate, inference_name, plot)
    print(f"MCMC converged at {warmup} steps.")

    diagnostics["gelman_rubin"] = GR_factors
    diagnostics["effective_sample_size"] = ess

    # Perform inference
    predictive = Predictive(model, samples)(x=X, y=None)

    # Quantiles
    target_interval = 0.95  # draw and compute the 95% confidence interval
    q_low, q_hi = np.quantile(predictive["obs"].cpu().numpy().squeeze(), [(1-target_interval)/2, 1-(1-target_interval)/2], axis=0) # 40-quantile
    diagnostics["width95"] = q_hi - q_low
    
    target_interval = 0.99  # draw and compute the 99% confidence interval
    q_low, q_hi = np.quantile(predictive["obs"].cpu().numpy().squeeze(), [(1-target_interval)/2, 1-(1-target_interval)/2], axis=0) # 40-quantile
    diagnostics["width99"] = q_hi - q_low

    # Mean Squared Error
    mean = np.mean(predictive["obs"].cpu().numpy().squeeze(), axis=0)
    mse = np.mean((mean-Y.cpu().numpy())**2)
    diagnostics["mse"] = mse

    # Compute calibration error
    predictive2 = Predictive(model, samples)(x=X2, y=None)
    # Calibrate
    cal_error, new_cal_error = calibrate(predictive, predictive2, Y, Y2, folder="mcmc", plot=plot)
    diagnostics["cal_error"] = cal_error
    diagnostics["new_cal_error"] = new_cal_error

    # FIXME: use "discrete" CRPS and add CRPS after calibration
    # Continuous ranked probability score
    crps = eval_crps(predictive['obs'], Y.squeeze())
    diagnostics["crps"] = crps

    return predictive, diagnostics



########################
#       Dropout        #
########################


def train_DO(model, X, Y, X_val, Y_val, lr, epochs):

    torch_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse_loss = torch.nn.MSELoss()

    # initialize the early_stopping object
    checkpoint_path = "./checkpoints/DO/"
    early_stopping = EarlyStopping(patience=20, verbose=False, path=checkpoint_path)

    start_time = process_time()

    with trange(epochs) as t:
        for epoch in t:
            model.train()
            torch_optimizer.zero_grad()
            loss = mse_loss(model(X).squeeze(), Y)
            loss.backward()
            torch_optimizer.step()

            # display progress bar
            t.set_description(f"Epoch {epoch+1}")
            t.set_postfix({"loss":float(loss / Y.shape[0])})

            # Early stopping
            model.eval()
            valid_loss = mse_loss(model(X_val), Y_val).item()

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            
            model.train()
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

    train_time = process_time() - start_time

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path + "checkpoint.pt"))

    diagnostics = {
        "train_time": train_time,
        "final_loss": float((loss / Y.shape[0]).detach().cpu())
    }

    return diagnostics


def pred_DO(model, X_val, Y_val, X_test, Y_test, num_samples, plot, sweep, diagnostics, quantiles):

    # Use validation set for hyperparameters tuning
    if sweep:
        X, Y = X_val, Y_val
        X2, Y2 = X_test, Y_test
    else:
        X, Y = X_test, Y_test
        X2, Y2 = X_val, Y_val

    # Perform inference
    start_time = process_time()
    predictive = []
    for n in range(num_samples):
        predictive.append(model(X).detach().cpu().squeeze())
    predictive = np.stack(predictive, axis=0)
    inference_time = process_time() - start_time
    diagnostics["inference_time"] = inference_time

    # Compute calibration error
    predictive2 = []
    for n in range(num_samples):
        predictive2.append(model(X2).detach().cpu().squeeze())
    predictive2 = np.stack(predictive2, axis=0)

    # Calibrate
    cal_error, new_cal_error, new_quantiles = calibrate(predictive, predictive2, Y, Y2, quantiles, folder="dropout", plot=plot)
    diagnostics["cal_error"] = cal_error
    diagnostics["new_cal_error"] = new_cal_error

    # Width at 0.95 quantile
    q_low, q_hi = np.quantile(predictive, [quantiles[2], quantiles[-2]], axis=0) # 40-quantile
    diagnostics["width"] = np.mean(q_hi - q_low)
    # After calibration
    new_q_low, new_q_hi = np.quantile(predictive, [new_quantiles[2], new_quantiles[-2]], axis=0) # 40-quantile
    diagnostics["new_width"] = np.mean(new_q_hi - new_q_low)

    # Check coverage with 95% quantiles
    coverage, avg_length = compute_coverage_len(Y.cpu().numpy(), q_low, q_hi)
    diagnostics["coverage"] = coverage
    diagnostics["avg_length"] = avg_length
    # Re-compute after calibration
    coverage, avg_length = compute_coverage_len(Y.cpu().numpy(), new_q_low, new_q_hi)
    diagnostics["new_coverage"] = coverage
    diagnostics["new_avg_length"] = avg_length

    # Mean Squared Error wrt the median
    median = np.quantile(predictive, quantiles[int(len(quantiles)/2)], axis=0) # median
    mse = np.mean((median-Y.cpu().numpy())**2)
    diagnostics["mse"] = mse
    # After calibration
    median = np.quantile(predictive, new_quantiles[int(len(new_quantiles)/2)], axis=0) # median
    mse = np.mean((median-Y.cpu().numpy())**2)
    diagnostics["new_mse"] = mse

    # Empirical continuous ranked probability score
    e_crps = eval_crps(torch.from_numpy(predictive).cpu(), Y.cpu().squeeze())
    diagnostics["e_crps"] = e_crps
    # Numerical continuous ranked probability score
    tau = np.quantile(predictive, quantiles, axis=0)
    n_crps = num_eval_crps(quantiles, tau, Y.cpu().squeeze().numpy())
    diagnostics["crps"] = n_crps
    # Compute CRPS after calibration
    tau = np.quantile(predictive, new_quantiles, axis=0)
    new_n_crps = num_eval_crps(new_quantiles, tau, Y.cpu().squeeze().numpy())
    diagnostics["new_crps"] = new_n_crps


    return predictive, diagnostics