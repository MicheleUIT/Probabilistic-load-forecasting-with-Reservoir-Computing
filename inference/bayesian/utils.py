import torch
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from pyro.ops.stats import gelman_rubin, effective_sample_size
from sklearn.isotonic import IsotonicRegression


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 18


def acceptance_rate(mean_accept_prob):
    """
    Convert the mean_accept_prob computed by NUTS kernel
    into a more easily interpretable acceptance rate
    """
    
    acc_rates = []

    for c in mean_accept_prob:
        acc_rate = []
        for n in range(len(c)):
            acc_rate.append((n+1)*c[n] - n*c[n-1])
        acc_rates.append(acc_rate)

    return np.asarray(acc_rates)


def check_convergence(samples, acc_rate, inference_name, plot=False):
    """
    Check convergence with two methods:
    - via trace plots: where the rate of change becomes stable around zero
    - via Gelman-Rubin factor: where it goes below 1.1
    """

    print("Checking convergence...")
    if plot:
        print("Creating trace plots...")

    # Using trace plots
    conv = []

    for s in range(len(samples)):
        for name, param in samples[s].items():
            param = param.squeeze()

            if param.dim()==3:
                for i in range(param.shape[1]):
                    for j in range(param.shape[2]):
                        conv.append(trace_plot(param[:,i,j].cpu(), name + f"_{i}-{j}", plot, inference_name, s))
            elif param.dim()==2:
                for i in range(param.shape[1]):
                    conv.append(trace_plot(param[:,i].cpu(), name + f"_{i}", plot, inference_name, s))
            elif param.dim()==1:
                conv.append(trace_plot(param.cpu(), name, plot, inference_name, s))
            else:
                raise ValueError(f"Parameter {name} of dim {param.dim()}, expected 1, 2 or 3.")
    
        conv.append(trace_plot(acc_rate[s], "acceptance_rate", plot, inference_name, s))
        print(f"Chain {s} completed.")
    
    trace_plot_thr = max(conv)

    # Using Gelman-Rubin
    samples_val = [] # store sample values for each chain
    samples_key = list(samples[0].keys()) # store parameters' names

    for s in samples:
        samples_val.append(list(s.values()))

    r_hats = []
    ess = []
    size = 500
    step = 100
    for p in range(len(samples_val[0])):
        r_hats.append(gelman_rubin(torch.stack([l[p].cpu() for l in samples_val]).unfold(1,size,step),
                                   chain_dim=0, sample_dim=-1))
        # compute also effective sample size
        ess.append(effective_sample_size(torch.stack([l[p].cpu().squeeze() for l in samples_val]),
                                   chain_dim=0, sample_dim=-1))
        
    r_max = []
    for r in r_hats:
        r_tmp = r.squeeze()
        while r_tmp.dim() > 1:
            r_tmp = r_tmp.max(dim=-1)[0].squeeze()
        r_max.append(r_tmp)

    r_max = torch.stack(r_max)

    gelman_rubin_thr = r_hat_plot(r_max.cpu().numpy(), samples_key, plot, inference_name)

    # Return the threshold found with Gelman-Rubin,
    # samples as dictionary after cutting them at gelman_rubin_thr,
    # the last GR factor for each parameter,
    # effective sample size for each parameter
    samples_dict = {samples_key[k] : torch.cat([c[k][gelman_rubin_thr:] for c in samples_val]) for k in range(len(samples_key))}
    GR_dict = {samples_key[k] : r_max[k,-1] for k in range(len(samples_key))}
    ess_dict = {samples_key[k]: ess[k] for k in range(len(samples_key))}

    return trace_plot_thr, gelman_rubin_thr, samples_dict, GR_dict, ess_dict


def trace_plot(variable, name, plot, inference_name, chain_id):
    # Compute a moving average of the rate of change of ´variable´
    r = np.diff(variable)
    av_r = np.convolve(r, np.ones(50)/50, mode='valid')
    x = np.asarray(range(len(av_r)))

    # Change color when ´av_r´ goes below 10%
    cond = np.abs(av_r/(np.max(av_r)-np.min(av_r)))<0.10
    col = np.where(cond, 'r', 'b')

    # Threshold for convergence set where ´av_r´ is consistently under 10%
    t = len(cond) - np.argmin(np.flip(cond))

    # Create the two plots
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, figsize=(15,10))
        ax1.set_title(f"{name} trace plot - chain {chain_id}")
        ax1.grid()
        ax1.plot(variable)
        ax1.vlines(t, ymin=variable.min(), ymax=variable.max(), colors='g', linestyles='dashed', label="Convergence point")
        ax2.set_title("Moving average of the rate of change")
        ax2.grid()
        ax2.scatter(x, av_r, color=col)
        ax2.vlines(t, ymin=av_r.min(), ymax=av_r.max(), colors='g', linestyles='dashed', label="Convergence point")
        
        # Save plots
        save_path = f'./results/plots/{inference_name}/convergence/{chain_id}/'
        Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
        plt.savefig(f'{save_path}{name}.png')
        plt.close(fig)

    return t


def r_hat_plot(r_max, name, plot, inference_name):

    # Find where r_hat becomes consistently smaller than 1.1
    cond = r_max<1.1
    t = cond.shape[1] - max(np.min(np.argmin(np.flip(cond, axis=-1), axis=-1)), 1000)
    t = max(t, 0)

    if plot:
        plt.figure(figsize=(15,5))
        plt.title(f"Gelman-Rubin factor")
        plt.yscale('log')
        plt.ylabel(r"$\hat{R}$")
        plt.xlabel("steps")
        plt.grid()
        plt.plot(r_max.T)
        plt.hlines(y=1, xmin=0, xmax=r_max.shape[1]-1, colors='g', linestyles='dashed', label=r"$\hat{R}=1$")
        plt.vlines(t, ymin=r_max.min(), ymax=r_max.max(), colors='b', linestyles='dashed', label="Convergence point")
        plt.legend(name)

        # Save plots
        save_path = f'./results/plots/{inference_name}/convergence/gelman_rubin/'
        Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
        plt.savefig(f'{save_path}gelman_rubin.png')
        plt.close()
    
    return t


def compute_coverage_len(y_test, y_lower, y_upper):
    """ 
    Compute average coverage and length of prediction intervals
    """
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
    avg_length = np.mean(abs(y_upper - y_lower))
    avg_length = avg_length/(y_test.max()-y_test.min())
    
    return coverage, avg_length


def plot_forecast(predictive, Y, diffXte, diffYte, CI, name, length=200):

    # draw and compute the confidence interval CI
    target_interval = CI[1] - CI[0]
    q_low, q_hi = np.quantile(predictive["obs"].cpu().numpy().squeeze(), CI, axis=0) # 40-quantile
    mean = np.mean(predictive["obs"].cpu().numpy().squeeze(), axis=0)

    fig = plt.figure(figsize=(15,5))
    plt.plot((Y.cpu()+diffYte)[:length], label='true value', color='k')
    plt.fill_between(np.arange(predictive["obs"].shape[1])[:length], (q_low+diffYte)[:length], (q_hi+diffYte)[:length], alpha=0.3, label='0.95 PI')
    plt.plot((mean+diffYte)[:length], label='prediction')
    plt.legend(loc='best')
    plt.grid()

    # Show and save plot
    save_path = f'./results/plots/'
    Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
    plt.savefig(f'{save_path}{name}.png')
    plt.show()
    plt.close(fig)

    pass


def check_calibration(q, Y, quantiles):
    """
    It computes the calibration error according to formula (9)
    of paper https://arxiv.org/pdf/1807.00263.pdf
    """

    # Compute predicted CDF
    predicted_cdf = np.mean(Y.cpu().numpy().squeeze() <= q, axis=1)

    # Compute calibration error
    w = 1
    cal_error = np.sum(w*(predicted_cdf-quantiles)**2)
    
    return cal_error, predicted_cdf


def calibrate(predictive, predictive2, Y, Y2, quantiles, folder, plot=False):
    """
    Function that computes the calibration error on the test dataset,
    train a calibrator on the evaluation dataset and check again the 
    error on test dataset (or vice versa)
    """

    # Check calibration on first dataset
    q = np.quantile(predictive, quantiles, axis=0)
    cal_error, unc_cdf = check_calibration(q, Y, quantiles)

    # Calibrate on second dataset
    # Compute predicted CDF
    q2 = np.quantile(predictive2, quantiles, axis=0)
    predicted_cdf = np.mean(Y2.cpu().numpy().squeeze() <= q2, axis=1)

    # Fit calibrator
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(predicted_cdf, quantiles)

    # Check again calibration on first dataset
    new_quantiles = isotonic.transform(quantiles)
    new_q = np.quantile(predictive, new_quantiles, axis=0)
    new_cal_error, cal_cdf = check_calibration(new_q, Y, quantiles)

    if plot:
        ax = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        # ax.plot(new_quantiles, predicted_cdf, '-s', color='green', label='Cal data')
        ax.plot(quantiles, unc_cdf, '-x', color='purple', label='Uncalibrated')
        ax.plot(quantiles, cal_cdf, '-+', color='red', label='Calibrated')
        ax.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
        ax.set_xlabel('Predicted', fontsize=17)
        ax.set_ylabel('Empirical', fontsize=17)
        ax.set_title('Predicted CDF vs Empirical CDF', fontsize=17)
        ax.legend(fontsize=10)

        save_path = './results/plots/' + folder + '/'
        Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
        plt.savefig(f'{save_path}' + 'calibration' + '.png')
        plt.clf()
    
    return cal_error, new_cal_error, new_quantiles


def num_eval_crps(quantiles, tau, y):
    """
    It computes a discrete version of the CRPS
    so to use it to evaluate quantile regression
    """

    q = np.asarray(quantiles)[:,np.newaxis]
    H = np.heaviside(tau-y, 0)
    dx = np.asarray([tau[i+1,:] - tau[i,:] for i in range(tau.shape[0]-1)])
    crps = np.mean(np.sum(dx*((q-H)[1:,:]**2), 0))

    return crps