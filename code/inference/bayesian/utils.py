import torch

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from pyro.ops.stats import gelman_rubin, effective_sample_size



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


def plot_forecast(predictive, Y, name):
    # draw and compute the 95% confidence interval
    target_interval = 0.95
    q_low, q_hi = np.quantile(predictive["obs"].cpu().numpy().squeeze(), [(1-target_interval)/2, 1-(1-target_interval)/2], axis=0) # 40-quantile
    mean = np.mean(predictive["obs"].cpu().numpy().squeeze(), axis=0)

    fig = plt.figure(figsize=(15,5))
    plt.plot(Y.cpu()[:200], label='true value', color='k')
    plt.fill_between(np.arange(predictive["obs"].shape[1])[:200], q_low[:200], q_hi[:200], alpha=0.3, label=str(target_interval)+' PI')
    plt.plot(mean[:200], label='prediction')
    plt.legend(loc='best', fontsize=10)
    plt.grid()

    # Show and save plot
    save_path = f'./results/plots/'
    Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
    plt.savefig(f'{save_path}{name}.png')
    plt.show()
    plt.close(fig)

    pass


def check_calibration(predictive, Y, folder, plot=False):
    """
    It computes the calibration error according to formula (9)
    of paper https://arxiv.org/pdf/1807.00263.pdf, then it plots
    optionally the calibration graph
    """
    # Compute predicted CDF
    predicted_cdf = np.mean(predictive["obs"].cpu().numpy().squeeze() <= Y.cpu().numpy().squeeze(), axis=0)

    # Compute empirical CDF
    empirical_cdf = np.zeros(len(predicted_cdf))
    for i, p in enumerate(predicted_cdf):
        empirical_cdf[i] = np.sum(predicted_cdf <= p)/len(predicted_cdf)

    # Compute calibration error
    w = 1 # NOTE: add a weight?
    cal_error = np.sum(w*(predicted_cdf-empirical_cdf)**2)
    
    # Plot calibration graph
    if plot:
        ax = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.scatter(predicted_cdf, empirical_cdf, alpha=0.7, s=3)
        ax.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
        ax.set_xlabel('Predicted', fontsize=17)
        ax.set_ylabel('Empirical', fontsize=17)
        ax.set_title('Predicted CDF vs Empirical CDF', fontsize=17)
        ax.legend(fontsize=17)

        save_path = './results/plots/' + folder + '/'
        Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
        plt.savefig(f'{save_path}' + 'calibration' + '.png')
        plt.clf()
    
    return cal_error


### TODO: Finish if it is needed
def calibrate(folder):
    """
    Function that computes the calibration error on the test dataset,
    train a calibrator on the evaluation dataset and check again the 
    error on test dataset
    """
    # Check calibration on test dataset
    cal_error = check_calibration()

    # Calibrate on eval dataset

    # Check again calibration on test dataset