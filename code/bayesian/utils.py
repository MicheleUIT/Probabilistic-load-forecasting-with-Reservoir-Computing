import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path



def check_convergence(samples, acc_rate, plot):
    conv = []

    for name, param in samples.items():
        param = param.squeeze()
        if param.dim()>1:
            for i in range(param.shape[1]):
                conv.append(trace_plot(param[:,i].cpu(), name + f"_{i}", plot))
        else:
            conv.append(trace_plot(param.cpu(), name, plot))
    
    conv.append(trace_plot(acc_rate, "acceptance_rate", plot))
    
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
        ax1.vlines(t, ymin=variable.min(), ymax=variable.max(), colors='g', linestyles='dashed', label="Convergence point")
        ax2.set_title("Moving average of the rate of change")
        ax2.grid()
        ax2.scatter(x, av_r, color=col)
        ax2.vlines(t, ymin=av_r.min(), ymax=av_r.max(), colors='g', linestyles='dashed', label="Convergence point")
        
        # Save plots
        save_path = f'./results/plots/mcmc/convergence/'
        Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
        plt.savefig(f'{save_path}{name}.png')
        plt.close(fig)

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
    save_path = f'./results/plots/mcmc/'
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