import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path



def compute_coverage_len(y_test, y_lower, y_upper):
    """ 
    Compute average coverage and length of prediction intervals
    """
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
    avg_length = np.mean(abs(y_upper - y_lower))
    avg_length = avg_length/(y_test.max()-y_test.min())
    
    return coverage, avg_length


def plot_forecast(predictive, Y):
    mean_index = int(predictive.shape[1]/2 - 1)
    q_low, mean, q_hi = predictive[:,0].cpu().numpy(), predictive[:,mean_index].cpu().numpy(), predictive[:,-1].cpu().numpy()

    fig = plt.figure(figsize=(15,5))
    plt.plot(Y.cpu()[:200], label='true value', color='k')
    plt.fill_between(np.arange(mean.shape[0])[:200], q_low[:200], q_hi[:200], alpha=0.3, label='0.95 PI')
    plt.plot(mean[:200], label='prediction')
    plt.legend(loc='best', fontsize=10)
    plt.grid()

    # Show and save plot
    name = "qr"
    save_path = f'./results/plots/'
    Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
    plt.savefig(f'{save_path}{name}.png')
    plt.show()
    plt.close(fig)

    pass


def check_calibration(predictive, Y, quantiles, folder, plot=False):
    """
    It computes the calibration error according to formula (9)
    of paper https://arxiv.org/pdf/1807.00263.pdf, then it plots
    optionally the calibration graph
    """
    # Compute predicted CDF
    predicted_cdf = np.mean(Y.unsqueeze(dim=1).cpu().numpy() <= predictive.cpu().numpy(), axis=0)

    # Compute calibration error
    w = 1 # NOTE: add a weight?
    cal_error = np.sum(w*(predicted_cdf-quantiles)**2)
    
    # Plot calibration graph
    if plot:
        ax = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.scatter(quantiles, predicted_cdf, alpha=0.7, s=3)
        ax.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
        ax.set_xlabel('Predicted', fontsize=17)
        ax.set_ylabel('Empirical', fontsize=17)
        ax.set_title('Predicted CDF vs Empirical CDF', fontsize=17)
        ax.legend(fontsize=17)

        save_path = './results/plots/' + folder + '/'
        Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
        plt.savefig(f'{save_path}' + 'calibration' + '.png')
        plt.show()
        plt.clf()
    
    return cal_error