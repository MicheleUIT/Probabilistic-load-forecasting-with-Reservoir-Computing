import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.isotonic import IsotonicRegression



def compute_coverage_len(y_test, y_lower, y_upper):
    """ 
    Compute average coverage and length of prediction intervals
    """
    coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
    avg_length = np.mean(abs(y_upper - y_lower))
    avg_length = avg_length/(y_test.max()-y_test.min())
    
    return coverage, avg_length


def plot_forecast(predictive, Y):
    mean_index = int(predictive.shape[1]/2)
    q_low, mean, q_hi = predictive[:,2].cpu().numpy(), predictive[:,mean_index].cpu().numpy(), predictive[:,-2].cpu().numpy()

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


def check_calibration(predictive, Y, quantiles):
    """
    It computes the calibration error according to formula (9)
    of paper https://arxiv.org/pdf/1807.00263.pdf
    """
    # Compute predicted CDF
    predicted_cdf = np.mean(Y.unsqueeze(dim=1).cpu().numpy() <= predictive.cpu().numpy(), axis=0)

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

    # Check calibration on test dataset
    cal_error, unc_cdf = check_calibration(predictive2, Y2, quantiles)

    # Calibrate on eval dataset
    # Fit calibrator
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(quantiles, predictive.cpu().numpy())

    # Check again calibration on test dataset
    new_quantiles = isotonic.transform(quantiles)
    new_cal_error, cal_cdf = check_calibration(predictive2, Y2, new_quantiles)

    # Plot calibration graph
    if plot:
        ax = plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.plot(quantiles, unc_cdf, '-o', color='purple', label='Uncalibrated')
        ax.plot(quantiles, cal_cdf, '-o', color='purple', label='Calibrated')
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
    
    return cal_error, new_cal_error


def eval_crps(quantiles, tau, y):
    """
    It computes a discrete version of the CRPS
    so to use it to evaluate quantile regression
    """

    crps = np.mean(np.sum((np.asarray(quantiles)-np.heaviside(tau-y, 0))**2, 1))

    return crps