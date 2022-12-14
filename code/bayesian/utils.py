import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path



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