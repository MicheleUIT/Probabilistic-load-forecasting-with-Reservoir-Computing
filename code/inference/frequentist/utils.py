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
    q_low, mean, q_hi = predictive[:,0].cpu().numpy(), predictive[:,1].cpu().numpy(), predictive[:,2].cpu().numpy()

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