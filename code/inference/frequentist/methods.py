import torch
import numpy as np

from time import process_time
from tqdm import trange
from inference.frequentist.utils import compute_coverage_len, calibrate, eval_crps



####################################
#       Quantile regression        #
####################################


def train_QR(model, X, Y, lr, epochs, quantiles):

    model.train()

    torch_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    start_time = process_time()

    with trange(epochs) as t:
        for epoch in t:
            torch_optimizer.zero_grad()
            loss = quantile_loss(quantiles, model(X), Y)
            loss.backward()
            torch_optimizer.step()

            # display progress bar
            t.set_description(f"Epoch {epoch+1}")
            t.set_postfix({"loss":float(loss / Y.shape[0])})

    train_time = process_time() - start_time

    diagnostics = {
        "train_time": train_time,
        "final_loss": float((loss / Y.shape[0]).detach().cpu())
    }

    return diagnostics


def pred_QR(model, X_val, Y_val, X_test, Y_test, plot, sweep, diagnostics, quantiles):

    model.eval()

    # Use validation set for hyperparameters tuning
    if sweep:
        X, Y = X_val, Y_val
        X2, Y2 = X_test, Y_test
    else:
        X, Y = X_test, Y_test
        X2, Y2 = X_val, Y_val

    # Perform inference
    start_time = process_time()
    predictive = model(X).detach().squeeze()
    inference_time = process_time() - start_time
    diagnostics["inference_time"] = inference_time

    # Compute calibration error
    predictive2 = model(X2).detach().squeeze()
    # Calibrate
    cal_error, new_cal_error, new_quantiles = calibrate(predictive, predictive2, Y, Y2, quantiles, folder="q_regr", plot=plot)
    diagnostics["cal_error"] = cal_error
    diagnostics["new_cal_error"] = new_cal_error

    # 0.95 quantiles
    q_low = predictive[:,2].cpu().numpy()
    q_hi = predictive[:,-2].cpu().numpy()
    diagnostics["width"] = np.mean(q_hi - q_low)
    # After calibration
    new_q_low = predictive2[:,2].cpu().numpy()
    new_q_hi = predictive2[:,-2].cpu().numpy()
    diagnostics["new_width"] = np.mean(new_q_hi - new_q_low)

    # Check coverage with 95% quantiles
    if predictive.dim() > 1:
        coverage, avg_length = compute_coverage_len(Y.cpu().numpy(), q_low, q_hi)
        diagnostics["coverage"] = coverage
        diagnostics["avg_length"] = avg_length
        # Re-compute after calibration
        coverage, avg_length = compute_coverage_len(Y.cpu().numpy(), new_q_low, new_q_hi)
        diagnostics["new_coverage"] = coverage
        diagnostics["new_avg_length"] = avg_length
    
    # Mean Squared Error
    mean_index = int(predictive.shape[1]/2)
    mean = predictive[:,mean_index].cpu().numpy()
    mse = np.mean((mean-Y.cpu().numpy())**2)
    diagnostics["mse"] = mse
    # After calibration
    mean_index = int(predictive2.shape[1]/2)
    mean = predictive2[:,mean_index].cpu().numpy()
    mse = np.mean((mean-Y.cpu().numpy())**2)
    diagnostics["new_mse"] = mse

    # Continuous ranked probability score
    crps = eval_crps(quantiles, predictive.cpu().numpy(), Y.unsqueeze(dim=1).cpu().numpy())
    diagnostics["crps"] = crps
    # Compute CRPS after calibration
    new_crps = eval_crps(new_quantiles, predictive.cpu().numpy(), Y.unsqueeze(dim=1).cpu().numpy())
    diagnostics["new_crps"] = new_crps

    return predictive, diagnostics


def quantile_loss(quantiles, output, target):
    losses = []
    for i, q in enumerate(quantiles):
        error = target-output[:,i]
        losses.append(torch.max(q*error, (q-1)*error))
    loss = torch.mean(torch.sum(torch.stack(losses), dim=-1))
    return loss