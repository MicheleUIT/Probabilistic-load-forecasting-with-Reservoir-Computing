import torch
import pmdarima
import numpy as np

from time import process_time
from tqdm import trange
from inference.frequentist.utils import compute_coverage_len, calibrate, eval_crps
from inference.early_stopping import EarlyStopping
from ESN.utils import to_torch



####################################
#       Quantile regression        #
####################################


def train_QR(model, X, Y, X_val, Y_val, lr, epochs, quantiles):

    torch_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # initialize the early_stopping object
    checkpoint_path = "./checkpoints/QR/"
    early_stopping = EarlyStopping(patience=20, verbose=False, path=checkpoint_path)

    start_time = process_time()

    with trange(epochs) as t:
        for epoch in t:
            model.train()
            torch_optimizer.zero_grad()
            loss = quantile_loss(quantiles, model(X), Y)
            loss.backward()
            torch_optimizer.step()

            # display progress bar
            t.set_description(f"Epoch {epoch+1}")
            t.set_postfix({"loss":float(loss / Y.shape[0])})

            # Early stopping
            model.eval()
            valid_loss = quantile_loss(quantiles, model(X_val), Y_val).item()

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            
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
    # calibration doesn't make sense for QR
    cal_error, new_cal_error, new_quantiles = calibrate(predictive, predictive2, Y, Y2, quantiles, folder="q_regr", plot=plot)
    diagnostics["cal_error"] = cal_error
    diagnostics["new_cal_error"] = new_cal_error    # it does not refer to same quantiles
    diagnostics["quantiles"] = quantiles
    diagnostics["new_quantiles"] = new_quantiles

    # Width of 0.95 quantiles
    q_low = predictive[:,2].cpu().numpy()
    q_hi = predictive[:,-2].cpu().numpy()
    diagnostics["width"] = np.mean(q_hi - q_low)
    # After calibration it does not refer to same quantiles
    diagnostics["new_width"] = float('nan')

    # Check coverage with 95% quantiles
    if predictive.dim() > 1:
        coverage, avg_length = compute_coverage_len(Y.cpu().numpy(), q_low, q_hi)
        diagnostics["coverage"] = coverage
        diagnostics["avg_length"] = avg_length
        # After calibration it does not refer to same quantiles
        diagnostics["new_coverage"] = float('nan')
        diagnostics["new_avg_length"] = float('nan')
    
    # Mean Squared Error wrt the median
    mean_index = int(predictive.shape[1]/2)
    median = predictive[:,mean_index].cpu().numpy()
    mse = np.mean((median-Y.cpu().numpy())**2)
    diagnostics["mse"] = mse
    # After calibration it does not refer to the median
    diagnostics["new_mse"] = float('nan')

    # Continuous ranked probability score
    crps = eval_crps(quantiles, predictive.cpu().numpy(), Y.unsqueeze(dim=1).cpu().numpy())
    diagnostics["crps"] = crps
    # After calibration it does not refer to same quantiles
    diagnostics["new_crps"] = float('nan')

    return predictive, diagnostics


def quantile_loss(quantiles, output, target):
    losses = []
    for i, q in enumerate(quantiles):
        error = target-output[:,i]
        losses.append(torch.max(q*error, (q-1)*error))
    loss = torch.mean(torch.sum(torch.stack(losses), dim=-1))
    return loss



#####################################
#               ARIMA               #
#####################################


def train_ARIMA(X, start_p=1, start_q=1, max_p=4, max_q=4, m=1, start_P=0, d=1, D=1):

    start_time = process_time()

    # automatically fit the optimal ARIMA model for given time series
    arima_model = pmdarima.auto_arima(
        y=X,
        start_p=start_p, start_q=start_q,
        max_p=max_p, max_q=max_q, m=m,
        start_P=start_P,
        seasonal=False,         # seasonality already removed in pre-processing
        d=d, D=D,
        trace=False,            # no debug info
        error_action='ignore',  # don't want to know if an order does not work
        suppress_warnings=True, # don't want convergence warnings
        stepwise=True)
    print("Best model selected: ", arima_model)

    train_time = process_time() - start_time

    diagnostics = {
        "train_time": train_time,
        "final_loss": float('nan')  # couldn't find a way to retrieve final loss value
    }

    return arima_model, diagnostics


def pred_ARIMA(model, X_val, Y_val, X_test, Y_test, horizon, plot, diagnostics, quantiles):

    # Make predictions and update the model
    q_lows, q_highs = [], []

    start_time = process_time()

    for q1, q2 in zip(quantiles[1:len(quantiles)//2], quantiles[::-1]):
        pred_list, ci_list = [], []
        with trange(X_val.shape[0] + X_test.shape[0], desc = "CI {0:.2f}".format(q2-q1)) as t:
            for step in t:
                y_hat, conf_int = model.predict(n_periods=horizon, return_conf_int=True, alpha=q2-q1)
                pred_list.append(y_hat[-1])
                ci_list.append(conf_int[-1,:])
        
        # collect predictions, CI, and residuals
        preds = np.hstack(pred_list)
        preds_ci = np.vstack(ci_list)

        q_lows.append(preds_ci[:,0])
        q_highs.append(preds_ci[:,1])

    inference_time = process_time() - start_time
    diagnostics["inference_time"] = inference_time / 2  # only the time spent on val test

    # Combine all quantiles and prediction
    q_lows = np.vstack(q_lows[::-1]).T
    q_highs = np.vstack(q_highs).T

    predictive = np.hstack([q_lows, preds[:,None], q_highs])
    predictive2 = torch.from_numpy(predictive[X_val.shape[0]:,:]) # corresponding to test set
    predictive = torch.from_numpy(predictive[:X_val.shape[0],:])  # corresponding to val test

    # Compute calibration error
    # Calibrate
    cal_error, new_cal_error, new_quantiles = calibrate(predictive, predictive2, Y_val, Y_test, quantiles[1:], folder="arima", plot=plot)
    diagnostics["cal_error"] = cal_error
    diagnostics["new_cal_error"] = new_cal_error    # it does not refer to same quantiles
    diagnostics["quantiles"] = quantiles[1:]
    diagnostics["new_quantiles"] = new_quantiles

    # Width of 0.95 quantiles
    q_low = predictive[:,1].cpu().numpy()
    q_hi = predictive[:,-1].cpu().numpy()
    diagnostics["width"] = np.mean(q_hi - q_low)
    # After calibration it does not refer to same quantiles
    diagnostics["new_width"] = float('nan')

    # Check coverage with 95% quantiles
    if predictive.dim() > 1:
        coverage, avg_length = compute_coverage_len(Y_val.cpu().numpy(), q_low, q_hi)
        diagnostics["coverage"] = coverage
        diagnostics["avg_length"] = avg_length
        # After calibration it does not refer to same quantiles
        diagnostics["new_coverage"] = float('nan')
        diagnostics["new_avg_length"] = float('nan')
    
    # Mean Squared Error wrt the median
    mean_index = int(predictive.shape[1]/2)
    median = predictive[:,mean_index].cpu().numpy()
    mse = np.mean((median-Y_val.cpu().numpy())**2)
    diagnostics["mse"] = mse
    # After calibration it does not refer to the median
    diagnostics["new_mse"] = float('nan')

    # Continuous ranked probability score
    crps = eval_crps(quantiles[1:], predictive.cpu().numpy(), Y_val.unsqueeze(dim=1).cpu().numpy())
    diagnostics["crps"] = crps
    # After calibration it does not refer to same quantiles
    diagnostics["new_crps"] = float('nan')


    return predictive, diagnostics