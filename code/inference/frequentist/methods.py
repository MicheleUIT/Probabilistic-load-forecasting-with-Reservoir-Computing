import torch
import numpy as np

from time import process_time
from tqdm import trange
from inference.frequentist.utils import compute_coverage_len, check_calibration, eval_crps



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


def pred_QR(model, X, Y, plot, diagnostics, quantiles):

    model.eval()

    # Perform inference
    start_time = process_time()
    predictive = model(X).detach().squeeze()
    inference_time = process_time() - start_time
    diagnostics["inference_time"] = inference_time

    # 0.99 quantiles
    q_low = predictive[:,1].cpu().numpy()
    q_hi = predictive[:,-1].cpu().numpy()
    diagnostics["width99"] = q_hi - q_low
    # and 0.95 quantiles
    q_low = predictive[:,2].cpu().numpy()
    q_hi = predictive[:,-2].cpu().numpy()
    diagnostics["width95"] = q_hi - q_low

    # Check coverage
    # NOTE: is it useful?
    if predictive.dim() > 1:
        _, avg_length = compute_coverage_len(Y.cpu().numpy(), q_low, q_hi)
        diagnostics["avg_length"] = avg_length
    
    # Mean Squared Error
    mean_index = int(predictive.shape[1]/2)
    mean = predictive[:,mean_index].cpu().numpy()
    mse = np.mean((mean-Y.cpu().numpy())**2)
    diagnostics["mse"] = mse

    # Compute calibration error
    diagnostics["cal_error"] = check_calibration(predictive, Y, quantiles, "q_regr", plot=plot)

    # Continuous ranked probability score
    crps = eval_crps(quantiles, predictive.cpu().numpy(), Y.unsqueeze(dim=1).cpu().numpy())
    diagnostics["crps"] = crps

    return predictive, diagnostics


def quantile_loss(quantiles, output, target):
    losses = []
    for i, q in enumerate(quantiles):
        error = target-output[:,i]
        losses.append(torch.max(q*error, (q-1)*error))
    loss = torch.mean(torch.sum(torch.stack(losses), dim=-1))
    return loss