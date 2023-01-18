import torch
import numpy as np

from time import process_time
from inference.frequentist.utils import compute_coverage_len



####################################
#       Quantile regression        #
####################################


def train_QR(model, X, Y, quantiles, lr, epochs):
    model.train()

    torch_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


    start_time = process_time()
    for epoch in range(epochs):
        torch_optimizer.zero_grad()
        loss = quantile_loss(quantiles, model(X), Y.unsqueeze(dim=-1))
        loss.backward()
        torch_optimizer.step()
        if epoch % np.ceil(epochs/10) == 0:
            print("[iteration %04d] loss: %.6f" % (epoch + 1, loss / Y.shape[0]))
    train_time = process_time() - start_time

    diagnostics = {
        "train_time": train_time,
    }

    return diagnostics


def pred_QR(model, X, Y, diagnostics):
    model.eval()

    # Perform inference
    predictive = model(X).detach().squeeze()

    # Check calibration
    if predictive.dim() > 1:
        _, avg_length = compute_coverage_len(Y.cpu().numpy(), predictive[:,0].cpu().numpy(), predictive[:,2].cpu().numpy())
        diagnostics["cal_error"] = avg_length

    return predictive, diagnostics


def quantile_loss(quantiles, output, target):
    losses = []
    for i, q in enumerate(quantiles):
        losses.append(torch.max(q*(target-output[:,i]), (q-1)*(target-output[:,i])))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss