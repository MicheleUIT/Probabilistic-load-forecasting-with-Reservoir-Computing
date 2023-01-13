import torch
import numpy as np

from time import process_time
from inference.bayesian.models import TorchModel



####################################
#       Quantile regression        #
####################################


def train_QR(model, X, Y, q, lr, epochs):
    model.train()

    torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    start_time = process_time()
    for epoch in range(epochs):
        torch_optimizer.zero_grad()
        for x, y in zip(X, Y):
            loss = quantile_loss(q, model(x), y)
            loss.backward()
            torch_optimizer.step()
        if epoch % 20 == 0:
            print("[iteration %04d] loss: %.4f" % (epoch + 1, loss / Y.shape[0]))
    train_time = process_time() - start_time

    diagnostics = {
        "train_time": train_time,
    }

    return diagnostics


def pred_QR(model, X, Y):
    model.eval()
    return model(X)


def quantile_loss(q, output, target):
    return torch.maximum(q*(target-output), (q-1)*(target-output))