import torch
import wandb
import os

from bayesian.models import BayesianModel, TorchModel

import pyro
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean

config = {
            "dataset": "acea",
            "model_widths": [20, 10, 1],
            "activation": "tanh",
            "distributions": ["gauss", "unif", "gauss"],
            "parameters": [[0,1],[0,10]],
            "inference": "svi"
            }

os.environ["WANDB_MODE"]="offline"
wandb.init(project="bayes_rc", config=config)
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def main():

    t_m = TorchModel(config.model_widths, config.activation).to(device)

    model = BayesianModel(t_m, config, device)
    model.render_model(model_args=(torch.rand(1,20), torch.rand(1,1)))

    # should I customize the guide?
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
    pyro.render_model(guide, model_args=(torch.rand(1,20), torch.rand(1,1)), render_distributions=True, filename="guide.png")


if __name__ == "__main__":
    main()