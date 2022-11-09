import torch
import wandb
import os

from bayesian.models import BayesianModel, TorchModel
from bayesian.inference import inference
from dataset.data_loaders import load_dataset

import pyro
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean

config = {
            "dataset": "acea",
            "model_widths": [20, 10, 1],
            "activation": "tanh",
            "distributions": ["gauss", "unif", "gauss"],
            "parameters": [[0,1],[0,10]],
            "inference": "mcmc"
            }

os.environ["WANDB_MODE"]="offline"
wandb.init(project="bayes_rc", config=config)
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def main():

    # Build Torch model
    t_m = TorchModel(config.model_widths, config.activation).to(device)

    # Build Bayesian model
    model = BayesianModel(t_m, config, device)
    model.render_model(model_args=(torch.rand(1,20), torch.rand(1,1)))

    # Build a guide for SVI
    ### TODO: I should customize the guide
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
    pyro.render_model(guide, model_args=(torch.rand(1,20), torch.rand(1,1)), render_distributions=True, filename="guide.png")

    # Load data
    X, Y = load_dataset(config.dataset)

    # Do inference
    inference(config, X, Y)


if __name__ == "__main__":
    main()