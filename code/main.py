import torch
import wandb
import os

from bayesian.models import BayesianModel

config = {
            "pyro_model": "model1",
            "params": [[0,1],[0,10]]
            }

os.environ["WANDB_MODE"]="offline"
wandb.init(project="bayes_rc", config=config)
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def main():

    n_dims = 20
    in_features = n_dims
    out_features = 1
    torch_model = torch.nn.Sequential(
                    torch.nn.Linear(in_features, 10),
                    torch.nn.Tanh(),
                    torch.nn.Linear(10, out_features)
                ).to(device)

    model = BayesianModel(torch_model, config, device)
    model.render_model(model_args=(torch.rand(1,20), torch.rand(1,1))) # how to print it?


if __name__ == "__main__":
    main()