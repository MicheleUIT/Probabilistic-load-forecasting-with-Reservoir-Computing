import pyro
import torch
import wandb
import os
import json

import pandas as pd

from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
from pathlib import Path
from inference.bayesian.models import TorchModel, BayesianModel
from inference.inference import inference
from ESN.utils import run_esn


config = {
            "dataset": "acea",
            "model_widths": [50, 10, 1],
            "activation": "tanh",
            "distributions": ["gauss", "unif", "gauss"],
            "parameters": [[0,1],[0,10]],
            "dim_reduction": False,
            "inference": "svi",
            "lr": 0.03,
            "num_iterations": 100,
            "plot": False,
            "seed": 1
            }

# os.environ["WANDB_MODE"]="offline"
wandb.init(project="bayes_rc", config=config)
config = wandb.config

# Select one GPU if more are available
os.environ["CUDA_VISIBLE_DEVICES"]='0'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Save/Load ESN state
n_internal_units = json.load(open('ESN/configs/ESN_hyperparams.json', 'r'))['n_internal_units']
save_path = './ESN/saved/' + f'{config.dataset}/dim_red_{config.dim_reduction}/'
Path(save_path).mkdir(parents=True, exist_ok=True) # create folder if it does not exist
file_path = save_path + f'esn_states_{n_internal_units}units.pt'

if os.path.isfile(file_path):
    Ytr, train_embedding, val_embedding, Yte, test_embedding = torch.load(file_path, map_location=torch.device(device))
else:
    Ytr, train_embedding, val_embedding, Yte, test_embedding = run_esn(config.dataset, device, dim_reduction=config.dim_reduction)
    torch.save([Ytr, train_embedding, val_embedding, Yte, test_embedding], file_path)


times = []
cal_errors = []
crpss = []
losses = []

for s in range(config.seed):
    # Set seed for reproducibility
    pyro.set_rng_seed(s)

    torch_model = TorchModel(config.model_widths, config.activation).to(device)
    svi_model = BayesianModel(torch_model, config, device)

    pyro.clear_param_store()

    # To enforce all the parameters in the guide on the GPU, since we use an autoguide
    if device != 'cpu':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    guide = AutoMultivariateNormal(svi_model, init_loc_fn=init_to_mean)

    # Perform inference
    num_samples = 2000
    predictive, diagnostics = inference(config, svi_model, guide, X_train=train_embedding, Y_train=Ytr, 
                                        X_test=test_embedding, Y_test=Yte, num_samples=num_samples, inference_name=None)


    times.append(diagnostics['train_time'])
    cal_errors.append(diagnostics['cal_error'])
    crpss.append(diagnostics['crps'])
    if "final_loss" in diagnostics.keys():
        losses.append(diagnostics['final_loss'])
    else:
        losses.append(0)


    wandb.log({"seed": s})
    wandb.log({"train_time": diagnostics['train_time']})
    wandb.log({"cal_error": diagnostics['cal_error']})
    wandb.log({"crps": diagnostics['crps']})
    if "final_loss" in diagnostics.keys():
        wandb.log({"final_loss": diagnostics['final_loss']})


df = pd.DataFrame({"seed": range(config.seed), "train_times": times, "cal_errors": cal_errors, "CRPS": crpss, "final_loss": losses})
with pd.ExcelWriter(f"results/results.xlsx", mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name=f"sheet_{config.dataset}_{config.inference}", index=False) 