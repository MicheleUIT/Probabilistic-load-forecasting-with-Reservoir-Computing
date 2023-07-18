import pyro
import torch
import wandb
import os
import json

import numpy as np
import pandas as pd

from pyro.infer.autoguide import AutoMultivariateNormal, AutoLowRankMultivariateNormal, init_to_mean
from pathlib import Path
from inference.bayesian.models import TorchModel, BayesianModel, HorseshoeSSVS
from inference.inference import inference
from ESN.utils import run_esn


config = {
            "dataset": "acea",
            "model_widths": [512,1],
            "activation": "tanh",
            "distributions": ["gauss", "unif", "gauss"],
            "parameters": [[0,1],[0,10]],
            "dim_reduction": False,
            "dropout_p": 0.2,
            "num_chains": 2,
            "num_samples": 8000,
            "inference": "ssvs",
            "lr": 0.01,
            "num_iterations": 500,
            "low_rank": False,
            "rank": None,
            "plot": False,
            "seed": 5,
            "print_results": False,
            "sweep": True
            }

os.environ["WANDB_MODE"]="online" if config['sweep'] else "offline"
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
    Ytr, train_embedding, Yval, val_embedding, Yte, test_embedding = torch.load(file_path, map_location=torch.device(device))
else:
    Ytr, train_embedding, Yval, val_embedding, Yte, test_embedding, _, _ = run_esn(config.dataset, device, dim_reduction=config.dim_reduction)
    torch.save([Ytr, train_embedding, Yval, val_embedding, Yte, test_embedding], file_path)


# Quantiles
quantiles = [0, 0.005]
for n in range(39):
    quantiles.append(0.025*(n+1))
quantiles.append(0.995)


train_times = []
inf_times = []
cal_errors, new_cal_errors = [], []
e_crpss = []
crpss, new_crpss = [], []
losses = []
widths, new_widths = [], []
mses, new_mses = [], []
coverages, new_coverages = [], []
grs, effs = [], []

for s in range(config.seed):
    # Set seed for reproducibility
    pyro.set_rng_seed(s)

    pyro.clear_param_store()

    torch_model = TorchModel(config.model_widths, config.activation).to(device)
    guide = None
    if config.inference == "ssvs":
        model = HorseshoeSSVS(config.activation, device=device)
    elif config.inference == "q_regr":
        model = TorchModel(config.model_widths, config.activation, quantiles=quantiles).to(device)
    elif config.inference == "dropout":
        model = TorchModel(config.model_widths, config.activation, dropout=True, p=config.dropout_p).to(device)
    else:
        model = BayesianModel(torch_model, config, device)

        # To enforce all the parameters in the guide on the GPU, since we use an autoguide
        if device != 'cpu':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        if not config.low_rank:
            guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
        else:
            guide = AutoLowRankMultivariateNormal(model, init_loc_fn=init_to_mean, rank=config.rank)

    # Perform inference
    predictive, diagnostics = inference(config, model, guide, 
                                        X_train=train_embedding, Y_train=Ytr, 
                                        X_val=val_embedding, Y_val=Yval,
                                        X_test=test_embedding, Y_test=Yte,
                                        quantiles=quantiles)


    train_times.append(diagnostics['train_time'])
    cal_errors.append(diagnostics['cal_error'])
    new_cal_errors.append(diagnostics['new_cal_error'])
    widths.append(diagnostics['width'])
    new_widths.append(diagnostics['new_width'])
    crpss.append(diagnostics['crps'])
    new_crpss.append(diagnostics['new_crps'])
    mses.append(diagnostics['mse'])
    new_mses.append(diagnostics['new_mse'])
    coverages.append(diagnostics['coverage'])
    new_coverages.append(diagnostics['new_coverage'])

    if "final_loss" in diagnostics.keys(): # MCMC doesn't have a loss
        losses.append(diagnostics['final_loss'])
    else:
        losses.append(0)

    if "inference_time" in diagnostics.keys(): # MCMC doesn't have a inference_time
        inf_times.append(diagnostics['inference_time'])
    else:
        inf_times.append(0)

    if "gelman_rubin" in diagnostics.keys(): # only MCMC methods have the gelman rubin factor
        grs.append(diagnostics['gelman_rubin'])
    else:
        grs.append(0)

    if "effective_sample_size" in diagnostics.keys(): # only MCMC methods have the gelman rubin factor
        effs.append(diagnostics['effective_sample_size'])
    else:
        effs.append(0)



m_time = np.asarray(train_times).mean()
s_time = np.asarray(train_times).std()
m_inf_time = np.asarray(inf_times).mean()
s_inf_time = np.asarray(inf_times).std()
m_cal = np.asarray(cal_errors).mean()
s_cal = np.asarray(cal_errors).std()
m_new_cal = np.asarray(new_cal_errors).mean()
s_new_cal = np.asarray(new_cal_errors).std()
m_width = np.asarray(widths).mean()
s_width = np.asarray(widths).std()
m_new_width = np.asarray(new_widths).mean()
s_new_width = np.asarray(new_widths).std()
# m_e_crps = np.asarray(e_crpss).mean()
# s_e_crps = np.asarray(e_crpss).std()
m_crps = np.asarray(crpss).mean()
s_crps = np.asarray(crpss).std()
m_new_crps = np.asarray(new_crpss).mean()
s_new_crps = np.asarray(new_crpss).std()
m_mse = np.asarray(mses).mean()
s_mse = np.asarray(mses).std()
m_new_mse = np.asarray(new_mses).mean()
s_new_mse = np.asarray(new_mses).std()
m_cov = np.asarray(coverages).mean()
s_cov = np.asarray(coverages).std()
m_new_cov = np.asarray(new_coverages).mean()
s_new_cov = np.asarray(new_coverages).std()
m_loss = np.asarray(losses).mean()
s_loss = np.asarray(losses).std()


wandb.log({"m_train_time": m_time,
           "s_train_time": s_time,
           "m_cal_error": m_cal,
           "s_cal_error": s_cal,
           "m_new_cal_error": m_new_cal,
           "s_new_cal_error": s_new_cal,
        #    "m_e_crps": m_e_crps,
        #    "s_e_crps": s_e_crps,
           "m_crps": m_crps,
           "s_crps": s_crps,
           "m_new_crps": m_new_crps,
           "s_new_crps": s_new_crps,
           "m_mse": m_mse,
           "s_mse": s_mse,
           "m_new_mse": m_new_mse,
           "s_new_mse": s_new_mse,
           "m_width": m_width,
           "s_width": s_width,
           "m_new_width": m_new_width,
           "s_new_width": s_new_width,
           "m_cov": m_cov,
           "s_cov": s_cov,
           "m_new_cov": m_new_cov,
           "s_new_cov": s_new_cov,
           "m_final_loss": m_loss,
           "s_final_loss": s_loss,
           "gel_rub": grs,
           "eff_size": effs})


if config.print_results:
    df = pd.DataFrame({"seed": range(config.seed), "train_times": train_times, "inf_times": inf_times,
                       "cal_errors": cal_errors, "new_cal_errors": new_cal_errors,
                       "coverage": coverages, "new_coverage": new_coverages,
                       "width": widths, "new_width": new_widths,
                       "MSE": mses, "new_MSE": new_mses,
                    #    "e_CRPS": e_crpss,
                       "CRPS": crpss, "new_CRPS": new_crpss,
                       "final_loss": losses})
    with pd.ExcelWriter(f"results/results.xlsx", mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=f"sheet_{config.dataset}_{config.inference}", index=False) 