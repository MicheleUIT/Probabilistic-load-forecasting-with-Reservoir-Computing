import torch
import json
import wandb
import os
import ESN.esnet as esnet
import numpy as np

from tqdm import trange
from inference.early_stopping import EarlyStopping
from dataset.data_loaders import load_dataset, generate_datasets
from ESN.utils import to_torch


config = {
            "input_scaling": 0.1,
            "spectral_radius": 0.95
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

# Load dataset
print("Loading dataset")
X, Y = load_dataset("acea")

# Set ESN hyperparams
esn_config = json.load(open('ESN/configs/ESN_hyperparams.json', 'r'))
esn_config['input_scaling'] = config.input_scaling
esn_config['spectral_radius'] = config.spectral_radius

Xtr, Ytr, Xval, Yval, Xte, Yte = generate_datasets(X, Y, test_percent = 0.25, val_percent = 0.25)

print("Running ESN")
Yte_pred, _, train_states, _, val_states, _, test_states, _ = esnet.run_from_config_return_states(Xtr, Ytr, 
                                                                                                Xte, Yte, 
                                                                                                esn_config, 
                                                                                                validation=True,
                                                                                                Xval=Xval,
                                                                                                Yval=Yval)

X_tr = to_torch(train_states, device)
Y_tr = to_torch(Ytr, device).squeeze()
X_val = to_torch(val_states, device)
Y_val = to_torch(Yval, device).squeeze()
X_te = to_torch(test_states, device)
Y_te = to_torch(Yte, device).squeeze()

epochs = 100
MSEloss = torch.nn.MSELoss()

# check point path for early_stopping
checkpoint_path = "./checkpoints/ESN/"

mse = []

for s in range(10):
    torch.manual_seed(s)

    model = torch.nn.Sequential(torch.nn.Linear(512,1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    early_stopping = EarlyStopping(patience=20, verbose=False, path=checkpoint_path)

    with trange(epochs) as t:
        for epoch in t:
            model.train()
            optimizer.zero_grad()
            loss = MSEloss(model(X_tr), Y_tr)
            loss.backward()
            optimizer.step()

            # display progress bar
            t.set_description(f"Epoch {epoch+1}")
            t.set_postfix({"loss":float(loss / Y_tr.shape[0])})

            # Early stopping
            model.eval()
            valid_loss = MSEloss(model(X_val), Y_val).item()

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path + "checkpoint.pt"))

    model.eval()
    pred = model(X_te).detach().cpu().numpy()
    mse.append(np.mean((pred-Y_te.cpu().numpy())**2))

mse = np.asarray(mse)

# check hypothesis about the whitening power of RC
cov = np.cov(test_states.T)
white = np.mean((cov-np.eye(cov.shape[0]))**2)

wandb.log({"m_mse": mse.mean(),
           "s_mse": mse.std(),
           "white": white})