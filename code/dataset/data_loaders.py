import numpy as np

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


def load_dataset(name):
    if name == "acea":
        return load_acea()
    else:
        raise ValueError(f"{name} dataset not defined.")


def load_acea():
    forecast_horizon = 24 

    mat = loadmat('dataset/TS_Acea.mat')  # load mat-file
    ACEA_data = mat['X'] # original resolution (1 = 10 mins)
    ACEA_data = ACEA_data[::6] # hourly forecast
    ACEA_data = ACEA_data[:7000] 

    X = ACEA_data[:-forecast_horizon]
    Y = ACEA_data[forecast_horizon:]

    return X, Y


def generate_datasets(X, Y, test_percent = 0.15, val_percent = 0.15, scaler = StandardScaler):
    n_data,_ = X.shape

    n_te = np.ceil(test_percent*n_data).astype(int)
    n_val = np.ceil(val_percent*n_data).astype(int)
    n_tr = n_data - n_te - n_val

    # Split dataset
    Xtr = X[:n_tr, :]
    Ytr = Y[:n_tr, :]

    Xval = X[n_tr:-n_te, :]
    Yval = Y[n_tr:-n_te, :]

    Xte = X[-n_te:, :]
    Yte = Y[-n_te:, :]

    # Scale
    Xscaler = scaler()
    Yscaler = scaler()

    # Fit scaler on training set
    Xtr = Xscaler.fit_transform(Xtr)
    Ytr = Yscaler.fit_transform(Ytr)

    # Transform the rest
    Xval = Xscaler.transform(Xval)
    Yval = Yscaler.transform(Yval)

    Xte = Xscaler.transform(Xte)
    Yte = Yscaler.transform(Yte)

    # add constant input
    Xtr = np.concatenate((Xtr,np.ones((Xtr.shape[0],1))),axis=1)
    Xval = np.concatenate((Xval,np.ones((Xval.shape[0],1))),axis=1)
    Xte = np.concatenate((Xte,np.ones((Xte.shape[0],1))),axis=1)

    return Xtr, Ytr, Xval, Yval, Xte, Yte