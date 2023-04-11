import json
import ESN.esnet as esnet

from torch import from_numpy
from dataset.data_loaders import load_dataset, generate_datasets


def run_esn(dataset, device, dim_reduction=True):
    """
    Run ESN and returns either the reservoir states, or the embeddings produce by dimensionality reduction
    """

    X, Y = load_dataset(dataset)

    # Set ESN hyperparams
    config = json.load(open('ESN/configs/ESN_hyperparams.json', 'r'))

    Xtr, Ytr, Xval, Yval, Xte, Yte = generate_datasets(X, Y, test_percent = 0.25, val_percent = 0.25)
    print("Tr: {:d}, Val: {:d}, Te: {:d}".format(Xtr.shape[0], Xval.shape[0], Xte.shape[0]))

    # Train and compute predictions
    # Use the ´_states´ variable if you want the embedding to be the identity
    Yte_pred, _, train_states, train_embedding, val_states, val_embedding, test_states, test_embedding = esnet.run_from_config_return_states(Xtr, Ytr, 
                                                                                                                Xte, Yte, 
                                                                                                                config, 
                                                                                                                validation=True,
                                                                                                                Xval=Xval,
                                                                                                                Yval=Yval)

    if dim_reduction==True:
        # Return emedding of states via some dimensionality reduction technique
        return to_torch(Ytr, device).squeeze(), to_torch(train_embedding, device), \
                to_torch(Yval, device).squeeze(), to_torch(val_embedding, device), \
                to_torch(Yte, device).squeeze(), to_torch(test_embedding, device)
    else:
        # Return the raw reservoir states
        return to_torch(Ytr, device).squeeze(), to_torch(train_states, device), \
                to_torch(Yval, device).squeeze(), to_torch(val_states, device), \
                to_torch(Yte, device).squeeze(), to_torch(test_states, device)


def to_torch(array, device):
    """
    Transform numpy arrays to torch tensors and move them to `device`
    """
    
    dtype = 'float32'
    array = array.astype(dtype)
    return from_numpy(array).to(device)