import json
import ESN.esnet as esnet

from dataset.data_loaders import load_dataset, generate_datasets


def run_esn(dataset, dim_reduction=True):
    X, Y = load_dataset(dataset)

    # Set ESN hyperparams
    config = json.load(open('ESN/configs/ESN_hyperparams.json', 'r'))

    ### TODO: consider to make this more general with more datasets
    Xtr, Ytr, Xval, Yval, Xte, Yte = generate_datasets(X, Y, test_percent = 0.15, val_percent = 0.15)
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
        return Ytr, train_embedding, val_embedding, test_embedding, Yte
    else:
        # Return the raw reservoir states
        return Ytr, train_states, val_states, Yte, test_states