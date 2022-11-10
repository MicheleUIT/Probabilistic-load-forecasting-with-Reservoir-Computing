import json
import ESN.esnet as esnet

from dataset.data_loaders import load_dataset, generate_datasets


def run_esn(dataset):
    X, Y = load_dataset(dataset)

    # Set ESN hyperparams
    config = json.load(open('ESN/configs/ESN_hyperparams.json', 'r'))

    ### TODO: consider to make this more general with more datasets
    Xtr, Ytr, Xval, Yval, Xte, Yte = generate_datasets(X, Y, test_percent = 0.15, val_percent = 0.15)
    print("Tr: {:d}, Val: {:d}, Te: {:d}".format(Xtr.shape[0], Xval.shape[0], Xte.shape[0]))

    # Train and compute predictions
    Yte_pred, _, _, train_embedding, _, val_embedding, _, test_embedding = esnet.run_from_config_return_states(Xtr, Ytr, 
                                                                                                                Xte, Yte, 
                                                                                                                config, 
                                                                                                                validation=True,
                                                                                                                Xval=Xval,
                                                                                                                Yval=Yval)
    return Ytr, train_embedding, val_embedding, test_embedding, Yte