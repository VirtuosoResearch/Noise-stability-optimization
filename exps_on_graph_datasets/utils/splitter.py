import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

def cv_random_split(dataset, fold_idx = 0, seed=0):
    """
    :param dataset:
    :param task_idx:
    :param null_value:
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed:
    :param smiles_list: list of smiles corresponding to the dataset obj, or None
    :return: train, valid, test slices of the input dataset obj. If
    smiles_list != None, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [data.y.item() for data in dataset]

    idx_list = []

    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, val_idx = idx_list[fold_idx]

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(val_idx)]

    return train_dataset, valid_dataset