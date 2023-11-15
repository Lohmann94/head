import torch


def normalize_targets(datasets):
    # Normaliser på baggrund af træningssættet
    for dataset in datasets.values():

        tra_mean = dataset['tra'].targets.mean()
        tra_std = dataset['tra'].targets.std()

        dataset['tra'].targets = (dataset['tra'].targets - tra_mean) / tra_std
        dataset['val'].targets = (dataset['val'].targets - tra_mean) / tra_std
        dataset['tes'].targets = (dataset['tes'].targets - tra_mean) / tra_std

    return datasets


def batch_normalize_targets(tra_dataset, val_dataset, tes_dataset):
    # Normaliser på baggrund af træningssættet

    tra_mean = tra_dataset['tra'].targets.mean()
    tra_std = tra_dataset['tra'].targets.std()

    tra_dataset['tra'].targets = (
        tra_dataset['tra'].targets - tra_mean) / tra_std
    val_dataset['val'].targets = (
        val_dataset['val'].targets - tra_mean) / tra_std
    tes_dataset['tes'].targets = (
        tes_dataset['tes'].targets - tra_mean) / tra_std

    return tra_dataset, val_dataset, tes_dataset, tra_mean, tra_std
