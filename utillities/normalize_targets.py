import torch
def normalize_targets(datasets):
    #Normaliser på baggrund af træningssættet
    #TODO standard afvigelse og gennemsnit på ensemble predictions af energier

    for dataset in datasets.values():

        tra_mean = dataset['tra'].targets.mean()
        tra_std = dataset['tra'].targets.std()

        dataset['tra'].targets = (dataset['tra'].targets - tra_mean) / tra_std
        dataset['val'].targets = (dataset['val'].targets - tra_mean) / tra_std
        dataset['tes'].targets = (dataset['tes'].targets - tra_mean) / tra_std
        
    return datasets