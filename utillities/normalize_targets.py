import torch
def normalize_targets(datasets):
    
    for dataset in datasets.values():

        combined = torch.cat((dataset['tra'].targets, dataset['val'].targets, dataset['tes'].targets))
        combined_mean = combined.mean()
        combined_std = combined.std()

        dataset['tra'].targets = (dataset['tra'].targets - combined_mean) / combined_std
        dataset['val'].targets = (dataset['val'].targets - combined_mean) / combined_std
        dataset['tes'].targets = (dataset['tes'].targets - combined_mean) / combined_std
        
    return datasets