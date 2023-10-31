import matplotlib.pyplot as plt
import numpy as np

def ensemble_calcs(train_preds, train_targets, val_preds, val_targets, test_preds, test_targets, folder_name):
    print('Calculating Ensemble Predictions:')

    train_ensemble_preds = {target_index: {epoch: [] for epoch in range(
        len(train_preds[0]))} for target_index in range(len(train_preds[0][0]))}

    val_ensemble_preds = {target_index: {epoch: [] for epoch in list(
        val_preds[0].keys())} for target_index in range(len(val_preds[0][0]))}

    test_ensemble_preds = {target_index: []
                           for target_index in range(len(test_preds[0]))}

    for i in range(len(train_preds[0][0])):
        for epoch in range(len(train_preds[0])):
            for model in range(len(train_preds)):
                train_ensemble_preds[i][epoch].append(
                    train_preds[model][epoch][i])

    for i in range(len(val_preds[0][0])):
        for epoch in list(val_preds[0].keys()):
            for model in range(len(val_preds)):
                val_ensemble_preds[i][epoch].append(
                    val_preds[model][epoch][i])

    for i in range(len(test_preds[0])):
        for model in range(len(test_preds)):
            test_ensemble_preds[i].append(test_preds[model][i])


    train_ensemble_stats = {target_index: {epoch: {'mean': 0, 'std': 0} for epoch in range(
        len(train_preds[0]))} for target_index in range(len(train_preds[0][0]))}

    val_ensemble_stats = {target_index: {epoch: {'mean': 0, 'std': 0} for epoch in list(
        val_preds[0].keys())} for target_index in range(len(val_preds[0][0]))}

    test_ensemble_stats = {target_index: {'mean': 0, 'std': 0}
                           for target_index in range(len(test_preds[0]))}
    
    for target_index in list(train_ensemble_preds.keys()):
        for epoch in list(train_ensemble_preds[target_index].keys()):
            train_ensemble_stats[target_index][epoch]['mean'] = np.mean(
                train_ensemble_preds[target_index][epoch])
            train_ensemble_stats[target_index][epoch]['std'] = np.std(
                train_ensemble_preds[target_index][epoch])

    for target_index in list(val_ensemble_preds.keys()):
        for epoch in list(val_ensemble_preds[target_index].keys()):
            val_ensemble_stats[target_index][epoch]['mean'] = np.mean(
                val_ensemble_preds[target_index][epoch])
            val_ensemble_stats[target_index][epoch]['std'] = np.std(
                val_ensemble_preds[target_index][epoch])
    
    for target_index in list(test_ensemble_preds.keys()):
        test_ensemble_stats[target_index]['mean'] = np.mean(
            test_ensemble_preds[target_index])
        test_ensemble_stats[target_index]['std'] = np.std(
            test_ensemble_preds[target_index])

    print(train_ensemble_stats)
    print(val_ensemble_stats)
    print(test_ensemble_stats)

    
    

            