import matplotlib.pyplot as plt
import numpy as np
import json

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

    all_stats = {
        'train_ensemble_stats': train_ensemble_stats,
        'val_ensemble_stats': val_ensemble_stats,
        'test_ensemble_stats': test_ensemble_stats
    }

    data = {
        'train_targets': train_targets,
        'val_targets': val_targets,
        'test_targets': test_targets
    }

    all_stats_file_path = folder_name + '/all_stats.json'
    data_file_path = folder_name + '/data.json'

    json_data_stats = json.dumps(all_stats)
    json_data = json.dumps(data)

    with open(all_stats_file_path, 'w') as stats_file:
        stats_file.write(json_data_stats)

    with open(data_file_path, 'w') as file:
        file.write(json_data)
    

    plt.clf()

    # Extract the number of predictions
    num_predictions = len(test_targets)

    # Initialize arrays for mean predictions and standard deviations
    mean_preds = np.zeros(num_predictions)
    std_preds = np.zeros(num_predictions)

    # Extract the mean and standard deviation values for each prediction
    for i in range(num_predictions):
        mean_preds[i] = test_ensemble_stats[i]['mean']
        std_preds[i] = test_ensemble_stats[i]['std']

    # Create an array of indices
    indices = np.arange(num_predictions)

    # Plot the mean predictions with standard deviations as error bars
    plt.errorbar(indices, mean_preds, yerr=std_preds, fmt='o', label='Mean with Std Dev')

    # Plot the target values as points
    plt.scatter(indices, test_targets, color='red', label='Target Values')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    '''
    For plotting purposes: 

    import matplotlib.pyplot as plt

# Example data
time_points = [1, 2, 3, 4, 5]
target_value = 10
mean_values = [9, 10, 11, 9.5, 10.5]
std_values = [0.5, 0.3, 0.4, 0.2, 0.4]

# Plotting
plt.errorbar(time_points, mean_values, yerr=std_values, fmt='o-', label='Mean ± Std')
plt.plot(time_points, [target_value] * len(time_points), 'r--', label='Target Value')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Target Value vs Mean with Standard Deviation')
plt.legend()
plt.grid(True)
plt.show()
    '''

    
    

            