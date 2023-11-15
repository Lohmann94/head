import os
import csv
import numpy as np


def test_calcs(test_targets, test_preds, folder_name, plotting=True):

    print('Saving test calculations...')
    if not os.path.exists(f'{folder_name}/test_calculations'):
        os.mkdir(f'{folder_name}/test_calculations')

    with open(f'{folder_name}/test_calculations/targets.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the dictionary to the CSV file
        for key, value in test_targets.items():
            writer.writerow([key] + value)

    with open(f'{folder_name}/test_calculations/predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the dictionary to the CSV file
        for key, value in test_preds.items():
            writer.writerow([key] + value)

    all_test_targets = test_targets[0]

    prediction_target_index = {index: []
                               for index in range(len(test_preds[0]))}

    for i in range(len(test_preds)):
        for k in range(len(test_preds[i])):
            prediction_target_index[k].append(test_preds[i][k])

    ensemble_means = []
    ensemble_stds = []
    ensemble_variances = []

    for target in prediction_target_index:
        # prediction_target_index[target]
        ensemble_means.append(np.mean(prediction_target_index[target]))
        ensemble_stds.append(np.std(prediction_target_index[target]))
        ensemble_variances.append(np.var(prediction_target_index[target]))

    with open(f'{folder_name}/test_calculations/ensemble_stats.txt', 'w') as file:
        file.writelines(
            [f'ensemble_means Line {i+1}: {x}\n' for i, x in enumerate(ensemble_means)])
        file.writelines(
            [f'ensemble_stds Line {i+1}: {x}\n' for i, x in enumerate(ensemble_stds)])
        file.writelines(
            [f'ensemble_variances Line {i+1}: {x}\n' for i, x in enumerate(ensemble_variances)])

    # Plotting the data

    error_means_targets = [abs(pred - mean) for pred, mean in zip(ensemble_means ,all_test_targets)]

    uncertainty_dict = dict(zip(ensemble_variances, error_means_targets))

    sorted_uncertainty_dict = sorted_dict = dict(sorted(uncertainty_dict.items()))
    if plotting:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.scatter(range(len(all_test_targets)), all_test_targets,
                    color='blue', label='Targets')
        plt.scatter(range(len(ensemble_means)), ensemble_means,
                    color='red', label='Mean Predictions')
        #plt.errorbar(range(len(ensemble_means)), ensemble_means, yerr=ensemble_variances,
                     #fmt='none', color='red', capsize=5, label='Variances')
        plt.errorbar(range(len(ensemble_means)), ensemble_means, yerr=ensemble_stds,
                     fmt='none', color='green', capsize=5, label='Standard Deviations')
        # Plot individual test predictions as dots
        for i, model in enumerate(test_preds):
            plt.scatter(
                np.arange(len(test_preds[model])) + 0.3, test_preds[model], color='black')
            if i == len(test_preds) - 1:
                plt.scatter(np.arange(len(
                    test_preds[model])) + 0.3, test_preds[model], color='black', label='Model Predictions')

        # Adding labels and legend
        plt.xlabel('Test Targets')
        plt.ylabel('Predicted Values')
        plt.xticks(range(len(all_test_targets)),
                   list(range(len(all_test_targets))))
        plt.legend()

        plt.savefig(
            f'{folder_name}/test_calculations/test_scatter.png',
        )

        # Display the plot
        plt.show()

        plt.clf()

        variances = list(sorted_uncertainty_dict.keys())
        absolute_errors = list(sorted_uncertainty_dict.values())

        # Define the number of bins
        num_bins = min(int(len(variances) / 2), 10)

        # Calculate the percentile values for the bin boundaries
        percentiles = np.linspace(0, 100, num_bins + 1)
        bin_boundaries = np.percentile(variances, percentiles)

        # Create empty lists to store the predictions for each bin
        binned_predictions = [[] for _ in range(num_bins)]

        # Bin the predictions based on the error values
        for prediction, error in zip(absolute_errors, variances):
            bin_index = np.searchsorted(bin_boundaries, error) - 1
            binned_predictions[bin_index].append(prediction)

        # Plotting the predictions against the bin boundaries as scatter plots
        for i, bin_pred in enumerate(binned_predictions):
            bin_x = np.full(len(bin_pred), percentiles[i])
            plt.scatter(bin_x, bin_pred,alpha=0.5, label=f'Bin {i+1}')

        plt.xlabel('Variance Bins')
        plt.ylabel('Absolute Errors')
        plt.title('Variance vs Absolute Error Bins')
        plt.legend()

        # Display the plot
        plt.show()

    
