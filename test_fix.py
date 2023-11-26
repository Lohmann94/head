import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


painn_64_test_targets = ["models/experiments/real/cloud/test_results/preload/test_targets_81607.pkl",
                         "models/experiments/real/cloud/test_results/preload/test_targets_90492.pkl",
                         "models/experiments/real/cloud/test_results/preload/test_targets_325049.pkl",
                         "models/experiments/real/cloud/test_results/preload/test_targets_370938.pkl",
                         "models/experiments/real/cloud/test_results/preload/test_targets_815727.pkl"
                         ]

painn_64_test_preds = ["models/experiments/real/cloud/test_results/preload/test_preds_81607.pkl",
                       "models/experiments/real/cloud/test_results/preload/test_preds_90492.pkl",
                       "models/experiments/real/cloud/test_results/preload/test_preds_325049.pkl",
                       "models/experiments/real/cloud/test_results/preload/test_preds_370938.pkl",
                       "models/experiments/real/cloud/test_results/preload/test_preds_815727.pkl"]

painn_64_test_losses = ["models/experiments/real/cloud/test_results/preload/test_losses_81607.pkl",
                        "models/experiments/real/cloud/test_results/preload/test_losses_90492.pkl",
                        "models/experiments/real/cloud/test_results/preload/test_losses_325049.pkl",
                        "models/experiments/real/cloud/test_results/preload/test_losses_370938.pkl",
                        "models/experiments/real/cloud/test_results/preload/test_losses_815727.pkl"]

painn_128_test_targets = ["models/experiments/real/cloud/test_results/preload/test_targets_109232.pkl",
                          "models/experiments/real/cloud/test_results/preload/test_targets_624224.pkl",
                          "models/experiments/real/cloud/test_results/preload/test_targets_706647.pkl",
                          "models/experiments/real/cloud/test_results/preload/test_targets_759338.pkl",
                          "models/experiments/real/cloud/test_results/preload/test_targets_788938.pkl"
                          ]

painn_128_test_preds = ["models/experiments/real/cloud/test_results/preload/test_preds_109232.pkl",
                        "models/experiments/real/cloud/test_results/preload/test_preds_624224.pkl",
                        "models/experiments/real/cloud/test_results/preload/test_preds_706647.pkl",
                        "models/experiments/real/cloud/test_results/preload/test_preds_759338.pkl",
                        "models/experiments/real/cloud/test_results/preload/test_preds_788938.pkl"]

painn_128_test_losses = ["models/experiments/real/cloud/test_results/preload/test_losses_109232.pkl",
                         "models/experiments/real/cloud/test_results/preload/test_losses_624224.pkl",
                         "models/experiments/real/cloud/test_results/preload/test_losses_706647.pkl",
                         "models/experiments/real/cloud/test_results/preload/test_losses_759338.pkl",
                         "models/experiments/real/cloud/test_results/preload/test_losses_788938.pkl"]


def loss_plotter():
    loss_values = []

    for file_path in painn_128_test_losses:
        with open(file_path, "rb") as file:
            # Load the data from the pickle file
            data = pickle.load(file)
            loss_values.append(sum(data))

    # Colors for each bar
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    mean_value = np.mean(loss_values)
    print(mean_value)

    x = [0, 1, 2, 3, 4]

    # Create a bar plot
    sns.barplot(x=x, y=loss_values)
    plt.axhline(mean_value, color='red', linestyle='--', linewidth=2)

    # Customize the plot
    plt.xlabel('Model')
    plt.ylabel('Total MSE Loss')
    plt.title(
        'Total test Loss per Model in MPNN128 Ensemble (Dash: Mean Loss of models)')
    plt.xticks(x, ['MPNN128_1', 'MPNN128_2',
               'MPNN128_3', 'MPNN128_4', 'MPNN128_5'])

    # Show the plot
    plt.show()
    pass


def ensemble_preds_plotter():

    target_path = painn_128_test_targets[0]
    current_list = painn_128_test_preds

    targets = [] 

    with open(target_path, "rb") as file:
            # Load the data from the pickle file
            data = pickle.load(file)
            for point in data:
                targets.append(point)

    sum_preds = [0 for i in range(705)]

    for file_path in current_list:
        with open(file_path, "rb") as file:
            # Load the data from the pickle file
            data = pickle.load(file)
            for index, point in enumerate(data):
                sum_preds[index] += point

    mean_preds = [i / 5 for i in sum_preds]
    mse = [mean_squared_error(targets, mean_preds)]

    for file_path in current_list:
        with open(file_path, "rb") as file:
            # Load the data from the pickle file
            data = pickle.load(file)
            mse.append(mean_squared_error(targets, data))
    
    print(mse)


    # Bar colors
    colors = ['red', 'blue', 'blue', 'blue', 'blue', 'blue']

    # Plot the bar chart
    plt.bar(np.arange(len(mse)), mse, color=colors)

    # Set the different color for the first bar
    plt.bar(0, mse[0], color='green')

    # Set labels and title
    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error for MPNN64 Ensemble (green) and individual models (blue)')
    # Set x-axis limits
    plt.ylim(0.8, 1.3)

    # Show the plot
    plt.show()



    variance_preds = [0 for i in range(705)]
    error_list = [0 for i in range(705)]

    for file_path in current_list:
        with open(file_path, "rb") as file:
            # Load the data from the pickle file
            data = pickle.load(file)
            for index, point in enumerate(data):
                variance_preds[index] += (mean_preds[index] - point)**2
                error_list[index] += (point - targets[index])
    
    variances = np.sort(variance_preds)
    errors = np.sort(error_list)

    # Calculate the bin size
    bin_size = len(variances) // 9

    # Split the variances into equal-sized bins
    variance_bins = [variances[i:i+bin_size] for i in range(0, len(variances), bin_size)]
    error_bins = [errors[i:i+bin_size] for i in range(0, len(errors), bin_size)]

    # Print the bins
    for i, bin in enumerate(variance_bins):
        print(f"Bin {i+1}: {bin}")
    
    for k, bin in enumerate(error_bins):
        print(f"Bin {k+1}: {bin}")

    # Calculate the root mean variance of each bin
    rmv_bins = [np.sqrt(np.mean(bin)) for bin in variance_bins]
    rmse_bins = [np.sqrt(np.mean(bin)**2) for bin in error_bins]

    ence_holder = 0
    for index, variance in enumerate(rmv_bins):
        ence_holder += abs(variance - rmse_bins[index]) / variance
    ence = ence_holder / len(rmv_bins)

    # Print the root mean variance of each bin
    for i, rmv in enumerate(rmv_bins):
        print(f"Bin {i+1}: RMV = {rmv}")

    for k, rmse in enumerate(rmse_bins):
        print(f"Bin {k+1}: RMSE = {rmse}")

    # Plot the scatter points
    x=rmse_bins
    y=rmv_bins
    plt.scatter(x, y)

    # Plot the line going through the scatter points
    plt.plot(x, y, color='blue')

    # Set labels and title
    plt.xlabel('RMSE bins')
    plt.ylabel('RMV bins')
    plt.title('Scatter plot of RMSE and RMV bins for ensemble of MPNN64 models')

    # Show the plot
    plt.show()
    

    
    pass


ensemble_preds_plotter()
