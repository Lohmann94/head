from matplotlib import pyplot as plt
import numpy as np
def single_loss_plotter(epochs, train_losses, val_losses, val_calc_index, plotting, ensemble_model, folder_name):
    plt.clf()
    
    # Creating subplots for train and validation losses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Plotting train loss
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.set_ylabel(f'Model {ensemble_model} Loss')

    # Plotting validation loss
    ax2.plot(np.arange(0, len(epochs), val_calc_index), val_losses, label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(f'Model {ensemble_model} Loss')

    # Setting legends
    ax1.legend()
    ax2.legend()

    # Ensuring proper spacing between subplots
    plt.tight_layout()

    plt.savefig(f'{folder_name}/model_{ensemble_model}_losses.png')

    if plotting:
        plt.show()

def test_loss_plotter(test_losses, plotting, folder_name):

    plt.clf()
    # Create the x-axis values (indices of the test_losses list)
    x = np.arange(len(test_losses))

    # Plot the bar chart
    plt.bar(x, test_losses)

    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Test Loss')
    plt.title('Test Losses for models')

    plt.savefig(f'{folder_name}/test_losses.png')

    if plotting:
        # Show the plot
        plt.show()