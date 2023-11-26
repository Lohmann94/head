import pickle
import numpy as np
import matplotlib.pyplot as plt

# Specify the file path of the pickle file
small_file_paths = ["models/experiments/real/cloud/final_kørsel/final_painn/painn_64_1/train_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_2/train_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_3/train_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_4/train_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_5/train_losses_0.pickle"]

big_file_paths = ["models/experiments/real/cloud/final_kørsel/final_painn/painn_128_1/train_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_128_2/train_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_128_3/train_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_128_4/train_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_128_5/train_losses_0.pickle"]

big_val_paths = [
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_128_1/val_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_128_2/val_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_128_3/val_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_128_4/val_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_128_5/val_losses_0.pickle"
]

small_val_paths = [
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_1/val_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_2/val_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_3/val_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_4/val_losses_0.pickle",
    "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_5/val_losses_0.pickle"
]

small_training_data = []
big_training_data = []
small_validation_data = []
big_validation_data = []

# Open the pickle file in read mode
for file_path in small_file_paths:
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        data = pickle.load(file)
        small_training_data.append(data)

# Open the pickle file in read mode
for file_path in big_file_paths:
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        data = pickle.load(file)
        big_training_data.append(data)

# Open the pickle file in read mode
for file_path in small_val_paths:
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        data = pickle.load(file)
        small_validation_data.append(data[0])

# Open the pickle file in read mode
for file_path in big_val_paths:
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        data = pickle.load(file)
        big_validation_data.append(data[0])

# Create a figure with two rows and two columns
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# List of colors for each plot
colors = ['red', 'blue', 'green', 'orange', 'purple']
big_models = ['128_1', '128_2', '128_3', '128_4', '128_5']
small_models = ['64_1', '64_2', '64_3', '64_4', '64_5']

# Create a figure with five subplots vertically stacked
fig, axs = plt.subplots(5, figsize=(6, 10))

# Iterate over the subplots and plot the values with individual colors
for i, ax in enumerate(axs):
    ax.plot(small_training_data[i], color=colors[i])
    ax.set_title(f"Training batch log training loss for model {small_models[i]} over batches")
    #ax.set_title(f"Validation batch log validation loss for model {small_models[i]} over batches")
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Batches')  # Set y-axis scale to logarithmic

# Adjust the layout of the subplots
plt.tight_layout()

# Show the plots
plt.show()
pass

# Now you can access the values from the data variable