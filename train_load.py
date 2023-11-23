import pickle

# Specify the path to your pickle file
file_path = "models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_64_3_final/train_losses_0.pickle"

# Open the pickle file in read mode
with open(file_path, 'rb') as file:
    # Load the contents of the pickle file
    data = pickle.load(file)

# Review the contents of the pickle file
print(data)