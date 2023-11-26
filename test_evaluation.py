import pandas as pd
import numpy as np

# File path of the CSV file
file_path = "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_2/test_calculations/targets.csv"
file_path_2 = "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_1/test_calculations/targets.csv"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)
df_2 = pd.read_csv(file_path_2)

# Print the DataFrame
print(df)