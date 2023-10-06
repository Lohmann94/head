import torch
from data.processed.cross_coupling import datasets
import numpy as np
from utillities.helper import Helper
from models.rbf import RadialBasisFunction
from models.f_cut import CosineCutoff
from models.models import PAINN, PAINN_2
from data.processed.cross_coupling import datasets
from tqdm import tqdm
import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

train_data = datasets.Cross_coupling_optimized_eom2Lig(start_index=0, end_index=2)
val_data = datasets.Cross_coupling_optimized_eom2Lig(start_index=5, end_index=7)
test_data = datasets.Cross_coupling_optimized_eom2Lig(start_index=7, end_index=9)
#data = datasets.Tetris()

# Check if a GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda:0")
else:
    print("GPU is not available.")
    device = torch.device("cpu")

# Network, loss function, and optimizer
#net = GNNInvariant(output_dim=data.num_graphs, state_dim = 5)
net = PAINN_2(num_phys_dims=3, num_message_passing_rounds=5, r_cut=4)
net.to(device)
loss_function = torch.nn.MSELoss()
val_loss_function = torch.nn.MSELoss()
#loss_function = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.01)

epochs = 1000

#TODO spørg: Hvorfor driller modellen når man prøver at smide et andet datasæt i?
for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(train_data)
    loss = loss_function(output, train_data.targets.to(device))
    loss.backward()
    optimizer.step()

    
    #TODO validation loop

    print(f'Epoch: {epoch}, Loss: {loss}, MSE: {loss.item()}')
'''
optimizer.zero_grad()
total_loss = 0
val_output = net(val_data)
val_loss = val_loss_function(val_output, val_data.targets.to(device))
total_loss += val_loss
print(f'Validation loss at epoch {epoch}: {val_loss}')
# Compute and report the average validation loss
average_loss = total_loss // epoch
print(f'Average Validation Loss: {average_loss}')

net.eval()
with torch.no_grad():
    output = net(test_data)
    loss = loss_function(output, test_data.targets.to(device))

'''
