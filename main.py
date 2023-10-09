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

#Manually set device:
device = torch.device("cpu")

# Network, loss function, and optimizer
#net = GNNInvariant(output_dim=data.num_graphs, state_dim = 5)
#TODO fiks at r_cut fungerer på painn2
#TODO check at loss bliver evalueret rigtigt på targets, i og med at vi flattener output

net = PAINN_2(num_phys_dims=3, num_message_passing_rounds=5, r_cut=4, device=device)
net.to(device)
loss_function = torch.nn.MSELoss()
#loss_function = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

epochs = 100

#TODO spørg: Hvorfor driller modellen når man prøver at smide et andet datasæt i?
for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(train_data)
    loss = loss_function(output, train_data.targets.to(device))
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f'Calculating Validation Loss:')
        net.eval()
        with torch.no_grad():
            total_loss = 0
            val_output = net(val_data)
            val_loss = loss_function(val_output, val_data.targets.to(device))
            total_loss += val_loss
            print(f'Validation loss at epoch {epoch}: {val_loss}')
            # Compute and report the average validation loss
            if epoch != 0:
                average_loss = total_loss // epoch
                print(f'Average Validation Loss: {average_loss}')
        net.train()

    
    #TODO validation loop

    print(f'Epoch: {epoch}, Loss: {loss}, MSE: {loss.item()}')


net.eval()
with torch.no_grad():
    print(f'Calculating Test Loss:')
    test_output = net(test_data)
    test_loss = loss_function(test_output, test_data.targets.to(device))
    print(f'Test loss: {test_loss}')
    
