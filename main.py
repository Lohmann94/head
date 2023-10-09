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
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

train_data = datasets.Cross_coupling_optimized_eom2Lig(start_index=tr_start_index, end_index=tr_end_index)
val_data = datasets.Cross_coupling_optimized_eom2Lig(start_index=v_start_index, end_index=v_end_index)
test_data = datasets.Cross_coupling_optimized_eom2Lig(start_index=te_start_index, end_index=te_end_index)
#data = datasets.Tetris()
gpu = os.getenv('GPU')
# Check if a GPU is available
if torch.cuda.is_available() and gpu == 'True':
    print("GPU is available!")
    torch.cuda.set_device(0)


# Network, loss function, and optimizer
#net = GNNInvariant(output_dim=data.num_graphs, state_dim = 5)
#TODO fiks at r_cut fungerer på painn2
#TODO check at loss bliver evalueret rigtigt på targets, i og med at vi flattener output
#TODO Spørg Mikkel hvad modellen skal defaulte til, hvis der ikke er nogen naboer indenfor skæringsgrænsen

net = PAINN_2(num_phys_dims=3, num_message_passing_rounds=5, r_cut=4)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

epochs = range(100)
val_calc_index = 5
val_epoch = 0
total_loss = 0

train_losses = []
val_losses = []
test_losses = []

#TODO spørg: Hvorfor driller modellen når man prøver at smide et andet datasæt i?
for epoch in epochs:
    print(f'Calculating Training Loss:')
    optimizer.zero_grad()
    output = net(train_data)
    loss = loss_function(output, train_data.targets)
    train_losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}, Training Loss: {loss}')

    #TODO fiks at alle tensors initialiseres til device og ikke flyttes

    if epoch % val_calc_index == 0:
        print(f'Calculating Validation Loss:')
        val_epoch += 1
        net.eval()
        with torch.no_grad():
            val_output = net(val_data)
            val_loss = loss_function(val_output, val_data.targets)
            val_losses.append(val_loss.item())
            total_loss += val_loss
            print(f'Validation loss at epoch {epoch}: {val_loss}')
            # Compute and report the average validation loss
            if epoch != 0:
                average_loss = total_loss / val_epoch
                print(f'Average Validation Loss: {average_loss}')
        net.train()


net.eval()
with torch.no_grad():
    print(f'Calculating Test Loss:')
    test_output = net(test_data)
    test_loss = loss_function(test_output, test_data.targets)
    test_losses.append(test_loss.item())
    print(f'Test loss: {test_loss}')

# Creating subplots for train and validation losses
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(epochs, train_losses, label='Train Loss')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(np.arange(0, len(epochs), val_calc_index), val_losses, label='Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()  # Ensure proper spacing between subplots
plt.show()


    
