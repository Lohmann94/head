import torch
import numpy as np
from utillities.helper import Helper
from utillities.plotting import single_loss_plotter, test_loss_plotter
from utillities.normalize_targets import normalize_targets
from models.rbf import RadialBasisFunction
from models.f_cut import CosineCutoff
from models.models import PAINN, PAINN_2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from data.dataprep import dataprep
from datetime import datetime

load_dotenv()

gpu = bool(os.getenv('GPU'))
test = bool(os.getenv('TEST'))
train_split = float(os.getenv('TRAIN_SPLIT'))
val_split = float(os.getenv('VAL_SPLIT'))
test_split = float(os.getenv('TEST_SPLIT'))
num_graphs = int(os.getenv('NUM_GRAPHS'))
ensemble = int(os.getenv('ENSEMBLE'))

num_phys_dims = int(os.getenv('NUM_PHYS_DIMS'))
num_message_passing_rounds = int(os.getenv('NUM_MESSAGE_PASSING_ROUNDS'))
r_cut = int(os.getenv('R_CUT'))

learning_rate = float(os.getenv('LEARNING_RATE'))
weight_decay = float(os.getenv('WEIGHT_DECAY'))

epochs = range(int(os.getenv('EPOCHS')))
validation_index = int(os.getenv('VALIDATION_INDEX'))
plotting = bool(os.getenv('PLOTTING'))

# Check if a GPU is available
if torch.cuda.is_available() and gpu == True:
    print("GPU is available!")
    torch.cuda.set_device(0)

print('Loading data...')
datasets = dataprep(test, train_split, val_split,
                    test_split, num_graphs, ensemble)

print('Normalizing Targets...')
#TODO check med Mikkel om normalisering over både tra, val og tes er rigtig
datasets = normalize_targets(datasets)

print(f'Ensemble size: {len(datasets)}')

# Get the current datetime
current_datetime = datetime.now()

# Format the datetime as a folder name string
folder_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


if test:
    folder_name = f'models/experiments/tests/{folder_time}'
else:
    new_folder_name = f'NG{num_graphs}_EN{ensemble}_RC{r_cut}_LR{learning_rate}_EP{epochs[-1]+1}_{folder_time}'
    folder_name = f'models/experiments/real/{new_folder_name}'

os.mkdir(folder_name)

test_losses = []

# Network, loss function, and optimizer
#net = GNNInvariant(output_dim=data.num_graphs, state_dim = 5)
# TODO fiks at r_cut fungerer på painn2

for ensemble_model in range(len(datasets)):

    net = PAINN_2(num_phys_dims=num_phys_dims,
                  num_message_passing_rounds=num_message_passing_rounds,
                  r_cut=r_cut)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    total_val_loss = 0

    for epoch in epochs:
        print(f'Calculating Training Loss:')
        optimizer.zero_grad()
        output = net(datasets[ensemble_model]['tra'])
        loss = loss_function(output, datasets[ensemble_model]['tra'].targets)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Training Loss: {loss}')

        # TODO fiks at alle tensors initialiseres til device og ikke flyttes

        if epoch % validation_index == 0:

            print(f'Calculating Validation Loss:')
            net.eval()

            with torch.no_grad():

                val_output = net(datasets[ensemble_model]['val'])
                val_loss = loss_function(
                    val_output, datasets[ensemble_model]['val'].targets)
                val_losses.append(val_loss.item())
                total_val_loss += val_loss
                print(f'Validation loss at epoch {epoch}: {val_loss}')
                # Compute and report the average validation loss
                if epoch != 0:
                    average_loss = total_val_loss / len(val_losses)
                    print(f'Average Validation Loss: {average_loss}')
            net.train()

    single_loss_plotter(epochs, train_losses, val_losses, validation_index, plotting, ensemble_model, folder_name)

    net.eval()
    with torch.no_grad():
        print(f'Calculating Test Loss:')
        test_output = net(datasets[ensemble_model]['tes'])
        test_loss = loss_function(
            test_output, datasets[ensemble_model]['tes'].targets)
        test_losses.append(test_loss.item())
        print(f'Test loss: {test_loss}')

    if not test:
        print('Saving model...')
        if not os.path.exists(f'{folder_name}/trained_models'):
            os.mkdir(f'{folder_name}/trained_models')
        save_path = f'{folder_name}/trained_models/model_{ensemble_model}.pth'
        torch.save(net.state_dict(), save_path)

#TODO Spørg mikkel om ensemble modeller skal have seperate test-sets eller samme for alle modeller
#TODO normalisér targets
test_loss_plotter(test_losses, plotting, folder_name)