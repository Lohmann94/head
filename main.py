import torch
import numpy as np
from utillities.helper import Helper
from utillities.plotting import single_loss_plotter, test_loss_plotter
from utillities.normalize_targets import normalize_targets
from utillities.ensemble import ensemble_calcs
from models.rbf import RadialBasisFunction
from models.f_cut import CosineCutoff
from models.models import PAINN, PAINN_2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from data.dataprep import dataprep
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

load_dotenv()

gpu = bool(os.getenv('GPU'))

test = bool(os.getenv('TEST'))

train_split = float(os.getenv('TRAIN_SPLIT'))
val_split = float(os.getenv('VAL_SPLIT'))
test_split = float(os.getenv('TEST_SPLIT'))
num_graphs = int(os.getenv('NUM_GRAPHS'))

ensemble = int(os.getenv('ENSEMBLE'))
different_ensembles = bool(os.getenv('DIFFERENT_ENSEMBLES'))

num_phys_dims = int(os.getenv('NUM_PHYS_DIMS'))
num_message_passing_rounds = int(os.getenv('NUM_MESSAGE_PASSING_ROUNDS'))
r_cut = int(os.getenv('R_CUT'))
patience = int(os.getenv('PATIENCE'))

learning_rate = float(os.getenv('LEARNING_RATE'))
weight_decay = float(os.getenv('WEIGHT_DECAY'))

epochs = range(int(os.getenv('EPOCHS')))
validation_index = int(os.getenv('VALIDATION_INDEX'))
plotting = bool(os.getenv('PLOTTING'))

# Check if a GPU is available
if torch.cuda.is_available() and gpu == True:
    print("GPU is available!")
    device = torch.device('cuda')
    torch.cuda.set_device(0)
    print(f"Device being utilized: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("GPU is not available. Using CPU.")



print('Loading data...')
datasets = dataprep(test, train_split, val_split,
                    test_split, num_graphs, ensemble,
                    different_ensembles)

print('Normalizing Targets...')

datasets = normalize_targets(datasets)

print(f'Ensemble size: {len(datasets)}')

# Get the current datetime
current_datetime = datetime.now()

# Format the datetime as a folder name string
folder_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


if test:
    folder_name = f'models/experiments/tests/{folder_time}'
else:
    new_folder_name = f'{folder_time}_NG{num_graphs}_EN{ensemble}_RC{r_cut}_LR{learning_rate}_EP{epochs[-1]+1}'
    folder_name = f'models/experiments/real/{new_folder_name}'

os.mkdir(folder_name)

test_losses = []

seeds = [random.sample(range(1, 1000), 1)[0] for i in range(len(datasets))]

train_preds = {ensemble: {epoch: None for epoch in epochs} for ensemble in range(len(datasets))}
val_preds = {ensemble: {0: None} for ensemble in range(len(datasets))}
test_preds = {ensemble: None for ensemble in range(len(datasets))}

train_targets = {0: None}
val_targets = {0: None}
test_targets = None

for ensemble_model in range(len(datasets)):

    torch.manual_seed(seeds[ensemble_model])

    net = PAINN_2(num_phys_dims=num_phys_dims,
                  num_message_passing_rounds=num_message_passing_rounds,
                  r_cut=r_cut)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    # In order to decay learning rate if validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, verbose=True)

    train_losses = {i: [] for i in range(len(datasets))} 
    val_losses = {i: [] for i in range(len(datasets))} 
    total_val_loss = 0

    train_target = datasets[ensemble_model]['tra'].targets.detach()
    train_targets[ensemble_model] = [value.item() for value in train_target]

    val_target = datasets[ensemble_model]['val'].targets.detach()
    val_targets[ensemble_model] = [value.item() for value in val_target]

    test_target = datasets[ensemble_model]['tes'].targets.detach()
    test_targets = [value.item() for value in test_target]

    for epoch in epochs:
        print(f'ensemble {ensemble_model} Calculating Training Loss:')
        optimizer.zero_grad()
        output = net(datasets[ensemble_model]['tra'])

        train_pred = output.detach()
        train_preds[ensemble_model][epoch] = [value.item() for value in train_pred]

        loss = loss_function(output, datasets[ensemble_model]['tra'].targets)
        train_losses[ensemble_model].append(loss.item())
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Training Loss: {loss}')

        if epoch % validation_index == 0:

            print(f'ensemble {ensemble_model} Calculating Validation Loss:')
            net.eval()

            with torch.no_grad():

                val_output = net(datasets[ensemble_model]['val'])
                val_loss = loss_function(
                    val_output, datasets[ensemble_model]['val'].targets)

                val_pred = val_output.detach()
                val_preds[ensemble_model][epoch] = [value.item() for value in val_pred]

                val_losses[ensemble_model].append(val_loss.item())
                total_val_loss += val_loss

                if len(val_losses[ensemble_model]) >= 2:
                    exp_smooth_val_loss = val_losses[ensemble_model][-2] * 0.9 + val_losses[ensemble_model][-1] * 0.1 
                else:
                    exp_smooth_val_loss = val_losses[ensemble_model][-1]

                scheduler.step(exp_smooth_val_loss)

                print(f'Validation loss at epoch {epoch}: {val_loss}')
                # Compute and report the average validation loss

                average_loss = total_val_loss / len(val_losses[ensemble_model])
                print(f'Average Validation Loss: {average_loss}')
            net.train()

    single_loss_plotter(epochs, train_losses[ensemble_model], val_losses[ensemble_model],
                        validation_index, plotting, ensemble_model, folder_name)

    net.eval()
    with torch.no_grad():
        print(f'ensemble {ensemble_model} Calculating Test Loss:')
        test_output = net(datasets[ensemble_model]['tes'])

        test_pred = test_output.detach()
        test_preds[ensemble_model] = [value.item() for value in test_pred]

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

test_loss_plotter(test_losses, plotting, folder_name)

ensemble_calcs(train_preds, train_targets, val_preds, val_targets, test_preds, test_targets, folder_name)
