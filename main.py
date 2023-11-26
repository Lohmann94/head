import torch
import numpy as np
from utillities.helper import Helper
from utillities.plotting import single_loss_plotter, test_loss_plotter
from utillities.normalize_targets import normalize_targets
from utillities.ensemble import ensemble_calcs
from utillities.load_env_vars import load_env_vars
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
import time
from data.processed.cross_coupling.datasets import Cross_coupling_optimized_eom2Lig, GraphBatchWrapper, collate_fn, nemoDataset, normalize_get
from torch.utils.data import DataLoader
from data.processed.cross_coupling.datasets import Tetris, Circles
from utillities.test_calculations import test_calcs
import pickle

start_time = time.time()

env_vars = load_env_vars()

# Check if a GPU is available andet set GPU
if torch.cuda.is_available() and env_vars['gpu'] == True:
    print("GPU is available!")
    device = torch.device('cuda')
    torch.cuda.set_device(0)
    print(f"Device being utilized: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("GPU is not available. Using CPU.")

# Get the current datetime
current_datetime = datetime.now()

best_metric = float('inf')

# Format the datetime as a folder name string
folder_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# Create experiement folder:
if env_vars['test']:

    folder_name = f'models/experiments/tests/{folder_time}'

    train_graphs = 2
    val_graphs = 2
    test_graphs = 20
    batch_size = 1

    print(
        f'Test setup: Train graphs: {train_graphs}, Val graphs: {val_graphs}, Test graphs: {test_graphs}, Batch size: {batch_size}')

else:
    new_folder_name = f'{folder_time}_NG{env_vars["num_graphs"]}_EN{env_vars["ensemble"]}_RC{env_vars["r_cut"]}_LR{env_vars["learning_rate"]}_EP{env_vars["epochs"][-1]+1}'
    folder_name = f'models/experiments/real/{new_folder_name}'

    train_graphs = int(env_vars['num_graphs'] * env_vars['train_split'])
    val_graphs = int(env_vars['num_graphs'] * env_vars['val_split'])
    test_graphs = int(env_vars['num_graphs'] * env_vars['test_split'])

    if env_vars['batched'] > test_graphs:
        batch_size = test_graphs
    else:
        batch_size = env_vars['batched']

    print(
        f'Real setup: Train graphs: {train_graphs}, Val graphs: {val_graphs}, Test graphs: {test_graphs}, Batch size: {batch_size}')

os.mkdir(folder_name)

test_losses = []

seeds = [49, 10, 100, 20, 30, 90, 52, 7, 9, 87]

print(f'Seeds: {seeds}')

save_interval = int(env_vars['epochs'][-1] * env_vars['save_interval'])

test_preds = {ensemble: [] for ensemble in range(env_vars['ensemble'])}

val_targets = {0: None}
test_targets = {ensemble: [] for ensemble in range(env_vars['ensemble'])}

normalize = normalize_get(indexes=[i for i in range(train_graphs)])

# TODO kig RBF og CosineCutoff igennem ift. papers, kør med to forskellige r-cuts for hver funktion for at se hvor det stikker af

all_train_losses = []

for ensemble_model in range(env_vars['ensemble']):

    torch.manual_seed(seeds[ensemble_model])

    train_dataset = nemoDataset(length=train_graphs, offset=0)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn)

    val_dataset = nemoDataset(length=val_graphs, offset=train_graphs)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn=collate_fn)

    test_dataset = nemoDataset(
        length=test_graphs, offset=train_graphs + val_graphs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, collate_fn=collate_fn)

    net = PAINN_2(num_phys_dims=env_vars['num_phys_dims'],
                  num_message_passing_rounds=env_vars['num_message_passing_rounds'],
                  r_cut=env_vars['r_cut'])

    loss_function = torch.nn.MSELoss()  # Alt L1Loss
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=env_vars['learning_rate'],
                                 weight_decay=env_vars['weight_decay'])

    # In order to decay learning rate if validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=env_vars['patience'], verbose=True)

    val_losses = {i: [] for i in range(env_vars['ensemble'])}
    total_val_loss = 0

    train_losses = []

    epochs_without_improvement = 0
    early_stopping_patience = 10
    for epoch in env_vars['epochs']:
        train_loss = 0.0
        print(f'ensemble {ensemble_model} Calculating Training Loss:')
        batch_index = 0
        for batch in tqdm(train_dataloader, desc='Training Progress'):
            batch.targets = (batch.targets - normalize.mean) / normalize.std
            optimizer.zero_grad()
            net.train()
            output = net(batch)
            loss = loss_function(
                output, batch.targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tqdm.write(
                f'Epoch: {epoch}, batch: {batch_index}, Training Loss: {loss}')
            batch_index += 1

        if not env_vars['test']:
            print('Saving model checkpoint...')
            if not os.path.exists(f'{folder_name}/trained_models'):
                os.mkdir(f'{folder_name}/trained_models')
            save_path = f'{folder_name}/trained_models/model_{epoch}_checkpoint.pth'
            torch.save(net, save_path)

        # Calculate average training loss for the epoch
        print(
            f'Total training loss over batches: {train_loss}, epoch {epoch}\n')
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        print(f'Average batch training loss: {train_loss}, epoch {epoch}\n')
        if len(train_losses) >= 5:
            print(f'5-mean training loss: {np.mean(train_losses[-5:])}\n')
        else:
            print(f'mean training loss: {np.mean(train_losses)}\n')

        if not env_vars['test'] and epoch % save_interval == 0:
            print('Saving model checkpoint...')
            if not os.path.exists(f'{folder_name}/trained_models'):
                os.mkdir(f'{folder_name}/trained_models')
            save_path = f'{folder_name}/trained_models/model_{ensemble_model}_checkpoint.pth'
            torch.save(net, save_path)

        if epoch % env_vars['validation_index'] == 0:

            print(
                f'ensemble {ensemble_model} Calculating Validation Loss:')
            net.eval()

            with torch.no_grad():
                val_batch_index = 0
                for val_batch in tqdm(val_dataloader, desc='Validation Progress'):
                    val_batch.targets = (
                        val_batch.targets - normalize.mean) / normalize.std
                    val_output = net(val_batch)
                    val_loss = loss_function(
                        val_output, val_batch.targets)

                    val_losses[ensemble_model].append(val_loss.item())
                    total_val_loss += val_loss

                    if len(val_losses[ensemble_model]) >= 2:
                        exp_smooth_val_loss = val_losses[ensemble_model][-2] * \
                            0.9 + val_losses[ensemble_model][-1] * 0.1
                    else:
                        exp_smooth_val_loss = val_losses[ensemble_model][-1]

                    if exp_smooth_val_loss < best_metric:
                        best_metric = exp_smooth_val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    if epochs_without_improvement >= early_stopping_patience:
                        print("Early stopping triggered. Stopping training.")
                        break
                    
                    scheduler.step(exp_smooth_val_loss)

                    tqdm.write(
                        f'Validation loss at epoch {epoch}: batch {val_batch_index}, {val_loss}')
                    val_batch_index += 1
                    # Compute and report the average validation loss

                    average_loss = total_val_loss / \
                        len(val_losses[ensemble_model])
                    tqdm.write(f'Average Validation Loss: {average_loss}')

                net.train()
            print(f'Saving losses...')
            with open(f'{folder_name}/train_losses_{ensemble_model}.pickle', 'wb') as file:
                pickle.dump(train_losses, file)
            with open(f'{folder_name}/val_losses_{ensemble_model}.pickle', 'wb') as file:
                pickle.dump(val_losses, file)

    all_train_losses.append(train_losses)
    single_loss_plotter(env_vars['epochs'], all_train_losses[ensemble_model], val_losses[ensemble_model],
                        env_vars['validation_index'], False, ensemble_model, folder_name)
    net.eval()
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, desc='Testing Progress'):
            test_batch.targets = (test_batch.targets -
                                  normalize.mean) / normalize.std
            tqdm.write(f'ensemble {ensemble_model} Calculating Test Loss:')
            test_output = net(test_batch)

            test_pred = test_output.detach()
            for pred in test_pred:
                test_preds[ensemble_model].append(pred.item())
            for target in test_batch.targets:
                test_targets[ensemble_model].append(target.item())

            test_loss = loss_function(
                test_output, test_batch.targets)
            test_losses.append(test_loss.item())
            tqdm.write(f'Test loss: {test_loss}')

    if not env_vars['test']:
        print('Saving model...')
        if not os.path.exists(f'{folder_name}/trained_models'):
            os.mkdir(f'{folder_name}/trained_models')
        save_path = f'{folder_name}/trained_models/model_{ensemble_model}.pth'
        torch.save(net, save_path)

test_loss_plotter(test_losses, False, folder_name)
test_calcs(test_targets, test_preds, folder_name)


end_time = time.time()
elapsed_time_seconds = end_time - start_time

elapsed_minutes = int(elapsed_time_seconds // 60)
elapsed_seconds = int(elapsed_time_seconds % 60)

print(f"Elapsed time: {elapsed_minutes} minutes and {elapsed_seconds} seconds")
