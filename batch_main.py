import torch
import numpy as np
from utillities.helper import Helper
from utillities.plotting import single_loss_plotter, test_loss_plotter
from utillities.normalize_targets import normalize_targets, batch_normalize_targets
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
from data.dataprep import batch_dataprep, batch_test_dataprep, batch_val_dataprep
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import time

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


'''
print('Loading data...')
datasets = dataprep(env_vars['test'], env_vars['train_split'], env_vars['val_split'],
                    env_vars['test_split'], env_vars['num_graphs'], env_vars['ensemble'],
                    env_vars['different_ensembles'], target=env_vars['target'])

print('Normalizing Targets...')

datasets = normalize_targets(datasets)



print(f'Ensemble size: {len(datasets)}')

seeds = [random.sample(range(1, 1000), 1)[0] for i in range(len(datasets))]

seeds[0] = 100

print(f'Seeds: {seeds}')


train_preds = {ensemble: {epoch: None for epoch in env_vars['epochs']}
               for ensemble in range(len(datasets))}
val_preds = {ensemble: {0: None} for ensemble in range(len(datasets))}
test_preds = {ensemble: None for ensemble in range(len(datasets))}

train_targets = {0: None}
val_targets = {0: None}
test_targets = None

'''

# Get the current datetime
current_datetime = datetime.now()

# Format the datetime as a folder name string
folder_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

# Create experiement folder:
if env_vars['test']:
    folder_name = f'models/experiments/tests/{folder_time}'
else:
    new_folder_name = f'{folder_time}_NG{env_vars["num_graphs"]}_EN{env_vars["ensemble"]}_RC{env_vars["r_cut"]}_LR{env_vars["learning_rate"]}_EP{env_vars["epochs"][-1]+1}'
    folder_name = f'models/experiments/real/{new_folder_name}'

os.mkdir(folder_name)

test_losses = []

save_interval = int(env_vars['epochs'][-1] * env_vars['save_interval'])

seeds = [random.sample(range(1, 1000), 1)[0]
         for i in range(env_vars['ensemble'])]

seeds[0] = 100

print(f'Seeds: {seeds}')

tra_num_graphs = int(env_vars['num_graphs'] * env_vars['train_split'])
val_num_graphs = int(env_vars['num_graphs'] * env_vars['val_split'])
test_num_graphs = int(env_vars['num_graphs'] * env_vars['test_split'])

full_train_dataset = batch_dataprep(
    num_graphs=tra_num_graphs,  different_ensembles=env_vars['different_ensembles'], offset=0,
    target=env_vars['target'])

val_dataset = batch_val_dataprep(
    num_graphs=val_num_graphs, different_ensembles=env_vars['different_ensembles'],
    offset=tra_num_graphs,
    target=env_vars['target'])

test_dataset = batch_test_dataprep(
    num_graphs=test_num_graphs,
    different_ensembles=env_vars['different_ensembles'],
    offset=val_num_graphs + tra_num_graphs, target=env_vars['target'])

_, val_dataset, test_dataset, tra_mean, tra_std = batch_normalize_targets(
    full_train_dataset, val_dataset, test_dataset
)

del full_train_dataset


val_losses = {i: [] for i in range(env_vars['ensemble'])}

# TODO kig RBF og CosineCutoff igennem ift. papers, kør med to forskellige r-cuts for hver funktion for at se hvor det stikker af

for ensemble_model in range(env_vars['ensemble']):

    torch.manual_seed(seeds[ensemble_model])

    total_val_loss = 0

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

    if env_vars['batched'] != 0:
        batch_size = env_vars['batched']
        tra_num_graphs = env_vars['num_graphs']

        offsets = [batch_size * i if batch_size * i < tra_num_graphs else tra_num_graphs %
                   batch_size for i in range((tra_num_graphs // batch_size) + 1)]
        batches = [batch_size if (1+i)*batch_size < tra_num_graphs else tra_num_graphs %
                   batch_size for i in range((tra_num_graphs // batch_size)+1)]

        for batch in range(len(batches)):

            datasets = batch_dataprep(num_graphs=batches[batch], offset=offsets[batch],
                                      different_ensembles=env_vars['different_ensembles'],
                                      target=env_vars['target'])
            datasets['tra'].targets = (
                datasets['tra'].targets - tra_mean) / tra_std

            for epoch in env_vars['epochs']:

                print(f'ensemble {ensemble_model} Calculating Training Loss:')
                optimizer.zero_grad()
                output = net(datasets['tra'])

                loss = loss_function(
                    output, datasets['tra'].targets)
                loss.backward()
                optimizer.step()
                print(f'Epoch: {epoch}, Training Loss: {loss}')

                if not env_vars['test'] and epoch % save_interval == 0:
                    print('Saving model checkpoint...')
                    if not os.path.exists(f'{folder_name}/trained_models'):
                        os.mkdir(f'{folder_name}/trained_models')
                    save_path = f'{folder_name}/trained_models/model_{ensemble_model}_checkpoint.pth'
                    torch.save(net.state_dict(), save_path)

                if epoch % env_vars['validation_index'] == 0:

                    print(
                        f'ensemble {ensemble_model} Calculating Validation Loss:')
                    net.eval()

                    with torch.no_grad():

                        val_output = net(val_dataset['val'])
                        val_loss = loss_function(
                            val_output, val_dataset['val'].targets)

                        val_losses[ensemble_model].append(val_loss.item())
                        total_val_loss += val_loss

                        if len(val_losses[ensemble_model]) >= 2:
                            exp_smooth_val_loss = val_losses[ensemble_model][-2] * \
                                0.9 + val_losses[ensemble_model][-1] * 0.1
                        else:
                            exp_smooth_val_loss = val_losses[ensemble_model][-1]

                        scheduler.step(exp_smooth_val_loss)

                        print(f'Validation loss at epoch {epoch}: {val_loss}')
                        # Compute and report the average validation loss

                        average_loss = total_val_loss / \
                            len(val_losses[ensemble_model])
                        print(f'Average Validation Loss: {average_loss}')
                    net.train()

    # single_loss_plotter(env_vars['epochs'], train_losses[ensemble_model], val_losses[ensemble_model],
                    # env_vars['validation_index'], env_vars['plotting'], ensemble_model, folder_name)
    net.eval()
    with torch.no_grad():
        print(f'ensemble {ensemble_model} Calculating Test Loss:')
        test_output = net(test_dataset['tes'])

        test_loss = loss_function(
            test_output, test_dataset['tes'].targets)
        test_losses.append(test_loss.item())
        print(f'Test loss: {test_loss}')

    if not env_vars['test']:
        print('Saving model...')
        if not os.path.exists(f'{folder_name}/trained_models'):
            os.mkdir(f'{folder_name}/trained_models')
        save_path = f'{folder_name}/trained_models/model_{ensemble_model}.pth'
        torch.save(net.state_dict(), save_path)

test_loss_plotter(test_losses, env_vars['plotting'], folder_name)

# ensemble_calcs(train_preds, train_targets, val_preds,
# val_targets, test_preds, test_targets, folder_name)

end_time = time.time()
elapsed_time_seconds = end_time - start_time

elapsed_minutes = int(elapsed_time_seconds // 60)
elapsed_seconds = int(elapsed_time_seconds % 60)

print(f"Elapsed time: {elapsed_minutes} minutes and {elapsed_seconds} seconds")
