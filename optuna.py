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
from hyperopt import fmin, tpe, hp, STATUS_OK

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

# Create experiement folder:
if True:
    train_graphs = 10
    val_graphs = 5
    batch_size = 5

    print(
        f'Test setup: Train graphs: {train_graphs}, Val graphs: {val_graphs} Batch size: {batch_size}')


seeds = [49, 10, 100, 20, 30, 90, 52, 7, 9, 87]

print(f'Seeds: {seeds}')

normalize = normalize_get(indexes=[i for i in range(train_graphs)])

torch.manual_seed(seeds[0])

train_dataset = nemoDataset(length=train_graphs, offset=0)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, collate_fn=collate_fn)

val_dataset = nemoDataset(length=val_graphs, offset=train_graphs)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True, collate_fn=collate_fn)


def objective(params):

    for ensemble_model in range(1):

        r_cut = params['r_cut']

        print(f'r_cut: {r_cut}')

        net = PAINN_2(num_phys_dims=env_vars['num_phys_dims'],
                      num_message_passing_rounds=env_vars['num_message_passing_rounds'],
                      r_cut=r_cut)

        loss_function = torch.nn.MSELoss()  # Alt L1Loss
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=0.004,
                                     weight_decay=env_vars['weight_decay'])

        for epoch in range(20):
            batch_index = 0
            for batch in tqdm(train_dataloader, desc='Training Progress'):
                batch.targets = (
                    batch.targets - normalize.mean) / normalize.std
                optimizer.zero_grad()
                net.train()
                output = net(batch)
                loss = loss_function(
                    output, batch.targets)
                loss.backward()
                optimizer.step()
                tqdm.write(
                    f'Epoch: {epoch}, batch: {batch_index}, Training Loss: {loss}')
                batch_index += 1

        net.eval()
        val_losses = []

        with torch.no_grad():
            val_batch_index = 0
            for val_batch in tqdm(val_dataloader, desc='Validation Progress'):
                val_batch.targets = (
                    val_batch.targets - normalize.mean) / normalize.std
                val_output = net(val_batch)
                val_loss = loss_function(
                    val_output, val_batch.targets)
                val_losses.append(val_loss.item())
        print(f'mean val loss: {np.mean(val_losses)}')
        return {'loss': np.mean(val_losses), 'status': STATUS_OK}


space = {
    'r_cut': hp.uniform('r_cut', 2.0, 5.0),
}  # Run the hyperparameter optimization

best = fmin(objective, space, algo=tpe.suggest, max_evals=20)

print("Best hyperparameters:", best)

# Compute and report the average validation loss
