import torch
from tqdm import tqdm
from data.processed.cross_coupling.datasets import collate_fn, nemoDataset, normalize_get
from torch.utils.data import DataLoader
from models.models import PAINN_2
from utillities.test_calculations import test_calcs
from utillities.load_env_vars import load_env_vars
from datetime import datetime
import pickle
import os
import random



def test_from_preloaded():
    
    network = "models/experiments/real/cloud/final_kørsel/final_painn/painn_64_3/trained_models/model_1_300_91_checkpoint.pth"
    random_number = random.randint(1, 1000000)

    # Set the seed value
    seed_value = 42
    random.seed(seed_value)

    # Check if a GPU is available andet set GPU
    if torch.cuda.is_available():
        print("GPU is available!")
        device = torch.device('cuda')
        torch.cuda.set_device(0)
        print(f"Device being utilized: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("GPU is not available. Using CPU.")
    
    folder_name = f'models/experiments/tests/preload/'

    test_preds = []
    test_targets = []
    '''
    train_graphs = 5642
    val_graphs = 705
    test_graphs = 705
    

    batch_size = 20
    '''

    train_graphs = 10
    val_graphs = 10
    test_graphs = 10
    

    batch_size = 5

    normalize = normalize_get(indexes=[i for i in range(train_graphs)])

    loss_function = torch.nn.MSELoss()  # Alt L1Loss
    test_losses = []

    test_dataset = nemoDataset(
            length=test_graphs, offset=train_graphs + val_graphs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn)

    net = torch.load(network)

    print(f'testing {network} on dataset {test_graphs} with offset {train_graphs + val_graphs}')

    net.eval()
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, desc='Testing Progress'):
            test_batch.targets = (test_batch.targets -
                                    normalize.mean) / normalize.std
            tqdm.write(f'model Calculating Test Loss:')
            test_output = net(test_batch)

            test_pred = test_output.detach()
            for pred in test_pred:
                test_preds.append(pred.item())
            for target in test_batch.targets:
                test_targets.append(target.item())
            

            test_loss = loss_function(
                        test_output, test_batch.targets)
            test_losses.append(test_loss.item())
            tqdm.write(f'Test loss: {test_loss}')

    # Save list1 as a pickle file
    file_path = os.path.join(folder_name, f'test_preds_{random_number}.pkl')
    with open(file_path, 'wb') as f1:
        pickle.dump(test_preds, f1)

    # Save list2 as a pickle file
    file_path = os.path.join(folder_name, f'test_targets_{random_number}.pkl')
    with open(file_path, 'wb') as f2:
        pickle.dump(test_targets, f2)

    # Save list3 as a pickle file
    file_path = os.path.join(folder_name, f'test_losses_{random_number}.pkl')
    with open(file_path, 'wb') as f3:
        pickle.dump(test_losses, f3)
    
test_from_preloaded()

