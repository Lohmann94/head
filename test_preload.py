import torch
from tqdm import tqdm
from data.processed.cross_coupling.datasets import collate_fn, nemoDataset, normalize_get
from torch.utils.data import DataLoader
from models.models import PAINN_2
from utillities.test_calculations import test_calcs
from utillities.load_env_vars import load_env_vars
from utillities.plotting import test_loss_plotter
from datetime import datetime
import io

small_nets_4 = [
        'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_64_1_final/trained_models/model_0_checkpoint.pth',
        'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_64_2_final/trained_models/model_0_checkpoint.pth',
        'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_64_3_final/trained_models/model_0_checkpoint.pth',
        'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_64_final_4/trained_models/model_0_checkpoint.pth',
        'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_64_1_final/trained_models/model_0_checkpoint.pth'
    ]

    
large_nets_4 = [
    'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_128_1_final/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_128_2_final/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_128_3_final/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_128_4_final/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/sidste_kørsel/696_AG_4.0/painn_128_5_final/trained_models/model_0_checkpoint.pth'
    ]

small_nets_3 = [
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_64_1/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_64_2/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_64_3/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_64_4/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_64_5/trained_models/model_0_checkpoint.pth'
]

large_nets_3 = [
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_128_1/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_128_2/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_128_3/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_128_4/trained_models/model_0_checkpoint.pth',
    'models/experiments/real/cloud/næst_sidste_kørsel/696_AG_3.0/painn_128_5/trained_models/model_0_checkpoint.pth'
]


def test_from_preloaded():
    
    nets = small_nets_3

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
    
    current_datetime = datetime.now()

    folder_time = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    folder_name = f'models/experiments/tests/preload/'

    test_preds = {ensemble: [] for ensemble in range(len(nets))}
    test_targets = {ensemble: [] for ensemble in range(len(nets))}

    train_graphs = int(env_vars['num_graphs'] * env_vars['train_split'])
    val_graphs = int(env_vars['num_graphs'] * env_vars['val_split'])
    test_graphs = int(env_vars['num_graphs'] * env_vars['test_split'])

    if env_vars['batched'] > test_graphs:
        batch_size = 10
    else:
        batch_size = 10

    normalize = normalize_get(indexes=[i for i in range(train_graphs)])

    loss_function = torch.nn.MSELoss()  # Alt L1Loss
    test_losses = []

    net_index = 0

    for network in nets:


        net = torch.load(network)

        test_dataset = nemoDataset(
            length=test_graphs, offset=train_graphs + val_graphs)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn)

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
                    test_preds[net_index].append(pred.item())
                for target in test_batch.targets:
                    test_targets[net_index].append(target.item())

                test_loss = loss_function(
                        test_output, test_batch.targets)
                test_losses.append(test_loss.item())
                tqdm.write(f'Test loss: {test_loss}')

        test_loss_plotter(test_losses, False, folder_name)

        net_index += 1
    test_calcs(test_targets, test_preds, folder_name)

test_from_preloaded()

