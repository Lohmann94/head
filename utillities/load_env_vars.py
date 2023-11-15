import os
from dotenv import load_dotenv

import os
from dotenv import load_dotenv

def load_env_vars():
    
    load_dotenv()

    env_vars = {
        'gpu': bool(os.getenv('GPU')),
        'test': bool(os.getenv('TEST')),
        'train_split': float(os.getenv('TRAIN_SPLIT')),
        'val_split': float(os.getenv('VAL_SPLIT')),
        'test_split': float(os.getenv('TEST_SPLIT')),
        'num_graphs': int(os.getenv('NUM_GRAPHS')),
        'ensemble': int(os.getenv('ENSEMBLE')),
        'different_ensembles': bool(os.getenv('DIFFERENT_ENSEMBLES')),
        'num_phys_dims': int(os.getenv('NUM_PHYS_DIMS')),
        'num_message_passing_rounds': int(os.getenv('NUM_MESSAGE_PASSING_ROUNDS')),
        'r_cut': float(os.getenv('R_CUT')),
        'patience': int(os.getenv('PATIENCE')),
        'learning_rate': float(os.getenv('LEARNING_RATE')),
        'weight_decay': float(os.getenv('WEIGHT_DECAY')),
        'epochs': range(int(os.getenv('EPOCHS'))),
        'validation_index': int(os.getenv('VALIDATION_INDEX')),
        'plotting': bool(os.getenv('PLOTTING')),
        'save_interval': float(os.getenv('SAVE_INTERVAL')),
        'target': str(os.getenv('TARGET')),
        'batched': int(os.getenv('BATCHED'))
    }

    return env_vars