import hashlib
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

## random seed
class RandomSeedMix_Generator:
    def __init__(self):
        self.basis = np.array([53533, 82321, 842373, 346397, 3400683, \
                                     # 9357, 38613, 7189057, 40303, 287469, \
                                     # 2921711, 12349, 78901, 11111, 15657, \
                                     704949, 260917, 1125043, 103823, 531441, 33333])
        
    def generate(self, epoch):
        seed_str = "_".join(map(str, list(self.basis+epoch*2)))
        hashed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
        return hashed % (2**63)  # PyTorch expects 63-bit seeds
    
def import_data(train_valid_test=['train', 'valid'], logger=None, params=None):
    """Create dummy data for testing purposes."""
    dataset = {}
    
    for data_name in params.DATAFILE_PREFIX:
        for dset in train_valid_test:
            dataset[f'{data_name}_{dset}'] = torch.load(
                f"{params.DIRECTORIES.DATA_PATH}/{data_name}_{dset}.pt", 
                map_location='cpu', weights_only=True
            ).clone() # [:params.BATCH_SIZE*params.ACCUM_STEP]  # only for trials
    logger.record(f"Tensors loaded for {' and '.join(train_valid_test)}: {dataset.keys()}")
    return dataset


def adjust_weightPN_dim(params, dataset):
    for k in dataset:
        if k.startswith('Y'):
            dim_y = len(dataset[k].size())
            break
    print("util", params.WEIGHT_P)
    if dim_y >= 2:
        for phase in params.WEIGHT_P:
            params.WEIGHT_P[phase] = params.WEIGHT_P[phase].unsqueeze(0)
            params.WEIGHT_N[phase] = params.WEIGHT_N[phase].unsqueeze(0)
    dim_w = len(params.WEIGHT_P[list(params.WEIGHT_P.keys())[0]].size())
    for _ in range(dim_y-dim_w):
        for phase in params.WEIGHT_P:
            params.WEIGHT_P[phase] = params.WEIGHT_P[phase].unsqueeze(-1)
            params.WEIGHT_N[phase] = params.WEIGHT_N[phase].unsqueeze(-1)
    return params
    

def update_dataloader_batch_size(dataloader, new_batch_size):
    """Create a new dataloader with updated batch size."""
    dataset = dataloader.dataset
    return DataLoader(
        dataset,
        batch_size=new_batch_size,
        shuffle=dataloader.sampler is None,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory
    )
