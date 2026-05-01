import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict
from modules.train_control.parameters import Parameters

class MyDataset(Dataset):
    def __init__(self,  params: Parameters|None=None, inference:bool=False):
        """ The real initlization is called by cls._from_dict """
        self.inference= inference
        self.DATAFILE_PREFIX = params.DATAFILE_PREFIX
        self.DATAFILE_PREFIX_MODEL_INPUT = params.DATAFILE_PREFIX_MODEL_INPUT
        self.DATAFILE_PREFIX_EVAL_INPUT = params.DATAFILE_PREFIX_EVAL_INPUT
        print("All Data Type", self.DATAFILE_PREFIX)
        print("All Model Input", self.DATAFILE_PREFIX_MODEL_INPUT)
        print("All Evaluation Input", self.DATAFILE_PREFIX_EVAL_INPUT)
        
    @classmethod
    def _from_dict(cls, all_inputs:dict, params:Parameters, inference:bool=True):
        """ This triggers __init__ """
        dataset = MyDataset(inference=inference, params=params)
        bs = []
        print(f"Report batch size")
        for key, value in all_inputs.items():
            setattr(dataset, key, value)
            n = value.size(0)
            print(key, n)
            bs.append(n)
            print(f"{key}: {len(value)}")
        # dataset.flatten_X()
        assert len(set(bs)) == 1
        return dataset

    # def flatten_X(self):
    #     self.X = self.X.reshape(self.X_A.size(0), -1)
        
    def __len__(self) -> int:
        if hasattr(self, "X"):
            return len(self.X)
        elif hasattr(self, "X1"):
            return len(self.X1)
        elif hasattr(self, "X_1"):
            return len(self.X_1)
        elif hasattr(self, "XA"):
            return len(self.XA)
        elif hasattr(self, "X_A"):
            return len(self.X_A)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        output = {}
        output['model_input'] = {k: getattr(self, k)[idx] for k in self.DATAFILE_PREFIX_MODEL_INPUT if k not in ['W0', 'Y']}
        if not self.inference:
            output['evaluation_input'] = {k: getattr(self, k)[idx] for k in self.DATAFILE_PREFIX_EVAL_INPUT}
        # print(output['evaluation_input'].keys())
        # quit()
        return output
        

def compile_data_train(dataset:dict|None=None, params: Parameters|None=None) -> Tuple[DataLoader, DataLoader]:
    """     Create DataLoader objects for train, validation, and test sets.
    
    Args:
        X_train, Y_train, W_train: Training data tensors
        X_valid, Y_valid, W_valid: Validation data tensors
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoaders
    
    Returns:
        Tuple of (train_loader, valid_loader) """
    # Create datasets
    data_loader = {}
    for i, target in enumerate(['train', 'valid']):
        data_loader[target] = MyDataset._from_dict({k[:k.rfind('_')]:dataset[k] for k in dataset if k.endswith(target)}, 
                                                      params=params, inference=params.inference)
    # Create dataloaders
        data_loader[target] = DataLoader(
            data_loader[target],
            batch_size=params.BATCH_SIZE,
            shuffle=True,
            num_workers=params.NUM_WORKERS,
            pin_memory=True
        )
    print(f"{target}_loader compiled")
    return data_loader

    
def compile_data_single(target:str='test', dataset:dict|None=None, params: Parameters|None=None) -> DataLoader:
    """     Create DataLoader objects for train, validation, and test sets.
    
    Args:
        target: 'test', 'inference', 'train' if fisher
        dataset: dictionary of individual raw tensors
        params: all Parameters
    
    Returns:
        output_dataset      """
    # Create datasets
    # print(dataset)
    output_dataset = MyDataset._from_dict({k[:k.rfind('_')]:dataset[k] for k in dataset if k.endswith(target)}, 
                                          params=params, inference=params.inference and not params.fisher_info) # 
        
    output_dataset = DataLoader(
        output_dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=False,
        num_workers=params.NUM_WORKERS,
        pin_memory=True
    )
    print(f"Dataloader for {target} compiled")
    return output_dataset
    