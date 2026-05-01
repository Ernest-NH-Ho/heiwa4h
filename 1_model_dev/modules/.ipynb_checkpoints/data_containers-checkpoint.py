from dataclasses import dataclass # , field
# from typing import Optional, Dict, Any, Union
import torch
import pandas as pd

        
@dataclass
class ModelOutput:
    y_pred: torch.Tensor
    intra_model_loss: torch.Tensor|float=0.
    intermediate_output: dict|None = None

# __init__, __repr__, __eq__
class EpochData:
    """Dataclass to encapsulate epoch training data."""
    def __init__(self, loss=None, dropout=None, metrics=None):
        if loss != None:
            self.loss = loss
        # if dropout != None:
        #     self.dropout = dropout
        if metrics != None:
            self.metrics = metrics
    
    def __str__(self):
        cout = ""
        # dropout', '
        for member in ('loss', 'metrics'):
            if hasattr(self, member):
                if member != 'metrics':
                    cout = f"{member} = {getattr(self, member):.4f}\n"
                else:
                    for metric in self.metrics:
                        cout += f"{metric} = {self.metrics[metric]}\n"
        return cout
        