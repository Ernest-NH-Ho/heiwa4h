from dataclasses import dataclass # , field
# from typing import Optional, Dict, Any, Union
import torch
import pandas as pd

        
@dataclass
class ModelOutput:
    y_pred: torch.Tensor
