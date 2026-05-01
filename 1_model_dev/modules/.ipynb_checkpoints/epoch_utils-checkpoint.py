from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import torch
import parameters as params

@dataclass
class ModelOutput:
    y_pred: torch.Tensor
    
@dataclass
class EpochData:
    """Dataclass to encapsulate epoch training data."""
    loss: float
    dropout: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize metrics based on metric_keys
        for key in params.metric_keys:
            if key not in self.metrics:
                self.metrics[key] = 0.0 if key in params.metric_keys else 0
    
    def __getattr__(self, name):
        if name in metric_keys:
            return self.metrics.get(name, 0)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name in params.metric_keys:
            if not hasattr(self, 'metrics'):
                super().__setattr__('metrics', {})
            self.metrics[name] = value
        else:
            super().__setattr__(name, value)

    def __str__(self):
        output = f"loss, {self.loss}\n"
        output += f"dropout, {self.dropout}\n"
        for metric in self.metrics:
            output += f"{metric}, {self.metrics[metric]}\n"
        return output