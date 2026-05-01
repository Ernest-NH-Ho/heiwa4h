# modules/function_loss.py
import torch
import torch.nn as nn
from .data_containers import EpochData
import math

class Binary_Cross_Entropy:
    epsilon = 1e-6
    
    @staticmethod
    def calculate(y_pred: torch.Tensor, y_true: torch.Tensor, 
                 weight_0: torch.Tensor, weight_p: float, weight_n: float) -> torch.Tensor:
        """
        Calculate binary cross entropy loss with sample weights and class weights.
        
        Args:
            y_pred: Predicted probabilities
            y_true: True binary labels
            weight_0: Sample weights from W_tensor
            weight_p: Weight for positive class (y=1)
            weight_n: Weight for negative class (y=0)
        
        Returns:
            Weighted binary cross entropy loss
        """
        loss = - weight_p * y_true.float() * torch.log10(y_pred + Binary_Cross_Entropy.epsilon) - \
        weight_n * (1-y_true.float()) * torch.log10(1 - y_pred + Binary_Cross_Entropy.epsilon) 
        loss = loss * weight_0.unsqueeze(dim=1)
        return loss.sum() / weight_0.sum()

def eval_classification(epoch_data: EpochData, return_metrics:bool=True) -> EpochData:
    """Calculate classification metrics from confusion matrix and return complete EpochData."""
    # Extract confusion matrix values
    tp = epoch_data.metrics['tp']
    tn = epoch_data.metrics['tn']
    fp = epoch_data.metrics['fp']
    fn = epoch_data.metrics['fn']

    # Make sure balance between +/- samples
    ratio = (tp+fn)/(tn+fp)
    tn *= ratio
    fp *= ratio
    
    # Avoid division by zero
    eps = 1e-8
    
    # Calculate metrics
    mcc = (tp*tn - tp*fn) / (math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)  # Also called recall
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    
    # F1 score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
    
    # Update the EpochData object
    epoch_data.metrics['mcc'] = mcc
    epoch_data.metrics['acc'] = accuracy
    epoch_data.metrics['f1'] = f1
    epoch_data.metrics['sens'] = sensitivity
    epoch_data.metrics['prec'] = precision
    epoch_data.metrics['spec'] = specificity

    if return_metrics:
        return epoch_data

        