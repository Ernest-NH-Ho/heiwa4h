import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.model.dropout_control import Dropout_tuner_double_bounded

class Swish(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_value:float|None=None):
        super().__init__()
        self.ffn = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.dropout_value = dropout_value
        
        if dropout_value != None:
            self.dropout = nn.Dropout(dropout_value)
            self.dropout.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.025, thres_Q1=1.015, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='max'
            )
            
            self.dropout_gate = nn.Dropout(dropout_value)
            self.dropout_gate.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.025, thres_Q1=1.015, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='max'
            )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.dropout_value:
            x = self.dropout(x)
        a = self.ffn(x)
        b = self.gate(x)
        if self.training:
            b = torch.clamp(b, min=-8, max=8)
        b = F.sigmoid(b)
        a = a*b
        return a
        
        
class SwiGLU_Simplified(nn.Module):
    def __init__(self, in_dim: int, dropout_value:float|None=None, out_dim: int|None=None,
                 expansion_factor: float = 4.):
        super().__init__()
        self.in_dim = in_dim
        self.dropout_value = dropout_value
        if out_dim != None:
            self.out_dim = out_dim
        else:
            self.out_dim = in_dim

        smallest_dim  = min(in_dim, self.out_dim)
        multiple = smallest_dim * expansion_factor // 64
        self.hidden_dim = round(64 * multiple)

        # SwiGLU components
        self.ffn_proj = nn.Linear(in_dim, self.hidden_dim)
        self.gate_proj = nn.Linear(in_dim, self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim, self.out_dim)

        # dropouts
        if dropout_value != None:
            self.dropout_ffn = nn.Dropout(dropout_value)
            self.dropout_ffn.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='min'
            )
            
            self.dropout_gate = nn.Dropout(dropout_value)
            self.dropout_gate.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='min'
            )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(in_dim)
        self.layer_norm2 = nn.LayerNorm(self.out_dim)
        
        # Residual connection projection (if dimensions don't match)
        if in_dim != self.out_dim:
            self.residual_proj = nn.Linear(in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """         Forward pass through SwiGLU FFN with residual connection.
        
        Args:
            x: Input tensor of shape (batch_size, in_dim)
            
        Returns:
            Output tensor of shape (batch_size, out_dim)
        """
        # Store input for residual connection
        residual = x
        
        # Layer normalization
        x = self.layer_norm1(x)
        
        # SwiGLU computation
        gate = self.gate_proj(x)  # (batch_size, hidden_dim)
        x = self.ffn_proj(x)      # (batch_size, hidden_dim)
        
        # Apply SwiGLU activation
        if self.training:
            gate = F.sigmoid(gate) # up
        
        # Apply dropout
        if self.dropout_value:
            gate = self.dropout_gate(gate)
            x = self.dropout_ffn(x)
        
        # Gating + Down projection
        x = x * gate
        x = self.down_proj(x)  # (batch_size, out_dim)
        
        # Residual connection
        if hasattr(self, "residual_proj"):
            residual = self.residual_proj(residual)
        
        # Add residual (only if dimensions match)
        assert residual.shape[-1] == x.shape[-1]
        x = x + residual
        
        # Final layer normalization
        x = self.layer_norm2(x)

        # F.sigmoid()
        return x

    
