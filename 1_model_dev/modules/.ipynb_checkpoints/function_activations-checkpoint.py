# modules/function_activations.py

import torch
import torch.nn as nn

class SwiGLU:
    """
    SwiGLU activation function implementation.
    SwiGLU(x, W, V, b, c) = (Swish(xW + b) ⊙ (xV + c))
    where Swish(x) = x * sigmoid(βx), typically β=1
    """
    
    @staticmethod
    def calculate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Calculate SwiGLU activation.
        
        Args:
            x: Input tensor
            gate: Gate tensor
            
        Returns:
            SwiGLU activated tensor
        """
        # Swish activation (also known as SiLU)
        swish_x = x * torch.sigmoid(x)
        
        # SwiGLU: swish(x) * gate
        return swish_x * gate
    
    @staticmethod
    def calculate_with_projections(x: torch.Tensor, W: torch.Tensor, V: torch.Tensor, 
                                 b: torch.Tensor = None, c: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate SwiGLU with linear projections.
        
        Args:
            x: Input tensor
            W: Weight matrix for main branch
            V: Weight matrix for gate branch  
            b: Bias for main branch
            c: Bias for gate branch
            
        Returns:
            SwiGLU activated tensor
        """
        # Linear projections
        main_branch = torch.matmul(x, W)
        gate_branch = torch.matmul(x, V)
        
        if b is not None:
            main_branch = main_branch + b
        if c is not None:
            gate_branch = gate_branch + c
        
        # Apply SwiGLU
        swish_main = main_branch * torch.sigmoid(main_branch)
        return swish_main * gate_branch

