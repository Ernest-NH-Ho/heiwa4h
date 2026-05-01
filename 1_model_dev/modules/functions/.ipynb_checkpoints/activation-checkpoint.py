import torch
import torch.nn as nn


class ThreeStep:
    """ -1, 0, +1 """
    @staticmethod
    def calculate(x:torch.Tensor, x_scale=0.4):
        x = x * x_scale
        return torch.tanh(x) - x * torch.exp(-(x**2))

class SigmoidWindow:
    """ 1 only between two values, 0 beyonds """
    """ Useful for masking a certain euclidean distances x_scale=10.0, """
    @staticmethod
    def calculate(x:torch.Tensor, lower_limit=10, upper_limit=12, num_ring=3):
        # print("dist", x.size())
        # print(x)
        criterion = (x>=lower_limit) & (x<=upper_limit)
        for i in range(1, num_ring):
            criterion = criterion | ((x>=lower_limit+i*7) & (x<=upper_limit+i*7))
        # print("criterion", criterion.size())
        # print(criterion)
        # quit()
        return torch.where(criterion, 1, 0) # torch.sigmoid((x-lower_limit)*x_scale) * (1 - torch.sigmoid((x-upper_limit)*x_scale))

# class Sigmoid_0_at_0:
#     """ 1 only between two values, 0 beyonds """
#     """ Useful for masking a certain euclidean distances x_scale=10.0, """
#     @staticmethod
#     def calculate(x:torch.Tensor):
#         # print("dist", x.size())
#         # print(x)
#         return torch.sigmoid(x-8)
        
# class SwiGLU:
#     """
#     SwiGLU activation function implementation.
#     SwiGLU(x, W, V, b, c) = (Swish(xW + b) ⊙ (xV + c))
#     where Swish(x) = x * sigmoid(βx), typically β=1
#     """
    
#     @staticmethod
#     def calculate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
#         """
#         Calculate SwiGLU activation.
        
#         Args:
#             x: Input tensor
#             gate: Gate tensor
            
#         Returns:
#             SwiGLU activated tensor
#         """
#         # Swish activation (also known as SiLU)
#         swish_x = x * torch.sigmoid(x)
        
#         # SwiGLU: swish(x) * gate
#         return swish_x * gate
    
#     @staticmethod
#     def calculate_with_projections(x: torch.Tensor, W: torch.Tensor, V: torch.Tensor, 
#                                  b: torch.Tensor = None, c: torch.Tensor = None) -> torch.Tensor:
#         """
#         Calculate SwiGLU with linear projections.
        
#         Args:
#             x: Input tensor
#             W: Weight matrix for main branch
#             V: Weight matrix for gate branch  
#             b: Bias for main branch
#             c: Bias for gate branch
            
#         Returns:
#             SwiGLU activated tensor
#         """
#         # Linear projections
#         main_branch = torch.matmul(x, W)
#         gate_branch = torch.matmul(x, V)
        
#         if b is not None:
#             main_branch = main_branch + b
#         if c is not None:
#             gate_branch = gate_branch + c
        
#         # Apply SwiGLU
#         swish_main = main_branch * torch.sigmoid(main_branch)
#         return swish_main * gate_branch

