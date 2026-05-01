import torch
import torch.nn as nn
import torch.nn.functional as F

class OrbibalMorphology:
    def __init__(self, ref_unit_vector:torch.tensor, radius_x:float=0, radius_y:float=0, center_coord:torch.tensor, orbital_type:str='p'):
        self.ref_unit_vector = ref_unit_vector  # this is the vector of the entire molecule
        self.orbital_type = orbital_type
        if self.orbital_type == 'p':
            self.radius_x = F.Softplus(nn.Parameter(torch.tensor([radius_x], dtype=torch.float32, device='cuda')))
            self.radius_y = F.Softplus(nn.Parameter(torch.tensor([radius_y], dtype=torch.float32, device='cuda')))
        else:
            self.radius_x = F.Softplus(nn.Parameter(torch.tensor([radius_x], dtype=torch.float32, device='cuda')))
            self.radius_y = F.Softplus(nn.Parameter(torch.tensor([radius_y], dtype=torch.float32, device='cuda')))
        self.center_coord = center_coord # center of the orbital


class OrbitalAttention:
    def __init__(self):
        self.