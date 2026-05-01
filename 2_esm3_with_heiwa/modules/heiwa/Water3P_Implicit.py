import torch
import torch.nn as nn
from typing import List

class Water3P_Implicit(nn.Module):
    """ Use 3-point model, assume homogeneous at the external env (Poisson-Boltzmann's system) """
    """ It has its own parameters under this class """
    def __init__(self, dim_water:int, dim_qk:int, dim_v:int, num_head:int, 
                 with_scaler:bool=True, dropout_value:float=0.1):
        super().__init__() # dim_v:int=0, dim_v_se3:int, 
        self.raw_emb = nn.Parameter(torch.randn(dim_water), requires_grad=True)
        self.raw_emb.pretrained = True
        self.dim_water = dim_water
        # self.dim_v_se3 = dim_v_se3 * 3 # must be multiple of 3
        self.dim_v = dim_v
        self.num_head = num_head
        # assert self.dim_v_se3 %3 == 0

        # K core
        self.K_H = nn.Linear(dim_water, dim_qk)
        self.K_H.pretrained = True
        self.K_L = nn.Linear(dim_water, dim_qk)
        self.K_L.pretrained = True
        self.K_O = nn.Linear(dim_water, dim_qk)
        self.K_O.pretrained = True
        self.K_L2 = nn.Linear(dim_water, dim_qk)
        self.K_L2.pretrained = True
        
        # K angle
        self.Ka_H = nn.Linear(dim_water, 3)
        self.Ka_H.pretrained = True
        self.Ka_L = nn.Linear(dim_water, 3)
        self.Ka_L.pretrained = True
        self.Ka_O = nn.Linear(dim_water, 3)
        self.Ka_O.pretrained = True
        self.Ka_L2 = nn.Linear(dim_water, 3)
        self.Ka_L2.pretrained = True
        self.angle_bias = nn.Parameter(torch.randn(num_head) * 0.125, requires_grad=True)
        self.angle_bias.pretrained = True

        # V scaler
        self.with_scaler = with_scaler
        if with_scaler:
            self.V_H = nn.Linear(dim_water, dim_v)
            self.V_H.pretrained = False
            self.V_L = nn.Linear(dim_water, dim_v)
            self.V_L.pretrained = False
            self.V_O = nn.Linear(dim_water, dim_v)
            self.V_O.pretrained = False
            self.V_L2 = nn.Linear(dim_water, dim_v)
            self.V_L2.pretrained = False
        else:
            self.dim_v = 0
    
    
    def generate_k(self):
        kh = self.K_H(self.raw_emb)
        kl = self.K_L(self.raw_emb)
        ko = self.K_O(self.raw_emb)
        kl2 = self.K_L2(self.raw_emb)
        return torch.vstack([kh, kl, ko, kl2])

    def generate_ka(self):
        kh = self.Ka_H(self.raw_emb) # self.dropout_Ka_H()
        kl = self.Ka_L(self.raw_emb) # self.dropout_Ka_L()
        ko = self.Ka_O(self.raw_emb) # self.dropout_Ka_O()
        kl2 = self.Ka_L2(self.raw_emb) # self.dropout_Ka_L()
        return torch.vstack([kh, kl, ko, kl2])
        

    def generate_v(self):
        vh = self.V_H(self.raw_emb)
        vl = self.V_L(self.raw_emb)
        vo = self.V_O(self.raw_emb)
        vl2 = self.V_L2(self.raw_emb)
        return torch.vstack([vh, vl, vo, vl2])
        
        