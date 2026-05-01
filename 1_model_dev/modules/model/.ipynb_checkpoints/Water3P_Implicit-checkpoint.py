import torch
import torch.nn as nn
from typing import List
from modules.model.dropout_control import Dropout_tuner_double_bounded

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
        if self.num_head > 2:
            self.K_O = nn.Linear(dim_water, dim_qk)
            self.K_O.pretrained = True
        if self.num_head == 4:
            self.K_L2 = nn.Linear(dim_water, dim_qk)
            self.K_L2.pretrained = True
        elif self.num_head == 5:
            self.K_Na = nn.Linear(dim_water, dim_qk)
            self.K_Na.pretrained = True
            self.K_Cl = nn.Linear(dim_water, dim_qk)
            self.K_Cl.pretrained = True
        # self.K_L2 = nn.Linear(dim_water, dim_qk)
        # self.K_L2.pretrained = True
        
        # K angle
        self.Ka_H = nn.Linear(dim_water, 3)
        self.Ka_H.pretrained = True
        self.Ka_L = nn.Linear(dim_water, 3)
        self.Ka_L.pretrained = True
        if self.num_head > 2:
            self.Ka_O = nn.Linear(dim_water, 3)
            self.Ka_O.pretrained = True
        if self.num_head == 4:
            self.Ka_L2 = nn.Linear(dim_water, 3)
            self.Ka_L2.pretrained = True
        elif self.num_head == 5:
            self.Ka_Na = nn.Linear(dim_water, 3)
            self.Ka_Na.pretrained = True
            self.Ka_Cl = nn.Linear(dim_water, 3)
            self.Ka_Cl.pretrained = True
        # self.Ka_L2 = nn.Linear(dim_water, 3)
        # self.Ka_L2.pretrained = True
        self.angle_bias = nn.Parameter(torch.randn(num_head) * 0.125, requires_grad=True)
        self.angle_bias.pretrained = True

        # V scaler
        self.with_scaler = with_scaler
        if with_scaler:
            self.V_H = nn.Linear(dim_water, dim_v)
            self.V_H.pretrained = False
            self.V_L = nn.Linear(dim_water, dim_v)
            self.V_L.pretrained = False
            if self.num_head > 2:
                self.V_O = nn.Linear(dim_water, dim_v)
                self.V_O.pretrained = False
            if self.num_head == 4:
                self.V_L2 = nn.Linear(dim_water, dim_v)
                self.V_L2.pretrained = False
            elif self.num_head == 5:
                self.V_Na = nn.Linear(dim_water, dim_v)
                self.V_Na.pretrained = False
                self.V_Cl = nn.Linear(dim_water, dim_v)
                self.V_Cl.pretrained = False
        else:
            self.dim_v = 0
    
        # === Dropout ===
        # K
        self.dropout_K_H = nn.Dropout(dropout_value*2)
        self.dropout_K_H.tuner = Dropout_tuner_double_bounded(
            metric=dropout_value*2, metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='max'
        )
        self.dropout_K_L = nn.Dropout(dropout_value*2)
        self.dropout_K_L.tuner = Dropout_tuner_double_bounded(
            metric=dropout_value*2, metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='max'
        )
        if self.num_head > 2:
            self.dropout_K_O = nn.Dropout(dropout_value*2)
            self.dropout_K_O.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value*2, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='max'
            )
        if self.num_head == 4:
            self.dropout_K_L2 = nn.Dropout(dropout_value*2)
            self.dropout_K_L2.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value*2, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='max'
            )
        elif self.num_head == 5:
            self.dropout_K_Na = nn.Dropout(dropout_value*2)
            self.dropout_K_Na.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value*2, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='max'
            )
            self.dropout_K_Cl = nn.Dropout(dropout_value*2)
            self.dropout_K_Cl.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value*2, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='max'
            )
    
        # V scaler
        if with_scaler:
            self.dropout_V_H = nn.Dropout(dropout_value*self.dim_v/64)
            self.dropout_V_H.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value*self.dim_v/64, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='max'
            )
            self.dropout_V_L = nn.Dropout(dropout_value*self.dim_v/64)
            self.dropout_V_L.tuner = Dropout_tuner_double_bounded(
                metric=dropout_value*self.dim_v/64, metric_upper_limit=1, metric_lower_limit=0, \
                thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                eval_interval=4, mode='max'
            )
            if self.num_head > 2:
                self.dropout_V_O = nn.Dropout(dropout_value*self.dim_v/64)
                self.dropout_V_O.tuner = Dropout_tuner_double_bounded(
                    metric=dropout_value*self.dim_v/64, metric_upper_limit=1, metric_lower_limit=0, \
                    thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                    eval_interval=4, mode='max'
                )
            if self.num_head == 4:
                self.dropout_V_L2 = nn.Dropout(dropout_value*self.dim_v/64)
                self.dropout_V_L2.tuner = Dropout_tuner_double_bounded(
                    metric=dropout_value*self.dim_v/64, metric_upper_limit=1, metric_lower_limit=0, \
                    thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                    eval_interval=4, mode='max'
                )
            elif self.num_head == 5:
                self.dropout_V_Na = nn.Dropout(dropout_value*self.dim_v/64)
                self.dropout_V_Na.tuner = Dropout_tuner_double_bounded(
                    metric=dropout_value*self.dim_v/64, metric_upper_limit=1, metric_lower_limit=0, \
                    thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                    eval_interval=4, mode='max'
                )
                self.dropout_V_Cl = nn.Dropout(dropout_value*self.dim_v/64)
                self.dropout_V_Cl.tuner = Dropout_tuner_double_bounded(
                    metric=dropout_value*self.dim_v/64, metric_upper_limit=1, metric_lower_limit=0, \
                    thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
                    eval_interval=4, mode='max'
                )

    def generate_k(self):
        kh = self.dropout_K_H(self.K_H(self.raw_emb))
        kl = self.dropout_K_L(self.K_L(self.raw_emb))
        if self.num_head == 2:
            return torch.vstack([kh, kl])
        elif self.num_head == 3:
            ko = self.dropout_K_O(self.K_O(self.raw_emb))
            return torch.vstack([kh, kl, ko])
        elif self.num_head == 4:
            ko = self.dropout_K_O(self.K_O(self.raw_emb))
            kl2 = self.dropout_K_L2(self.K_L2(self.raw_emb))
            return torch.vstack([kh, kl, ko, kl2])
        elif self.num_head == 5:
            ko = self.dropout_K_O(self.K_O(self.raw_emb))
            kna = self.dropout_K_Na(self.K_Na(self.raw_emb))
            kcl = self.dropout_K_Cl(self.K_Cl(self.raw_emb))
            return torch.vstack([kh, kl, ko, kna, kcl]) # kl2, 

    def generate_ka(self):
        kh = self.Ka_H(self.raw_emb) # self.dropout_Ka_H()
        kl = self.Ka_L(self.raw_emb) # self.dropout_Ka_L()
        if self.num_head == 2:
            return torch.vstack([kh, kl])
        elif self.num_head == 3:
            ko = self.Ka_O(self.raw_emb) # self.dropout_Ka_O()
            return torch.vstack([kh, kl, ko])
        elif self.num_head == 4:
            ko = self.Ka_O(self.raw_emb) # self.dropout_Ka_O()
            kl2 = self.Ka_L2(self.raw_emb) # self.dropout_Ka_O()
            return torch.vstack([kh, kl, ko, kl2])
        elif self.num_head == 5:
            ko = self.Ka_O(self.raw_emb) # self.dropout_Ka_O()
            kna = self.Ka_Na(self.raw_emb) # self.dropout_Ka_O()
            kcl = self.Ka_Cl(self.raw_emb) # self.dropout_Ka_O()
            return torch.vstack([kh, kl, ko, kna, kcl]) # , kl2
        
    def generate_v(self):
        vh = self.dropout_V_H(self.V_H(self.raw_emb))
        vl = self.dropout_V_L(self.V_L(self.raw_emb))
        if self.num_head == 2:
            return torch.vstack([vh, vl])
        elif self.num_head == 3:
            vo = self.dropout_V_O(self.V_O(self.raw_emb))
            return torch.vstack([vh, vl, vo])
        elif self.num_head == 4:
            vo = self.dropout_V_O(self.V_O(self.raw_emb))
            vl2 = self.dropout_V_L2(self.V_L2(self.raw_emb))
            return torch.vstack([vh, vl, vo, vl2])
        elif self.num_head == 5:
            vo = self.dropout_V_O(self.V_O(self.raw_emb))
            vna = self.dropout_V_Na(self.V_Na(self.raw_emb))
            vcl = self.dropout_V_Cl(self.V_Cl(self.raw_emb))
            return torch.vstack([vh, vl, vo, vna, vcl]) # vl2
        