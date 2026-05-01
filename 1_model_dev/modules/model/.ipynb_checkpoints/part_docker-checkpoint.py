import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from modules.model.dropout_control import Dropout_tuner_double_bounded
from modules.model.attention_self_geo import SelfAttentionGeometric
from modules.model.attention_cross import CrossAttention

class Docker(nn.Module):

    def __init__(self, dim_in:int, 
                 dim_qk_T_position_self:int, dim_qk_T_direction_self:int, 
                 num_heads_T_self:int, num_heads_R_self:int, 
                 attention_activation_self:str, 
                 dim_qk_cross:int, 
                 num_head_cross:int, 
                 attention_activation_cross:str, 
                 dropout:dict, num_ring:int, 
                 dim_v_T_self:int|None=None, dim_v_R_self:int|None=None, 
                 dim_qk_R_self:int|None=None, 
                 expansion_factor_self:float|None=None, 
                 dim_v_cross:int|None=None, 
                 expansion_factor_cross:float|None=None
                ):
        super().__init__()
        # lower_limit1:float=10, upper_limit1:float=12, lower_limit2:float=17, upper_limit2:float=19, 
        self.self_atten1 = SelfAttentionGeometric(
            dim_in=dim_in, 
            num_heads_T=num_heads_T_self, 
            num_heads_R=num_heads_R_self, 
            dim_qk_T_position=dim_qk_T_position_self, 
            dim_qk_T_direction=dim_qk_T_direction_self, 
            dim_v_T=dim_v_T_self, 
            dim_qk_R=dim_qk_R_self, 
            dim_v_R=dim_v_R_self, 
            expansion_factor=expansion_factor_self, 
            num_ring=num_ring, 
            dropout_value=0.15+dropout["high"]*(30-num_ring*(num_ring+1))/33 # * num_head_self * dim_v_self/(6 * 64)
        )

        self.cross_atten = CrossAttention(
            dim_x1=dim_in, 
            dim_x2=dim_in, 
            num_heads=num_head_cross, 
            dim_qk=dim_qk_cross, 
            dim_v=dim_v_cross,
            expansion_factor=expansion_factor_cross, 
            attention_activation=attention_activation_self, 
            dropout_value=0.05 + dropout["low"]*(30-num_ring*(num_ring+1))/33, 
            attention_map_only=True
        )
        

    def forward(self, X1, X2, C1, C2, sasa_gate_1, sasa_gate_2):
        """ X1 and X2 are already water-modified"""
        bs, l1, _ = X1.size()
        _, l2, _ = X2.size()

        emb1 = self.self_atten1(x=X1, m=sasa_gate_1, c=C1)
        emb2 = self.self_atten1(x=X2, m=sasa_gate_2, c=C2)

        # emb1 = self.self_atten2(x=emb1, m=sasa_gate_1, c=C1)
        # emb2 = self.self_atten2(x=emb2, m=sasa_gate_2, c=C2)
        
        atten = self.cross_atten(
            x1=emb1, x2=emb2, 
            m1=sasa_gate_1, 
            m2=sasa_gate_2, 
        )
        
        return atten