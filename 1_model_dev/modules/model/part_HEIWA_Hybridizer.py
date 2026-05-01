import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.model.basic_layers import Swish, SwiGLU_Simplified 
from modules.data_containers import ModelOutput

class HEIWA_Hybridizer(nn.Module):
    def __init__(self, in_dim, v_compress, expansion_factor, out_dim, dropout):
        super().__init__()
        mid_dim = round(in_dim*v_compress)
        self.Wo_w = Swish(in_dim=in_dim, out_dim=mid_dim, dropout_value=dropout["high"]*(in_dim/1600))
        self.out_proj = SwiGLU_Simplified(in_dim=mid_dim, out_dim=out_dim, dropout_value=dropout["low"]*expansion_factor, 
                                          expansion_factor=expansion_factor)
        self.out_proj.pretrained = False
        # self.out_proj = SwiGLU_2input(in_dim_ffn=dim_o, in_dim_gate=dim_prot, out_dim=dim_prot, dropout_value=dropout["low"]*1.25*expansion_factor, 
        #                               expansion_factor=expansion_factor)

    def forward(self, x_imp=None, x_exp=None, tim_gate=0, sasa_gate=1):
        """ RMSNorm should be done by the part that produces the x_imp and x_exp """
        bs, l, _ = x_imp.size()

        if x_imp!=None and x_exp!=None:
            water_modification = self.Wo_w(torch.cat((x_imp, x_exp), axis=-1))
        elif x_imp!=None:
            water_modification = self.Wo_w(x_imp)
        elif x_exp!=None:
            water_modification = self.Wo_w(x_exp)
            
        if isinstance(tim_gate, torch.Tensor):
            tim_gate = tim_gate.reshape(bs, l, 1)
        if isinstance(sasa_gate, torch.Tensor):
            sasa_gate = sasa_gate.reshape(bs, l, 1)
        
        water_modification = self.out_proj(water_modification) * (1-tim_gate) * sasa_gate
        return ModelOutput(y_pred=water_modification)
