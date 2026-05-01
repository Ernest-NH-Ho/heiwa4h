import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.heiwa.basic_layers import Swish, SwiGLU_Simplified
from modules.heiwa.file_handling import compress, decompress
from modules.heiwa.data_containers import ModelOutput

class HEIWA_Hybridizer(nn.Module):
    def __init__(self, in_dim, v_compress, expansion_factor, out_dim, dropout=None):
        super().__init__()
        mid_dim = round(in_dim*v_compress)
        self.Wo_w = Swish(in_dim=in_dim, out_dim=mid_dim, dropout_value=0)
        self.out_proj = SwiGLU_Simplified(in_dim=mid_dim, out_dim=out_dim, dropout_value=0, 
                                          expansion_factor=expansion_factor)
        self.out_proj.pretrained = False
        # self.out_proj = SwiGLU_2input(in_dim_ffn=dim_o, in_dim_gate=dim_prot, out_dim=dim_prot, dropout_value=dropout["low"]*1.25*expansion_factor, 
        #                               expansion_factor=expansion_factor)

    def _load_default(self, num_cpu):
        path = f"heiwa/HEIWA_Hybridizer_epoch_final_model.pth"
        decompress(f"{path}", num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location='cuda', weights_only=True)
        # print(state_dict.keys())
        # quit()
        incompatible = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", num_cpu=num_cpu)
        
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