import torch
import torch.nn as nn
from typing import Dict
from modules.heiwa.data_containers import ModelOutput
from modules.heiwa.file_handling import compress, decompress
from modules.heiwa.part_HEIWA_Imp import HEIWA_Imp
from modules.heiwa.part_HEIWA_Hybridizer import HEIWA_Hybridizer
from modules.heiwa.part_TIM import TIM
from modules.heiwa.part_SASA import ExposedResiduePrediction
from modules.heiwa.basic_layers import *

class HEIWA3H(nn.Module):
    def __init__(self):
        super().__init__()
        self.parts = nn.ModuleDict()

        self.parts['TIM'] = TIM(
            in_dim=1536, out_dim=1, mid_dim=6,
        )
        self.parts["ExposedResiduePrediction"] = ExposedResiduePrediction(
            dim_prot=1536, 
            dim_intermediate=1024, 
            dim_out=1,
        )
        
        self.parts['HEIWA_Imp'] = HEIWA_Imp(
            le_cha_bias_action = -999, 
            le_cha_bias_final = torch.tensor([0.2511, 0.2511, 0.2511, 0.2511], dtype=torch.float32, device='cuda:0'), 
            with_scaler=True, 
            num_solv_head=4, 
            num_prot_head=3, 
            head_label=['water', 'water', 'water', 'water'], 
            rot_head_idx=[0, 1, 2, 3], 
            head_require_rot=[True, True, True, True], 
            num_rot_head=4, 
            z_dim_to_qk_dim_ratio=5, 
            dim_qk=128,  
            dim_v=192,
            dim_prot=1536, 
            dim_sigmoid_factor=0,
        )
        
        self.parts["HEIWA_Hybridizer"] = HEIWA_Hybridizer(
            in_dim=192*4*3, 
            v_compress=0.8, 
            expansion_factor=4, 
            out_dim=1536, 
        )
        
        
        for part in ["TIM", "ExposedResiduePrediction", "HEIWA_Imp", "HEIWA_Hybridizer"]: # 
            print(f"Loading Pretrained model weights for {part}")
            self.parts[part]._load_default(num_cpu=24)

    def forward(self, x:torch.Tensor) -> ModelOutput:
        """Forward pass through the model."""
        bs, l, _ = x.size()

        tim_output = self.parts['TIM'](x)
        tim_gate = tim_output.y_pred
        
        ExposedResiduePrediction_output = self.parts["ExposedResiduePrediction"](x)
        sasa_gate = ExposedResiduePrediction_output.y_pred
        
        # Implicit
        heiwa_imp_output = self.parts['HEIWA_Imp'](x=x)
        water_modification_imp = heiwa_imp_output.y_pred

        # Hybydridizer
        hyd_output = self.parts["HEIWA_Hybridizer"](
            x_imp = water_modification_imp, 
            tim_gate = tim_gate, 
            sasa_gate = sasa_gate
        )
        water_modification = hyd_output.y_pred
        
        return x + water_modification
        