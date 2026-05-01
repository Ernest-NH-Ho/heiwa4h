import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.file_handling import compress, decompress
from modules.model.basic_layers import Swish
from modules.model.dropout_control import Dropout_tuner_double_bounded
from modules.data_containers import ModelOutput

class ThermoPred(nn.Module):
    def __init__(self, in_dim:int, intermediate_dim:int, out_dim:int, dropout:dict):
        super().__init__()
        self.swish1 = Swish(in_dim=in_dim, out_dim=intermediate_dim, dropout_value=dropout["low"]*2)
        # self.ffn1 = nn.Linear(in_dim, intermediate_dim)
        # self.ffn2 = nn.Linear(intermediate_dim, out_dim)
        self.swish2 = Swish(in_dim=intermediate_dim, out_dim=out_dim, dropout_value=None)
        # self.final_L_factor = nn.Parameter(torch.tensor([0], dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.final_L_bias = nn.Parameter(torch.tensor([-1], dtype=torch.float32, device='cuda'), requires_grad=True)

        # self._load_default(intermediate_dim)
        
        # self.dropout = nn.Dropout(dropout["low"]*0.5)
        # self.dropout.tuner = Dropout_tuner_double_bounded(
        #     metric=0.04+intermediate_dim*0.005, metric_upper_limit=1, metric_lower_limit=0, \
        #     thres_Q2=1.025, thres_Q1=1.015, max_step=0.25, epoch_scale=10, \
        #     eval_interval=4, mode='max'
        # ) # max_mode: thres_Q2=1.025, thres_Q1=1.015

    def _load_default(self, logger, num_cpu):
        # path = f"./models/data_20251228/Thermo_NewData20251228/Thermo_NewData_Swish_L100/esm3_bs56_dr1_3_i16/Thermo_epoch_63_model.pth"
        path = f"./models/data_202604/Thermo_NewData_Swish_L100_coarse/esm3_bs64_dr1_3_struci96/Thermo_epoch_final_model.pth"
        # path = f"models/data_202604/ImpV_SelfGate_rot1_angAtten1/HEIWA_SelfGate_rot1_angAtten1_exhd3_Imp/esm3_SelfGate_bs32_dr1_3_v48_v_compress0.8_Imp_hd3/Thermo_epoch_final_model.pth"
        # path = f"models/data_202604/ImpV_SelfGate_rot1_angAtten1/HEIWA_SelfGate_rot1_angAtten1_exhd3_Imp/esm3_SelfGate_bs32_dr1_3_v48_v_compress0.3_Imp_hd3/Thermo_epoch_final_model.pth"
        # path = f"./models/data_20251228/ImpV_SelfGate_rot1/HEIWA3P_MixV_trainedMixer_SelfGate_sgmd63_relu1_rot1_angAtten1_exhd5_Imp/esm3_MixV_trainedMixer_SelfGate_bs32_dr1_3_v320_v_compress0.4_expd4.0/Thermo_epoch_12_model.pth"
        decompress(f"{path}", logger=logger, num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location='cuda', weights_only=True)
        incompatible = self.load_state_dict(state_dict, strict=False)
        logger.record(f"Missing keys: {incompatible.missing_keys}")
        logger.record(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", logger=logger, num_cpu=num_cpu)
        
    def forward(self, model_input:Dict[str, torch.Tensor], water_modification:torch.Tensor|float=0, save_intermediate_layer:bool=False) -> torch.Tensor:
        bs, l, _ = model_input["X"].size()
        # print(model_input["X"])
        # print(water_modification)
        x = model_input["X"]+water_modification
        # 
        middle_layer = self.swish1(x)
        # print(self.bd_lstm(middle_layer)[0].size())
        # quit()
        # middle_layer = middle_layer + self.bd_lstm_res(self.bd_lstm(middle_layer)[0])
        # middle_layer = self.rmsn(middle_layer)
        
        final_layer = self.swish2(middle_layer).reshape(bs, l)
        final_layer = final_layer * model_input["M"]

        y_pred = final_layer.sum(dim=-1) / model_input["L"]
        if self.training:
            y_pred = torch.clamp(y_pred, max=7.5, min=-7.5)
        y_pred = F.sigmoid(y_pred)

        # Collect intermediate output if needed
        intermediate_output = None
        # print(save_intermediate_layer)
        if save_intermediate_layer:
            intermediate_output = final_layer
            
        return ModelOutput(y_pred=y_pred, intermediate_output=intermediate_output)
        