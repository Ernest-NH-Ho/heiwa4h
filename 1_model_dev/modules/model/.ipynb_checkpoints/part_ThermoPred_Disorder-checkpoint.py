import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.file_handling import compress, decompress
from modules.model.basic_layers import Swish
from modules.model.part_Disorder import DisorderPred
from modules.model.dropout_control import Dropout_tuner_double_bounded
from modules.data_containers import ModelOutput

class ThermoPred_DisorderCorrection(nn.Module):
    def __init__(self, in_dim:int, intermediate_dim:int, out_dim:int, dropout:dict):
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.rmsn = nn.RMSNorm(in_dim)
        self.swish1 = Swish(in_dim=in_dim, out_dim=intermediate_dim, dropout_value=dropout["low"])
        self.swish_disor = Swish(in_dim=in_dim, out_dim=intermediate_dim, dropout_value=dropout["low"])
        
        self.bd_lstm = nn.LSTM(
            input_size=intermediate_dim, 
            hidden_size=intermediate_dim*4, 
            num_layers=1, 
            bias=True, 
            dropout=dropout["low"]*0.1, 
            bidirectional=True,
        )
        self.ffn = nn.Linear(intermediate_dim, 1)

    def _load_default(logger, num_cpu):
        path = f"./models/data_202604/Thermo_NewData_Swish/Thermo_NewData_Swish_L100/esm3_bs32_dr1_3_i{self.intermediate_dim}/Thermo_epoch_final_model.pth"
        decompress(f"{path}", logger=logger, num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location='cuda', weights_only=True)
        incompatible = self.load_state_dict(state_dict, strict=False)
        logger.record(f"Missing keys: {incompatible.missing_keys}")
        logger.record(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", logger=logger, num_cpu=num_cpu)

    def forward(self, model_input, disorder_logit, water_modification=0, save_intermediate_layer=False):
        x = model_input['X']
        m = model_input['M']
        bs, l, _ = x.size()

        x = x + water_modification
        x = self.rmsn(x)
        middle_layer = self.swish1(x)

        disorder_layer = self.swish_disor(x)

        middle_layer = middle_layer + disorder_layer * disorder_logit
        middle_layer = middle_layer + self.bd_lstm(middle_layer)

        final_layer = self.ffn(middle_layer) * m

        y_pred = final_layer.sum(dim=-1) / model_input["L"]
        if self.training:
            y_pred = torch.clamp(y_pred, max=8, min=-8)
        y_pred = F.sigmoid(y_pred)

        # Collect intermediate output if needed
        intermediate_output = None
        if save_intermediate_layer:
            intermediate_output = {"final_layer":final_layer}
            
        return ModelOutput(y_pred=y_pred, intermediate_output=intermediate_output)