import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.file_handling import compress, decompress
from modules.model.basic_layers import Swish
from modules.model.dropout_control import Dropout_tuner_double_bounded
from modules.data_containers import ModelOutput


class DisorderPred(nn.Module):
    """ Can be used for  """
    def __init__(self, in_dim, intermediate_dim, out_dim, dropout_value:float):
        # dim_intermediate1, dim_intermediate2, 
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.swish = Swish(in_dim=in_dim, out_dim=intermediate_dim, dropout_value=dropout_value*0.1)
        self.ffn = nn.Linear(intermediate_dim, out_dim)

    def _load_default(self, logger, num_cpu):
        path = f"models/data_202604/Disorder_prediction_Swish_Linear_stage1/esm3_bs64_dr1_3_i{self.intermediate_dim}/DisorderPred_epoch_10_model.pth"
        decompress(f"{path}", logger=logger, num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location='cuda', weights_only=True)
        incompatible = self.load_state_dict(state_dict)
        logger.record(f"Missing keys: {incompatible.missing_keys}")
        logger.record(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", logger=logger, num_cpu=num_cpu)
        
    def forward(self, model_input):
        x = model_input['X']
        m = model_input['M']
        bs, l, _ = x.size()
        
        x = self.swish(x)
        x = self.ffn(x).reshape(bs, l) # * model_input['M_'+dataset_num]
        if self.training:
            x = torch.clamp(x, max=8, min=-8)
        x = F.sigmoid(x) * m
        x = torch.where(x>=0.6, 1, x)
        x = torch.where(x<0.25, 0, x)
        # x = torch.where(x<0.5, 0, 1)
        return ModelOutput(y_pred=x)

