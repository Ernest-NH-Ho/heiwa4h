import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.file_handling import compress, decompress
from modules.model.dropout_control import Dropout_tuner_double_bounded
from modules.data_containers import ModelOutput

class TIM(nn.Module):
    """ Designed for transmembrane & intramembrane identification """
    def __init__(self, in_dim, out_dim, mid_dim, dropout):
        super().__init__()
        # self.embed = nn.Embedding(21, 16)
        self.ffn11 = nn.Linear(in_dim, mid_dim)
        self.ffn11.pretrained = True
        self.dropout11 = nn.Dropout(dropout['low'])
        self.ffn12 = nn.Linear(in_dim, mid_dim)
        self.ffn12.pretrained = True
        self.dropout12 = nn.Dropout(dropout['low'])
        self.ffn2 = nn.Linear(mid_dim, out_dim)
        self.ffn2.pretrained = True
        
        self.dropout11.tuner = Dropout_tuner_double_bounded(
            metric=dropout['low'], metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='min'
        )
        self.dropout12.tuner = Dropout_tuner_double_bounded(
            metric=dropout['low'], metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='min'
        )
        
    def _load_default(self, logger, num_cpu):
        path = f"models/TIM/esm3_bs64_dr1_lr0.001_m6_swiglu_2layer/TIM_final_model.pth"
        decompress(f"{path}", logger=logger, num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location='cuda', weights_only=True)
        incompatible = self.load_state_dict(state_dict)
        logger.record(f"Missing keys: {incompatible.missing_keys}")
        logger.record(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", logger=logger, num_cpu=num_cpu)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # X_label = f"X_{dataset_num}"
        bs, l, d = x.size()
        a = self.ffn11(x) # .reshape(bs, l)
        a = self.dropout11(a)
        
        b = self.ffn12(x) # .reshape(bs, l)
        b = self.dropout12(b)
        if self.training:
            b = torch.clamp(b, min=-8, max=8)
        
        a = a * F.sigmoid(b)

        a = self.ffn2(a).reshape(bs, l)
        
        if self.training:
            a = torch.clamp(a, min=-8, max=8)
        # print("TIM", a)
        a = F.sigmoid(a)
        a = torch.where(a>=0.5, 1., 0.).float()
        # a = torch.where(a<0.5, 0, a)
        return ModelOutput(y_pred=a)  # * model_input[f'M_{dataset_num}']
        