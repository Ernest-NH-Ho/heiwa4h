import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.heiwa.file_handling import compress, decompress
from modules.heiwa.data_containers import ModelOutput

class TIM(nn.Module):
    """ Designed for transmembrane & intramembrane identification """
    def __init__(self, in_dim, out_dim, mid_dim):
        super().__init__()
        # self.embed = nn.Embedding(21, 16)
        self.ffn11 = nn.Linear(in_dim, mid_dim)
        self.ffn11.pretrained = True
        self.ffn12 = nn.Linear(in_dim, mid_dim)
        self.ffn12.pretrained = True
        self.ffn2 = nn.Linear(mid_dim, out_dim)
        self.ffn2.pretrained = True
        
        
    def _load_default(self, num_cpu):
        path = f"heiwa/TIM_epoch_final_model.pth"
        decompress(f"{path}", num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location='cuda', weights_only=True)
        incompatible = self.load_state_dict(state_dict)
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", num_cpu=num_cpu)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # X_label = f"X_{dataset_num}"
        bs, l, d = x.size()
        a = self.ffn11(x) # .reshape(bs, l)
        
        b = self.ffn12(x) # .reshape(bs, l)
        if self.training:
            b = torch.clamp(b, min=-8, max=8)
        
        a = a * F.sigmoid(b)

        a = self.ffn2(a).reshape(bs, l)
        
        if self.training:
            a = torch.clamp(a, min=-8, max=8)
        # print("TIM", a)
        a = F.sigmoid(a)
        a = torch.where(a>=0.5, 1., 0.)
        return ModelOutput(y_pred=a)  # * model_input[f'M_{dataset_num}']
        