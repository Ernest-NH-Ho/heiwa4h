import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.file_handling import compress, decompress
from modules.model.basic_layers import Swish
from modules.data_containers import ModelOutput

class ExposedResiduePrediction(nn.Module):
    """ Can be used for  """
    def __init__(self, dim_prot, dim_intermediate, dim_out, dropout_value:float):
        # dim_intermediate1, dim_intermediate2, 
        super().__init__()
        self.swish = Swish(in_dim=dim_prot, out_dim=dim_intermediate, dropout_value=dropout_value/500)
        self.ffn = nn.Linear(dim_intermediate, dim_out)

    def _load_default(self, logger, num_cpu):
        path = f"models/SASAgate_202604/SASA_bs32_dr1_3_i1024/ExposedResiduePrediction_epoch_47_model.pth"
        decompress(f"{path}", logger=logger, num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location='cuda', weights_only=True)
        incompatible = self.load_state_dict(state_dict)
        logger.record(f"Missing keys: {incompatible.missing_keys}")
        logger.record(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", logger=logger, num_cpu=num_cpu)
        
    def forward(self,  x, dataset_num=""):
        bs, l, _ = x.size()
        x = self.swish(x)
        x = self.ffn(x).reshape(bs, l) # * model_input['M_'+dataset_num]
        if self.training:
            x = torch.clamp(x, max=8, min=-8)
        x = F.sigmoid(x)
        # x = torch.where(x>0.6, 1, x)
        # x = torch.where(x<0.2, 0, x)
        return ModelOutput(y_pred=x)

