import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.file_handling import compress, decompress
from modules.model.basic_layers import Swish, SwiGLU_Simplified # SwiGLU_2input, 
from modules.model.dropout_control import Dropout_tuner_double_bounded
from modules.data_containers import ModelOutput

class HEIWA_Exp(nn.Module):
    def __init__(self, 
                 le_cha_bias_action:float, le_cha_bias_final:torch.Tensor, 
                 num_head:int, 
                 dim_qk:int,  dim_v:int, dim_prot:int, 
                 dim_sigmoid_factor: float, 
                 dropout:dict):
        super().__init__() # dim_w_v:int,
        self.parts = nn.ModuleDict()
        self.le_cha_bias_action = le_cha_bias_action
        self.le_cha_bias_final = le_cha_bias_final
        self.num_head = num_head # must keep, useful in forward
        self.dim_qk = dim_qk # must keep, useful in forward
        self.dim_v = dim_v   # only useful in hp_tuning (dim_v per head)
        self.dim_v_relu = self.dim_v - round(dim_v * dim_sigmoid_factor)
        print("dim_sigmoid_factor =", dim_sigmoid_factor)
        print("self.dim_v_relu =", self.dim_v_relu)

        self.Qp = nn.Linear(dim_prot, num_head*dim_qk)
        self.Qp.pretrained = False
        
        self.Qp_a = nn.Linear(dim_prot, num_head*3)
        self.Qp_a.pretrained = False
        self.Qp_a_bias = nn.Linear(dim_prot, num_head)
        self.Rp = nn.Linear(dim_prot, num_head*9)
        self.Rp.pretrained = False

        self.Kw = nn.Linear(dim_prot, num_head*dim_qk)
        self.Kw.pretrained = False

        self.Kw_a = nn.Linear(dim_prot, num_head*3)
        self.Kw_a.pretrained = False
        self.Kw_a_bias = nn.Parameter(torch.randn(max(2, num_head-2)) * 0.125, requires_grad=True)
        self.Kw_a_bias.pretrained = False
        
        dim_v_tot = dim_v * num_head
        self.dim_v_relu = self.dim_v - round(dim_v * dim_sigmoid_factor)
        
        self.Vw = nn.Linear(dim_prot, dim_v_tot)
        self.Vw.pretrained = False

        dim_v_tot_relu = self.dim_v_relu * num_head
        if dim_v_tot_relu > 0:
            self.rmsn_relu = nn.RMSNorm(dim_v_tot_relu)
            self.rmsn_relu.pretrained = False
        if dim_v_tot - dim_v_tot_relu > 0:
            self.rmsn_sigmoid = nn.RMSNorm(dim_v_tot - dim_v_tot_relu)
            self.rmsn_sigmoid.pretrained = False

        # ==== Dropout ====
        self.dropout_Qp = nn.Dropout(dropout['low']*0.5)
        self.dropout_Qp.tuner = Dropout_tuner_double_bounded(
            metric=dropout['low'], metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='min'
        )
        
        self.dropout_Qp_a_bias = nn.Dropout(dropout['low']*0.01)
        self.dropout_Qp_a_bias.tuner = Dropout_tuner_double_bounded(
            metric=dropout['low']*0.01, metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='min'
        )

        self.dropout_Kw = nn.Dropout(dropout['low']*0.5)
        self.dropout_Kw.tuner = Dropout_tuner_double_bounded(
            metric=dropout['low'], metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='min'
        )
        
        self.dropout_Vw = nn.Dropout(dropout['high']*1.6)
        self.dropout_Vw.tuner = Dropout_tuner_double_bounded(
            metric=dropout['high']*1.5, metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='min'
        )
        

    def _load_default(self, logger, num_cpu):
        path = f"./models/data_20251228/HEIWA3P_Atten/Qp_2.pth"
        decompress(f"{path}", logger=logger, num_cpu=num_cpu)
        
        state_dict = torch.load(path, 
                                map_location='cuda', weights_only=True)
        incompatible = self.load_state_dict(state_dict, strict=False)
        logger.record(f"Missing keys: {incompatible.missing_keys}")
        logger.record(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", logger=logger, num_cpu=num_cpu)
        
    def product_rotation_attention(self, prot_emb):
        return self.Rp(prot_emb)

    def forward(self, x:torch.Tensor, water_bias:nn.Parameter|str="empty", eps:float=1e-6) -> torch.Tensor:
        """Forward pass through the model."""
        bs, l, d = x.size()

        ## =================== Core attention ===========================================
        kw = self.Kw(self.dropout_Kw(x)).reshape(
            bs, l, 
            self.num_head, 
            self.dim_qk
        )
        qp = self.Qp(self.dropout_Qp(x)).reshape(
            bs, l, 
            self.num_head, 
            self.dim_qk
        )
        
        # =========================== Rotation attention: prot_qa ==================
        # Kw_a
        kw_a = self.Kw_a(x).reshape(
            bs, l, 
            self.num_head, 
            3
        )
        # # Qp_a
        qp_a_bias = self.Qp_a_bias(self.dropout_Qp_a_bias(x)).reshape(
            bs, l, 
            self.num_head
        )

        # Protein qa
        qp_a = self.Qp_a(x).reshape(
            bs, l, 
            self.num_head, 
            3
        ) # self.parts["Protein"].dropout_Qp_a()
        qp_a = F.normalize(qp_a, p=2, dim=-1, eps=1e-12)
        
        # Rotation matrix: Rp
        Rp = self.product_rotation_attention(prot_emb=x).reshape(
            bs, l, self.num_head, 
            3, 3
        )
        Rp = F.normalize(Rp, p=2, dim=-2, eps=1e-12)

        # Rotation of prot_qa
        qp_a = torch.einsum('...ij,...j->...i', Rp, qp_a)
        
        # ============== attention ==============
        atten = (qp * kw).sum(dim=-1)
        atten = atten + torch.where(self.le_cha_bias_action > self.le_cha_bias_final, self.le_cha_bias_action, self.le_cha_bias_final)

        atten_a = (qp_a * kw_a / 3**0.5).sum(dim=-1) * 8
        if self.training:
            atten_a = torch.clamp(atten_a, max=8, min=-8)

        if water_bias != "empty":
            solv_bias = torch.vstack([water_bias[0], water_bias[1], self.Kw_a_bias[0], self.Kw_a_bias[1]]).squeeze() # NaCl comes from self.Kw_a_bias
        else:
            solv_bias = torch.vstack([self.Kw_a_bias[0], self.Kw_a_bias[1]]).squeeze() # NaCl comes from self.Kw_a_bias
        atten = atten * (1 - F.sigmoid(atten_a + solv_bias + qp_a_bias))
        
        # ============== Atten_relu & Atten_sigmoid =================================
        atten_relu = F.relu(atten)
        
        if self.training:
            atten_sigmoid = torch.clamp(atten, max=8, min=-8)
            atten_sigmoid = F.sigmoid(atten_sigmoid)
        else:
            atten_sigmoid = F.sigmoid(atten)

        # Compile attetion loss
        # if self.num_head == 4:
        #     exp_atten_loss = ((atten_relu[..., 0] + atten_relu[..., 2]) * (atten_relu[..., 1] + atten_relu[..., 3])).sum() # water + NaCl
        # elif self.num_head == 2:
        exp_atten_loss = (atten_relu[..., 0] * atten_relu[..., 1]).sum() # NaCl
        
        # ======================= Make V ============================
        vw = self.Vw(self.dropout_Vw(x)).reshape(
            bs, l, 
            self.num_head, 
            self.dim_v
        )
        
        # =========================== Message passing | Do not apply SASA & TIM gate here ===================
        # relu
        if self.dim_v_relu > 0:
            water_modification_relu = atten_relu.reshape(bs, l, self.num_head, 1) * vw[..., :self.dim_v_relu]
            water_modification_relu = water_modification_relu.reshape(bs, l, -1)
            water_modification_relu = self.rmsn_relu(water_modification_relu)
        else:
            water_modification_relu = torch.tensor([], dtype=torch.float32, device="cuda")
        
        # sigmoid
        if self.dim_v_relu < self.dim_v:
            water_modification_sigmoid = atten_sigmoid.reshape(bs, l, self.num_head, 1) * vw[..., self.dim_v_relu:]
            water_modification_sigmoid = water_modification_sigmoid.reshape(bs, l, -1)
            water_modification_sigmoid = self.rmsn_sigmoid(water_modification_sigmoid)
        else:
            water_modification_sigmoid = torch.tensor([], dtype=torch.float32, device="cuda")
        
        water_modification = torch.cat((water_modification_relu, water_modification_sigmoid), axis=-1).reshape(bs, l, -1)
        
        return ModelOutput(
            y_pred=water_modification, 
            intra_model_loss=exp_atten_loss, 
            intermediate_output={"relu_head": [atten_relu[..., i] for i in range(self.num_head)], "sigmoid_cat":atten_sigmoid}
        )
