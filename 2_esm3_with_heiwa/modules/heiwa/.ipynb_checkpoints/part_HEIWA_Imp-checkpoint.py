import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.heiwa.file_handling import compress, decompress
from modules.heiwa.basic_layers import Swish, SwiGLU_Simplified
from modules.heiwa.Water3P_Implicit import Water3P_Implicit
from modules.heiwa.data_containers import ModelOutput

class HEIWA_Imp(nn.Module):
    def __init__(self, 
                 le_cha_bias_action:float, le_cha_bias_final:torch.Tensor, 
                 num_solv_head:int, num_prot_head:int, 
                 head_label:list, rot_head_idx:list, head_require_rot:list, 
                 num_rot_head:int, 
                 z_dim_to_qk_dim_ratio:float, dim_qk:int,  dim_v:int, 
                 dim_prot:int, 
                 dim_sigmoid_factor: float, 
                 with_scaler:bool=True
                ):
        super().__init__() # dim_w_v:int,
        self.parts = nn.ModuleDict()
        self.le_cha_bias_action = le_cha_bias_action
        self.le_cha_bias_final = le_cha_bias_final
        self.num_prot_head = num_prot_head # must keep, useful in forward
        self.num_solv_head = num_solv_head # must keep, useful in forward
        self.dim_qk = dim_qk # must keep, useful in forward
        self.dim_v = dim_v   # only useful in hp_tuning (dim_v per head)
        self.num_rot_head = num_rot_head # must keep, useful in forward
        self.rot_head_idx = rot_head_idx # must keep, useful in forward
        self.head_label = head_label # must keep, useful in forward
        self.head_require_rot = head_require_rot # must keep, useful in forward
        self.dim_v_relu = self.dim_v - round(dim_v * dim_sigmoid_factor)
        print("dim_sigmoid_factor =", dim_sigmoid_factor)
        print("self.dim_v_relu =", self.dim_v_relu)

        dim_water = round(dim_qk * z_dim_to_qk_dim_ratio)
        self.parts["Water3P_Implicit"] = Water3P_Implicit(
            dim_water=dim_water, dim_qk=dim_qk, dim_v=dim_v, num_head=num_solv_head, 
            with_scaler=with_scaler, dropout_value=0
        ) 
        
        self.parts["Protein"] = nn.ParameterDict()
        self.parts["Protein"].Qp = nn.Linear(dim_prot, dim_qk*num_prot_head*num_solv_head)
        self.parts["Protein"].Qp.pretrained = True
        self.parts["Protein"].Rp = nn.Linear(dim_prot, num_prot_head*num_rot_head*9)
        self.parts["Protein"].Rp.pretrained = True

        self.parts["Protein"].Qp_a = nn.Linear(dim_prot, num_prot_head*num_rot_head*3)
        self.parts["Protein"].Qp_a.pretrained = True
        self.parts["Protein"].Qp_a_bias = nn.Linear(dim_prot, num_prot_head*num_rot_head)

        dim_v_tot = dim_v * num_prot_head * num_solv_head
        
        dim_v_tot_relu = self.dim_v_relu * num_prot_head * num_solv_head
        if dim_v_tot_relu > 0:
            self.rmsn_relu = nn.RMSNorm(dim_v_tot_relu)
            self.rmsn_relu.pretrained = False
        if dim_v_tot - dim_v_tot_relu > 0:
            self.rmsn_sigmoid = nn.RMSNorm(dim_v_tot - dim_v_tot_relu)
            self.rmsn_sigmoid.pretrained = False


    def _load_default(self, num_cpu):
        path = f"heiwa/HEIWA_Imp_epoch_final_model.pth"
        decompress(f"{path}", num_cpu=num_cpu)
        state_dict = torch.load(path, 
                                map_location='cuda', weights_only=True)
        # print(state_dict.keys())
        # quit()
        incompatible = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {incompatible.missing_keys}")
        print(f"Unexpected keys: {incompatible.unexpected_keys}")
        compress(f"{path}", num_cpu=num_cpu)
        
    def product_rotation_attention(self, prot_emb):
        return self.parts["Protein"].Rp(prot_emb)

    def forward(self, x:torch.Tensor, eps:float=1e-6) -> torch.Tensor:
        """Forward pass through the model."""
        # assert ~(torch.isnan(x).any())
        bs, l, d = x.size()

        ## =================== Core attention ===========================================
        kw = self.parts["Water3P_Implicit"].generate_k() # w3_kh, w3_kl, w3_ko

        qp = self.parts["Protein"].Qp(x).reshape(
            bs, l, 
            self.num_prot_head, 
            self.num_solv_head, 
            self.dim_qk
        )
        
        # =========================== Rotation attention: qp_a ==================
        # kwa from implicit embedding
        kw_a = self.parts["Water3P_Implicit"].generate_ka()
        kw_a = F.normalize(kw_a, p=2, dim=-1, eps=1e-12)
        kw_a = kw_a.expand(
            bs, l, 
            self.num_prot_head, self.num_rot_head, 
            3
        )
        
        kw_a_bias = self.parts["Water3P_Implicit"].angle_bias
        

        # # Angle bias of water     
        qp_a_bias = self.parts["Protein"].Qp_a_bias(x).reshape(
            bs, l, 
            self.num_prot_head, 
            self.num_rot_head
        )
       
        # Protein qa
        qp_a = self.parts["Protein"].Qp_a(x).reshape(
            bs, l, 
            self.num_prot_head, 
            self.num_rot_head, 3
        ) 
        qp_a = F.normalize(qp_a, p=2, dim=-1, eps=1e-12)
        
        # Rotation matrix: Rp
        Rp = self.product_rotation_attention(prot_emb=x).reshape(
            bs, l, 
            self.num_prot_head, self.num_rot_head, 
            3, 3
        )

        # Rotation of qp_a
        qp_a = torch.einsum('...ij,...j->...i', Rp, qp_a)
        
        # ============== attention with w3_kh
        atten = (qp * kw).sum(dim=-1)
        atten = atten + torch.where(self.le_cha_bias_action > self.le_cha_bias_final, self.le_cha_bias_action, self.le_cha_bias_final)

        atten_a = (qp_a * kw_a / 3**0.5).sum(dim=-1) * 8 # /self.num_solv_head**0.5
        if self.training:
            atten_a = torch.clamp(atten_a, max=8, min=-8)
        atten = atten * (1 - F.sigmoid(atten_a + kw_a_bias + qp_a_bias))
        
        # ============== Atten_relu & Atten_sigmoid =================================
        atten_relu = F.relu(atten)
        
        if self.training:
            atten_sigmoid = torch.clamp(atten, max=8, min=-8)
            atten_sigmoid = F.sigmoid(atten_sigmoid)
        else:
            atten_sigmoid = F.sigmoid(atten)

        if self.num_solv_head == 3:
            imp_loss = (atten_relu[..., 0] * atten_relu[..., 1]).sum() * min(1, abs(self.le_cha_bias_action)) * 0.01
            intermediate_output = (atten_relu[..., 0], atten_relu[..., 1])
        elif self.num_solv_head == 5:
            imp_loss = ((atten_relu[..., 0] + atten_relu[..., -2]) * (atten_relu[..., 1] + atten_relu[..., -1])).sum() * min(1, abs(self.le_cha_bias_action)) * 0.01
            na_atten = atten_relu[..., -2].reshape(bs, l, -1).sum(-1)
            cl_atten = atten_relu[..., -1].reshape(bs, l, -1).sum(-1)
            intermediate_output = (na_atten, cl_atten)
        
        # ======================= Make V ============================
        vw_imp = self.parts["Water3P_Implicit"].generate_v() # num_solv_head, dim_v
            
        # =========================== Message passing | Do not apply SASA & TIM gate here ===================
        # relu
        if self.dim_v_relu > 0:
            water_modification_relu = atten_relu.reshape(bs, l, self.num_prot_head, self.num_solv_head, 1) * vw_imp[..., :self.dim_v_relu]
            water_modification_relu = water_modification_relu.reshape(bs, l, -1)
            water_modification_relu = self.rmsn_relu(water_modification_relu)
        else:
            water_modification_relu = torch.tensor([], dtype=torch.float32, device="cuda")
        
        # sigmoid
        if self.dim_v_relu < self.dim_v:
            water_modification_sigmoid = atten_sigmoid.reshape(bs, l, self.num_prot_head, self.num_solv_head, 1) * vw_imp[..., self.dim_v_relu:]
            water_modification_sigmoid = water_modification_sigmoid.reshape(bs, l, -1)
            water_modification_sigmoid = self.rmsn_sigmoid(water_modification_sigmoid)
        else:
            water_modification_sigmoid = torch.tensor([], dtype=torch.float32, device="cuda")

        water_modification = torch.cat((water_modification_relu, water_modification_sigmoid), axis=-1).reshape(bs, l, -1)
        # return ModelOutput(y_pred=water_modification, )
        return ModelOutput(y_pred=water_modification)

