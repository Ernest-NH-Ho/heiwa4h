import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from modules.model.basic_layers import Swish, SwiGLU_Simplified # SwiGLU_2input, 
from modules.model.dropout_control import Dropout_tuner_double_bounded
from modules.model.Water3P_Implicit_NaCl import Water3P_Implicit_Salt
from modules.data_containers import ModelOutput

class HEIWA_3P_Salt(nn.Module):
    def __init__(self, le_cha_bias:float, # exp_mid_dim:int, 
                 num_explicit_head:int, num_prot_head:int, 
                 salt_conc:dict, head_label:list, rot_head_idx:list, head_require_rot:list, 
                 num_rot_head:int, 
                 z_dim_to_qk_dim_ratio:float, dim_qk:int,  dim_v:int, 
                 dim_prot:int, v_compress:float, out_expansion_factor:float, 
                 dim_sigmoid_factor: float, 
                 dropout:dict, with_scaler:bool=True):
        super().__init__() # dim_w_v:int,
        self.parts = nn.ModuleDict()
        self.le_cha_bias = le_cha_bias
        self.num_prot_head = num_prot_head # must keep, useful in forward
        self.num_explicit_head = num_explicit_head # must keep, useful in forward
        self.dim_qk = dim_qk # must keep, useful in forward
        self.dim_v = dim_v   # only useful in hp_tuning (dim_v per head)
        self.num_rot_head = num_rot_head # must keep, useful in forward
        self.rot_head_idx = rot_head_idx # must keep, useful in forward
        self.head_label = head_label # must keep, useful in forward
        self.head_require_rot = head_require_rot # must keep, useful in forward
        self.salt_conc = salt_conc # must keep, useful in forward
        self.v_compress = v_compress
        self.dim_v_relu = self.dim_v - round(dim_v * dim_sigmoid_factor)
        print("dim_sigmoid_factor =", dim_sigmoid_factor)
        print("self.dim_v_relu =", self.dim_v_relu)

        # self.parts["Explicit"] = ExpImp_Hybridizer(in_dim=dim_prot, mid_dim=exp_mid_dim)
        
        dim_water = round(dim_qk * z_dim_to_qk_dim_ratio)
        self.parts["Water3P_Implicit_Salt"] = Water3P_Implicit_Salt(dim_water=dim_water, dim_qk=dim_qk, dim_v=dim_v, num_head=num_explicit_head, 
                                                                    with_scaler=with_scaler, dropout_value=dropout["low"]) # dim_v=dim_w_v, dim_v_se3=dim_v_se3, 
        self.parts["Protein"] = nn.ParameterDict()
        self.parts["Protein"].Qp = nn.Linear(dim_prot, dim_qk*num_prot_head*num_explicit_head)
        self.parts["Protein"].Qp.pretrained = True
        self.parts["Protein"].Rp = nn.Linear(dim_prot, num_prot_head*num_rot_head*9)
        self.parts["Protein"].Rp.pretrained = True

        self.parts["Protein"].Qp_a = nn.Linear(dim_prot, num_prot_head*num_rot_head*3)
        self.parts["Protein"].Qp_a.pretrained = True
        self.parts["Protein"].Qp_a_bias = nn.Linear(dim_prot, num_prot_head*num_rot_head)

        # This is explicit water
        # self.parts["Protein"].Kw_a = nn.Linear(dim_prot, num_prot_head*num_rot_head*3)
        # self.parts["Protein"].Kw_a.pretrained = True

        dim_v_tot = dim_v * num_prot_head * num_explicit_head
        dim_o = round(dim_v_tot*v_compress)
        
        self.Wo_w = Swish(in_dim=dim_v_tot, out_dim=dim_o, dropout_value=dropout["low"]*(dim_v/1600))
        self.Wo_w.pretrained = False

        # Only for Explicit_Implicit mixer
        # self.HEIWA_mixer = nn.Linear(dim_prot, num_prot_head*num_explicit_head)
        
        # if not with_scaler: # manually change this for HEIWA_mixer
        # self.Vw = Swish(in_dim=dim_prot, out_dim=dim_v_tot, dropout_value=dropout['low']*0.5)
        
        dim_v_tot_relu = self.dim_v_relu * num_prot_head * num_explicit_head
        if dim_v_tot_relu > 0:
            self.rmsn_relu = nn.RMSNorm(dim_v_tot_relu)
            self.rmsn_relu.pretrained = False
        if dim_v_tot - dim_v_tot_relu > 0:
            self.rmsn_sigmoid = nn.RMSNorm(dim_v_tot - dim_v_tot_relu)
            self.rmsn_sigmoid.pretrained = False

        # Prot Gate vs Self gate
        self.out_proj = SwiGLU_Simplified(in_dim=dim_o, out_dim=dim_prot, dropout_value=dropout["low"]*out_expansion_factor/6, 
                                          expansion_factor=out_expansion_factor)
        # self.out_proj = SwiGLU_2input(in_dim_ffn=dim_o, in_dim_gate=dim_prot, out_dim=dim_prot, dropout_value=dropout["low"]*1.25*out_expansion_factor, 
        #                               expansion_factor=out_expansion_factor)
        self.out_proj.pretrained = False

        # self._load_default()

        # ==== Dropout ====
        # self.Vw_dropout = nn.Dropout(dropout['low']*0.5)
        # self.Vw_dropout.tuner = Dropout_tuner_double_bounded(
        #     metric=dropout['low']*0.5, metric_upper_limit=1, metric_lower_limit=0, \
        #     thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
        #     eval_interval=4, mode='min'
        # )
        
        self.parts["Protein"].dropout_Qp = nn.Dropout(dropout['low']*0.5)
        self.parts["Protein"].dropout_Qp.tuner = Dropout_tuner_double_bounded(
            metric=dropout['low'], metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='min'
        )
        self.parts["Protein"].dropout_Qp_a_bias = nn.Dropout(dropout['low']*0.01)
        self.parts["Protein"].dropout_Qp_a_bias.tuner = Dropout_tuner_double_bounded(
            metric=dropout['low']*0.01, metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='min'
        )
        # self.parts["Protein"].dropout_Kw_a_bias = nn.Dropout(dropout['low']*0.01)
        # self.parts["Protein"].dropout_Kw_a_bias.tuner = Dropout_tuner_double_bounded(
        #     metric=dropout['low']*0.01, metric_upper_limit=1, metric_lower_limit=0, \
        #     thres_Q2=1.075, thres_Q1=1.045, max_step=0.25, epoch_scale=10, \
        #     eval_interval=4, mode='min'
        # )

    def _load_default(self):
        # print(os.path.exists(f"./models/HEIWA3P_Atten/esm3_bs16_dr1_3_lr3e-05_qk128_w3dzfact{z_dim_to_qk_dim_ratio}_hd3/HEISA_3P_final_model.pth"))
        state_dict = torch.load(f"./models/data_20251228/HEIWA3P_Atten/esm3_bs16_dr1_3_lr3e-05_qk128_w3dzfact5.0_hd3/HEISA_3P_final_model_NaCl.pth", 
                                map_location='cuda', weights_only=True)
        # state_dict = torch.load(f"./models/HEIWA3P/esm3_bs8_dr1_3_v96_v_compress0.6_w3dzfact5.0_expd4.0/HEIWA_3P_epoch_15_model.pth", 
        #                         map_location='cuda', weights_only=True)
        self.parts["Water3P_Implicit_Salt"].load_state_dict(state_dict, strict=False)
        # pass
        
    def product_rotation_attention(self, prot_emb):
        return self.parts["Protein"].Rp(prot_emb)

    def forward(self, x:torch.Tensor, tim_gate: torch.Tensor,  sasa_gate: torch.Tensor, exp_imp_hybridizer: torch.Tensor|int=0, eps:float=1e-6) -> torch.Tensor:
        """Forward pass through the model."""
        # assert ~(torch.isnan(x).any())
        bs, l, d = x.size()

        # ======== HEIWA_mixer ==========
        # HEIWA_mixer = self.HEIWA_mixer(x).reshape(bs, l, self.num_prot_head, self.num_explicit_head, 1)
        # if self.training:
        #     HEIWA_mixer = torch.clamp(HEIWA_mixer, max=8, min=-8)
        # HEIWA_mixer = F.sigmoid(HEIWA_mixer)
        if isinstance(exp_imp_hybridizer, torch.Tensor):
            exp_imp_hybridizer = exp_imp_hybridizer.reshape(bs, l, 1, 1, 1)
        
        ## =================== Core attention ===========================================
        w3_k = self.parts["Water3P_Implicit_Salt"].generate_k() # w3_kh, w3_kl, w3_ko
        # assert ~(torch.isnan(w3_k).any())
        # assert ~(torch.isinf(w3_k).any())
        # assert w3_k.abs().max() < 1e2
        # print(w3_k.size()) # (explicit_head, 128)
        # quit()

        prot_q = self.parts["Protein"].Qp(self.parts["Protein"].dropout_Qp(x)).reshape(
            bs, l, 
            self.num_prot_head, 
            self.num_explicit_head, 
            self.dim_qk
        )
        # assert ~(torch.isnan(prot_q).any())
        # assert ~(torch.isinf(prot_q).any())
        # assert prot_q.abs().max() <= 1e2
        # print(prot_q.size())  # bs, l, prot_hd, explicit_hd, 128
        # quit()
        
        # =========================== Rotation attention: prot_qa ==================
        # kwa from explicit embedding
        # water_ka_exp = self.parts["Protein"].Kw_a(x).reshape(bs, l, 
        #                                                 self.num_prot_head, 
        #                                                 self.num_rot_head, 3) # self.parts["Protein"].dropout_Qp_a()
        # water_ka_exp = F.normalize(water_ka_exp, p=2, dim=-1, eps=1e-12)
        # assert ~(torch.isnan(water_ka_exp).any())
        # print(water_ka.size()) # bs, l, prot_hd, explicit_hd, 3
        
        # kwa from implicit embedding
        w3_ka_imp = self.parts["Water3P_Implicit_Salt"].generate_ka()
        w3_ka_imp = F.normalize(w3_ka_imp, p=2, dim=-1, eps=1e-12)
        # assert ~(torch.isnan(w3_ka_imp).any())
        # assert ~(w3_ka_imp==0).all()

        # ******* heiwa_mixing
        w3_ka_imp = w3_ka_imp.expand(bs, l, self.num_prot_head, self.num_rot_head, 3)
        water_ka = w3_ka_imp # exp_imp_hybridizer * water_ka_exp + (1-exp_imp_hybridizer) * w3_ka_imp
        
        
        water_angle_bias = self.parts["Water3P_Implicit_Salt"].angle_bias
        

        # # Angle bias of water     
        prot_angle_bias = self.parts["Protein"].Qp_a_bias(self.parts["Protein"].dropout_Qp_a_bias(x)).reshape(
            bs, l, 
            self.num_prot_head, 
            self.num_rot_head
        )
        # assert ~(torch.isnan(prot_angle_bias).any())
        # print(prot_angle_bias.size()) # bs, l, prot_hd, explicit_hd
        # quit()

        
        # Protein qa
        # Protein qa
        prot_qa = self.parts["Protein"].Qp_a(x).reshape(bs, l, 
                                                        self.num_prot_head, 
                                                        self.num_rot_head, 3) # self.parts["Protein"].dropout_Qp_a()
        prot_qa = F.normalize(prot_qa, p=2, dim=-1, eps=1e-12)
        # assert ~(torch.isnan(prot_qa).any())
        # print(prot_qa.size()) # bs, l, prot_hd, explicit_hd, 3
        # quit()
        
        # Rotation matrix: Rp
        Rp = self.product_rotation_attention(prot_emb=x).reshape(
            bs, l, self.num_prot_head, self.num_rot_head, 3, 3
        )
        # assert ~(torch.isnan(Rp).any())
        # assert ~(Rp==0).all()
        Rp = F.normalize(Rp, p=2, dim=-2, eps=1e-12)

        # Rotation of prot_qa
        prot_qa = torch.einsum('...ij,...j->...i', Rp, prot_qa)
        # assert ~(torch.isnan(prot_qa).any())
        # assert ~(torch.isnan(prot_angle_bias).any())
        
        # ============== attention with w3_kh
        atten = torch.zeros((bs, l, self.num_prot_head, self.num_explicit_head)).float().cuda()
        # print(atten.size())   # bs, l, prot_hd, explicit_hd
        # quit()
        
        # print("head_require_rot:", self.head_require_rot)
        # print("rot_head_idx:", self.rot_head_idx)
        if self.num_rot_head > 0:
            atten_a = []
        for i in range(self.num_explicit_head):
            # Set Le Chaterlier's bias
            final_bias = math.log(self.salt_conc[self.head_label[i]])/2
            if final_bias > 0:
                factor = 1
            else:
                factor = -1
                
            if self.rot_head_idx[i] != None:
                i_rot = self.rot_head_idx[i]
                # If water angle bias is from implicit embedding
                # atten_a.append((prot_qa[..., i, :] * water_ka[..., i_rot, :] / 3**0.5).sum(dim=-1) * 8/self.num_rot_head**0.5 + \
                # self.parts["Water3P_Implicit_Salt"].angle_bias[i_rot] + prot_angle_bias[..., i_rot])
                # If water angle bias is from SE3 protein embedding
                atten_a.append((prot_qa[..., i_rot, :] * water_ka[..., i_rot, :] / 3**0.5).sum(dim=-1) * 8/self.num_rot_head**0.5 + \
                water_angle_bias[..., i_rot] + prot_angle_bias[..., i_rot])
                # print(f"atten_a[{i_rot}]", atten_a[..., i_rot])
                # assert ~(torch.isnan(atten_a[i_rot]).any())
                
                # if self.training:
                #     atten_a[..., i_rot] = torch.clamp(atten_a[..., i_rot], min=-8, max=8)
                atten_a[i_rot] = 1 - F.sigmoid(atten_a[i_rot])
                # assert ~(torch.isnan(atten_a[i_rot]).any())
                # assert ~(torch.isnan(water_ka[..., i_rot, :]).any())
                # assert ~(torch.isinf(atten_a[i_rot]).any())
                # assert ~(torch.isinf(water_ka[..., i_rot, :]).any())

                # fianl attention score = (q*k + LeChatBias) * atten_a
                atten[..., i] = ((prot_q[..., i, :] * w3_k[..., i, :] / self.dim_qk**0.5).sum(dim=-1) + max(final_bias, factor * self.le_cha_bias)) * atten_a[i_rot] # bs, l, 3, head, dim_qk
            else:
                atten[..., i] = ((prot_q[..., i, :] * w3_k[..., i, :] / self.dim_qk**0.5).sum(dim=-1) + max(final_bias, factor * self.le_cha_bias))
            # assert ~(torch.isinf(atten[..., i]).any())
            # assert ~(torch.isnan(atten[..., i]).any())
            # assert atten[..., i].abs().max() <= 1e2
            # print("atten", atten[..., i].abs().max().item(), atten[..., i])
        # atten = self.dropout_atten(atten)

        # ===== We only have 4 outputs: pre-train atten ====
        # # Combine L and O head
        # print(atten.size())
        # atten = torch.cat((atten[:, :, 0, ...].reshape(bs, l, 1, -1), atten[:, :, 1:, ...].sum(dim=2).reshape(bs, l, 1, -1)), dim=2)
        # # print(atten.size())
        # # # # Combine multiple heads dedicated to +/-ve prots
        # atten = torch.cat((atten[..., :-1].sum(dim=-1).reshape(bs, l, -1, 1), atten[..., -1].reshape(bs, l, -1, 1)), dim=-1)
        # # print(atten.size())
        # atten = atten.reshape(bs, l, 4) * 8

        
        # ============== Apply SASA gate and TIM gate to Atten ->  Atten_relu & Atten_sigmoid =================================
        atten_relu = F.relu(atten)
        
        if self.training:
            atten_sigmoid = torch.clamp(atten, max=8, min=-8)
            atten_sigmoid = F.sigmoid(atten_sigmoid)
        else:
            atten_sigmoid = F.sigmoid(atten)
            
        nacl_loss = (atten_sigmoid[..., -1] * atten_sigmoid[..., -2]).mean(axis=1).sum() * min(1, abs(self.le_cha_bias))
        h2o_loss = (atten_sigmoid[..., 0] * atten_sigmoid[..., 1]).mean(axis=1).sum() * min(1, abs(self.le_cha_bias)) * 0.25
        hna_loss = (atten_sigmoid[..., 0] - atten_sigmoid[..., -2]).mean(axis=1).sum() * min(1, abs(self.le_cha_bias)) * 0.01
        # Message passing with V
        # atten = atten.reshape()
        
        # ===================== Make V_se3 ===========================
        # w3_v = self.parts["Water3P_Implicit"].generate_v_se3()
        # Rotation
        # for i in range(3):
        #     w3_v_se3[i] = Rw[:, :, i, ...] @ w3_v_se3[i]
        # w3_v_h_tot, w3_v_l_tot, w3_v_o_tot = self.parts["Water3P_Implicit"].flatten_v_se3(w3_v_se3)

        # ======================= Make V ============================
        # if self.parts["Water3P_Implicit_Salt"].with_scaler:
            # w3_v = self.parts["Water3P_Implicit_Salt"].generate_v() # num_explicit_head, dim_v
        # else:
            # w3_v = self.Vw(x).reshape(bs, l, self.num_prot_head, self.num_explicit_head, self.dim_v) # num_explicit_head, dim_v
        w3_v_imp = self.parts["Water3P_Implicit_Salt"].generate_v() # num_explicit_head, dim_v
        # w3_v_exp = self.Vw(x).reshape(bs, l, self.num_prot_head, self.num_explicit_head, self.dim_v) # num_explicit_head, dim_v
        w3_v = w3_v_imp # exp_imp_hybridizer * w3_v_exp + (1-exp_imp_hybridizer) * w3_v_imp
            
        # =========================== Message passing | Do not apply SASA & TIM gate here ===================
        # relu
        if self.dim_v_relu > 0:
            water_modification_relu = self.rmsn_relu((atten_relu.reshape(bs, l, self.num_prot_head, self.num_explicit_head, 1) * w3_v[..., :self.dim_v_relu]).reshape(bs, l, -1))
        else:
            water_modification_relu = torch.tensor([], dtype=torch.float32, device="cuda")
        # print(water_modification_relu.size())
        # sigmoid
        if self.dim_v_relu < self.dim_v:
            water_modification_sigmoid = self.rmsn_sigmoid((atten_sigmoid.reshape(bs, l, self.num_prot_head, self.num_explicit_head, 1) * w3_v[..., self.dim_v_relu:]).reshape(bs, l, -1))
        else:
            water_modification_sigmoid = torch.tensor([], dtype=torch.float32, device="cuda")
        # print(water_modification_relu.size()) # bs, l, v_dim_tot * %relu
        # print(water_modification_sigmoid.size()) # bs, l, v_dim_tot * %sigmoid
        
        # print(water_modification_sigmoid.size())
        # combine + flatten
        water_modification = self.Wo_w(torch.cat((water_modification_relu, water_modification_sigmoid), axis=-1))
        # print(water_modification.size()) # bs, l, v_dim_tot * v_compress
        # print(water_modification.size())
        y_pred = self.out_proj(water_modification) * (1-tim_gate.reshape(bs, l, 1) * sasa_gate.reshape(bs, l, 1)) # , dim=-1
        # print(y_pred.size()) # bs, l, d
        # y_pred = self.out_proj(x_ffn=water_modification, x_gate=x)
        return ModelOutput(y_pred=y_pred, intra_model_loss=nacl_loss + h2o_loss + hna_loss)

