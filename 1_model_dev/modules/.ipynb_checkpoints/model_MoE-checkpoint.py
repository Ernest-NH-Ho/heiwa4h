import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.model_microlayer import SwiGLU_FFN #, Euclidean

import hashlib

def mix_seeds(seed_list):
    """Hash a list of seeds into one combined 63-bit seed."""
    if not isinstance(seed_list, list):
        raise TypeError("Expected a list of seeds.")
    seed_str = "_".join(map(str, seed_list))
    hashed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
    return hashed % (2**63)  # Torch seeds must fit in 63 bits


class MoE(nn.Module):
    def __init__(self, in_dim, out_dim, num_expert, top_k, dropout, threshold=0.8, expansion_factor=4):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.experts = nn.ModuleList([Expert(in_dim=in_dim, num_hidden=1, hidden_dim=320, 
                                             dropout=dropout)
                                      for _ in range(num_expert)])
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.gate = MoE_Gate(in_dim=in_dim, 
                             threshold=threshold, 
                             num_expert=num_expert, 
                             top_k=top_k) # in_dim, num_expert, top_k, 
        self.top_k = top_k
        self.num_expert = num_expert
        self.check_init()
        
    def check_init(self):
        print(f"\nChecking MoE model.\n")
        print(f"self.num_expert = {self.num_expert}")
        print(f"self.top_k = {self.top_k}")
        
    def add_expert(self):
        self.experts.append(SwiGLU_FFN(in_out_dim=self.in_dim, out_dim=self.out_dim, dropout=self.dropout))
        self.gate.add_one_dim()

    def expert_exam_one_by_one(self, x, gate_scores, all_mask):
        """ Activate the expert models one-followed-by-one """
        """ gate_scores has been softmaxed, so the masked expert will have 0 score now """
        bs, seq_len, _ = gate_scores.shape
        output = torch.zeros(bs, seq_len, device=x.device)
        # print(f"x.size", x.size())
        for i, expert in enumerate(self.experts):
            # print(f"expert {i}")
            weights = gate_scores[..., i]  # shape: [bs, seq_len]
            # print(weights.size())
            if torch.all(weights < 0.):
                continue
    
            mask = (weights >= 0.) & all_mask
            # mask = mask.expand(bs, seq_len, x.size(-1))
            if not mask.any():
                continue
            # print(mask[:3, :3])
            selected_x = x * mask.unsqueeze(-1).expand(bs, seq_len, x.size(-1))  # [N_active, D]
            # print(selected_x.size())
            if torch.isnan(selected_x).any():
                raise ValueError(f"Expert model produced nan via x selection by mask. max = {selected_x.max().item()}. min = {selected_x.min().item()}.")
            expert_out = expert(selected_x).reshape(bs, seq_len) * mask  # [N_active, D]
            if torch.isnan(expert_out).any():
                raise ValueError(f"Expert model produced nan via mask-weigted output. max = {expert_out.max().item()}. min = {expert_out.min().item()}.")
            # print(expert_out[:3, :3])
            # print(weights[:3, :3])
            # print(output[:3, :3])
            # quit()
            # Assign back (accumulate in case multiple experts affect same token)
            output = output + expert_out * weights # [mask] # .unsqueeze(-1 )
            ### weights has 
        # print(output[:3, :3])
        if torch.isnan(output).any():
            raise ValueError(f"Expert model produced nan when synthesizing output. max = {output.max().item()}. min = {output.min().item()}.")

        # Replace 0 with -inf in output
        # output = torch.where(output<=0., float('-inf'), output)
        return output
        
    def forward(self, x, all_mask):
        """disorder_mask: bs, seq_len. Bad token will have 0, good tokens will have 1"""
        # print(f"x, {x.device}")
        # print(f"all_mask, {all_mask.device}")
        x = self.norm(x)
        gate_scores = self.gate.check_k(x) # bs, seq_len, num_expert
        # print(gate_scores.size())
        max_gate_scores = gate_scores.max(dim=-1).values
        # print(max_gate_scores.size())
        # need_new_expert = ((max_gate_scores < self.gate.threshold).any() & (all_mask.bool()))
        # if need_new_expert.any() and self.gate.noise_level == 0:
        #     # 1. ADD ONE expert
        #     self.add_expert()
        #     # 2. Expand gate_scores by adding one more expert logit, set to threshold value
        #     gate_scores = self.gate.extrapolate_new_dim(gate_scores)
            
        # this produce the final gate_scores after masking unsuitable expert & softmax
        gate_scores = self.gate(gate_scores)
        # print(gate_scores[:3, :3])
        # print(torch.isnan(gate_scores).sum().item())
        if torch.isnan(gate_scores).any():
            raise ValueError(f"Gating layer outputing nan values after softmax. max = {gate_scores.max().item()}. min = {gate_scores.min().item()}.")
        # The 2nd term for return is to report expert usage
        return self.expert_exam_one_by_one(x, gate_scores, all_mask), ((gate_scores > 0.) & (all_mask.unsqueeze(-1))).sum(dim=1)
        

class Expert(nn.Module):
    def __init__(self, in_dim, num_hidden, hidden_dim, dropout, expansion_factor=4, out_dim=1):
        super(Expert, self).__init__()
        self.dropout0 = nn.Dropout(dropout)
        self.swiglu_ffn = SwiGLU_FFN(in_out_dim=in_dim, expansion_factor=expansion_factor, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.linear.gain = 0.5
        self.linear.orthogonal_loss = 'Single_W'
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 1)
        # self.linear1.Bias = 0.1

    def forward(self, x):
        x = self.swiglu_ffn(self.dropout0(x))
        x = self.norm1(self.dropout1(x))
        # print("a", x[:3, :3])
        x = self.linear(x)
        # print("b", x[:3, :3])
        x = self.linear1(self.norm2(x))
        # print("c", x[:3, :3])
        # F.relu()
        return x

class MoE_Gate(nn.Module):
    def __init__(self, in_dim, num_expert, top_k, initial_noise=0.4, threshold=0.8):
        super().__init__()
        self.in_dim = in_dim
        self.threshold = threshold
        self.num_expert = num_expert
        self.top_k = top_k
        # self.noise_level = initial_noise
        self.bias_max = 0.5
        self.bias_min = -0.35
        self.linear = nn.Linear(in_dim, num_expert)
        self.linear.gain = 0.5
        self.linear.orthogonal_loss = False
        self.norm = nn.LayerNorm(in_dim)
        # self.seed_list = [45, 42, 123, 456, 789, 1024, 1, 6, 7, 100, 4564324, 512, 888, 960, 8123, 11, 64, 5555]
        # self.make_generator()
        # self.seed_add_one()
        self.check_init()

    def check_init(self):
        print(f"\nChecking gating network.\n")
        print(f"self.num_expert = {self.num_expert}")
        print(f"self.top_k = {self.top_k}")

    def change_bias(self, i, change:float):
        with torch.no_grad():    # no gradients for init
            if change > 0:
                self.linear.bias[i] = torch.where(self.linear.bias[i] + change >= self.bias_max, self.bias_max, self.linear.bias[i] + change)      # (n_experts,)
            elif change < 0:
                self.linear.bias[i] = torch.where(self.linear.bias[i] + change <= self.bias_min, self.bias_min, self.linear.bias[i] + change) 
            # print(f"Bias of expert index {i} changed to {self.linear.bias[i]}.")

    def multiply_bias(self, factor):
        with torch.no_grad():
            self.linear.bias.mul_(factor)
    # def seed_add_one(self):
    #     self.seed_list = [rs+1 for rs in self.seed_list]
    #     self.make_generator()
        
    # def make_generator(self):
    #     mixed_seed = mix_seeds(self.seed_list)
    #     # g = torch.Generator().manual_seed(mixed_seed)
    #     self.generator = torch.Generator(device='cuda')
    #     self.generator.manual_seed(mixed_seed)

    # def add_one_dim(self):
    #     # Define a new linear layer
    #     new_linear = nn.Linear(self.in_dim, self.linear.out_features + 1).to(self.linear.device)
    #     nn.init.kaiming_uniform_(new_linear.weight, a=math.sqrt(5))  # init new weights
    # 
        # # Copy old weights
        # new_linear.weight.data[:-1] = self.linear.weight.data
        # new_linear.bias.data[:-1] = self.linear.bias.data
    
        # # Replace gate linear layer
        # self.linear = new_linear
        # self.num_expert += 1

        # # Set noise level to intermediate level
        # self.noise_level = 0.3

    # def extrapolate_new_dim(self, gate_scores):
    #     bs, seq_len, old_expert_count = gate_scores.shape
    #     new_expert_count = old_expert_count + 1
    
    #     new_gate_scores = gate_scores.new_zeros(bs, seq_len, new_expert_count)
    #     new_gate_scores[..., :-1] = gate_scores  # copy old scores
    
    #     # Set logit of the new expert to 0.8 for tokens with no valid expert
    #     new_gate_scores[no_expert_mask, -1] = self.gate.threshold  # typically 0.8
    
    #     return new_gate_scores  # now ready for masking + softmax
    
    def mask_logits(self, gate_scores):
        # topk_vals, topk_idx = gate_scores.topk(k, dim=-1)
        # mask = torch.zeros_like(gate_scores, dtype=torch.bool)
        # mask.scatter_(dim=-1, index=topk_idx, value=True)
        # gate_scores = gate_scores.masked_fill(~mask, float('-inf'))
        
        # gate_scores: [bs, seq_len, num_expert]
        # print(gate_scores.size())
        # print(gate_scores[:3, :3])
        # if self.top_k == 1:
        B, T, E = gate_scores.shape
        device  = gate_scores.device
    
        # 1) Base mask: experts that beat the threshold
        base_mask = gate_scores > self.threshold                 # bool [B,T,E]
    
        # 2) Always keep the top-k scoring experts (argmax if k=1)
        #    This is the fallback when nothing beats the threshold.
        topk_idx = gate_scores.topk(k=self.top_k, dim=-1).indices   # [B,T,k]
        fallback_mask = torch.zeros_like(base_mask, dtype=torch.bool)
        fallback_mask.scatter_(-1, topk_idx, True)             # set top-k True
    
        # 3) Combine: threshold winners  OR  mandatory top-k
        final_mask = base_mask | fallback_mask                 # [B,T,E]
    
        # 4) Mask logits – every row now has ≥1 finite entry
        gated_logits = gate_scores.masked_fill(~final_mask, float('-inf'))
        # print(gated_logits[:3, :3])
        return gated_logits
            
        # else:
        #     threshold_mask = gate_scores > self.threshold  # bs, seq_len, num_expert
        #     # print(threshold_mask.size())
        #     # print(threshold_mask[:3, :3])
        #     num_passing = threshold_mask.sum(dim=-1)  # [bs, seq_len]
        #     # print(num_passing.size())
        #     # print(num_passing[:3, :3])
        #     _, topk_idx = gate_scores.topk(self.top_k, dim=-1)  # [bs, seq_len, k] <- ranking
        #     # print(topk_idx.size())
        #     # print(topk_idx[:3, :3])
        #     # Build mask based on conditions
        #     mask = []
        #     # print(mask[:3, :3])
        #     for k in range(1, self.top_k + 1):
        #         mask.append(torch.zeros_like(gate_scores, dtype=torch.bool))
        #         if k == 1:
        #             active_mask = num_passing <= k
        #         else:
        #             active_mask = num_passing == k
        #         idx = topk_idx[..., :k]  # top-1 or top-2 based on condition
        #         # print(idx.dtype)
        #         # print(active_mask[:3, :3])
        #         mask[-1].scatter_(dim=-1, index=idx, src=active_mask.unsqueeze(-1).expand(-1, -1, self.num_expert))
        #     mask = torch.stack(mask, dim=0).any(dim=0)
        #     # mask |= (num_passing >= self.top_k).unsqueeze(-1) & threshold_mask  # default top-2 if >2 experts pass
        #     # Set others to -inf
        #     gated_logits = gate_scores.masked_fill(~mask, float('-inf'))
        #     return gated_logits
        

    def check_k(self, x):
        # print(f"x: {x.device}")
        return self.linear(self.norm(x)) # + self.noise_level * torch.randn((self.num_expert), device='cuda', generator=self.generator)
        
    def forward(self, gates):
        
        gates = self.mask_logits(gates)
        # print(gates.size())
        # print(gates[:3, :3])
        if torch.isnan(gates).any():
            raise ValueError(f"Raw gates values before softmax has nan. gates.max() = {gates.max().item()}. gates.min() = {gates.min().item()}.")
        return F.softmax(gates, dim=-1)


        