# cross_attention.py
import torch
import torch.nn as nn
import math
from modules.model.basic_layers import Swish, SwiGLU_Simplified
from modules.model.dropout_control import Dropout_tuner_double_bounded


class CrossAttention(nn.Module):
    def __init__(self, dim_x1, dim_x2, num_heads, dim_qk, dim_v, expansion_factor, 
                 attention_activation='softmax', attention_map_only=False, dropout_value=0.0):
        """
        Cross-Attention between two inputs x1 and x2.
        
        Args:
            dim_x1: Dimension of first input (query source)
            dim_x2: Dimension of second input (key/value source)
            num_heads: Number of attention heads
            dim_qk: Dimension per head for queries and keys
            dim_v: Dimension per head for values
            expansion_factor: Expansion factor for SwiGLU
            attention_activation: Activation for attention scores ('softmax', 'sigmoid', 'softplus', 'none')
            attention_map_only: If True, only compute Q and K, return attention map
            dropout_value: Dropout probability for Swish and SwiGLU layers
        """
        super(CrossAttention, self).__init__()
        
        self.dim_x1 = dim_x1
        self.dim_x2 = dim_x2
        self.num_heads = num_heads
        self.dim_qk = dim_qk
        self.attention_activation = attention_activation
        self.attention_map_only = attention_map_only
        
        # Total dimensions
        self.total_dim_qk = num_heads * dim_qk
        
        # Q from x1, K (and V) from x2
        self.W_q = nn.Linear(dim_x1, self.total_dim_qk, bias=False)
        self.W_k = nn.Linear(dim_x2, self.total_dim_qk, bias=False)
        
        self.q_dropout = nn.Dropout(dropout_value)
        self.q_dropout.tuner = Dropout_tuner_double_bounded(
            metric=dropout_value, metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.025, thres_Q1=1.015, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='max'
        )
        self.k_dropout = nn.Dropout(dropout_value)
        self.k_dropout.tuner = Dropout_tuner_double_bounded(
            metric=dropout_value, metric_upper_limit=1, metric_lower_limit=0, \
            thres_Q2=1.025, thres_Q1=1.015, max_step=0.25, epoch_scale=10, \
            eval_interval=4, mode='max'
        )

        # self.ln = nn.LayerNorm([num_heads, 1, 1])  # only here becauase i output the attention map after sigmoid
        
        if not attention_map_only:
            # Calculate total v dim
            self.dim_v = dim_v
            self.total_dim_v = num_heads * dim_v
            
            # V projection and residual layers only if we need full attention
            self.W_v = nn.Linear(dim_x2, self.total_dim_v, bias=False)
            
            # Swish layer: projects from (num_heads * dim_v) to dim_x1
            self.swish = Swish(in_dim=self.total_dim_v, out_dim=dim_x1, dropout_value=dropout_value)
            
            # SwiGLU layer: dim_x1 to dim_x1 with internal expansion
            self.swiglu = SwiGLU_Simplified(in_dim=dim_x1, expansion_factor=expansion_factor, dropout_value=dropout_value)
            
            # RMSNorm layers
            # self.norm_middle = nn.RMSNorm(dim_x1)
            # self.norm_final = nn.RMSNorm(dim_x1)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(dim_qk)
        
    def forward(self, x1, x2, m1, m2):
        """
        Args:
            x1: (batch, seq_len1, dim_x1) - query source
            x2: (batch, seq_len2, dim_x2) - key/value source
            
        Returns:
            If attention_map_only: attention_map (batch, num_heads, seq_len1, seq_len2)
            Otherwise: output (batch, seq_len1, dim_x1)
        """
        batch_size, seq_len1, _ = x1.shape
        seq_len2 = x2.shape[1]

        # Project to Q and K
        q = self.q_dropout(self.W_q(x1))  # (batch, seq_len1, num_heads * dim_qk)
        k = self.k_dropout(self.W_k(x2))  # (batch, seq_len2, num_heads * dim_qk)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len1, self.num_heads, self.dim_qk).transpose(1, 2)  # (batch, num_heads, seq_len1, dim_qk)
        k = k.reshape(batch_size, seq_len2, self.num_heads, self.dim_qk).transpose(1, 2)  # (batch, num_heads, seq_len2, dim_qk)
        
        # Compute attention scores
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, num_heads, seq_len1, seq_len2)

        # If attention_map_only, return the attention map
        if self.attention_map_only:
            attention = attention * m1.reshape(batch_size, 1, seq_len1, 1) * m2.reshape(batch_size, 1, 1, seq_len2)
            # attention = self.ln(attention) * 3
            attention = attention.sum(axis=1)
            attention = attention / attention.std()
            if self.training:
                attention = torch.clamp(attention, max=8, min=-8)
            attention = torch.sigmoid(attention)
            return attention
            
        # Apply activation
        if self.attention_activation == 'softmax':
            attention = torch.softmax(attention, dim=-1)
        elif self.attention_activation == 'sigmoid':
            if self.training:
                attention = torch.clamp(attention, max=8, min=-8)
            attention = torch.sigmoid(attention)
        elif self.attention_activation == 'softplus':
            attention = torch.nn.functional.softplus(attention)
        elif self.attention_activation == 'none' or self.attention_activation is None:
            attention = attention
        else:
            raise ValueError(f"Unknown attention_activation: {self.attention_activation}")

       
        # Otherwise, compute full attention with residuals
        v = self.W_v(x2)  # (batch, seq_len2, num_heads * dim_v)
        v = v.reshape(batch_size, seq_len2, self.num_heads, self.dim_v).transpose(1, 2)  # (batch, num_heads, seq_len2, dim_v)
        
        # Compute message
        message = torch.matmul(attention, v)  # (batch, num_heads, seq_len1, dim_v)
        
        # Concatenate heads
        message = message.transpose(1, 2).contiguous().reshape(batch_size, seq_len1, self.total_dim_v)  # (batch, seq_len1, num_heads * dim_v)
        
        # First residual: x1 + Swish(message), then RMSNorm
        middle_output = self.norm_middle(x1 + self.swish(message))  # (batch, seq_len1, dim_x1)
        
        # Second residual: middle_output + SwiGLU(middle_output), then RMSNorm
        final_output = self.norm_final(middle_output + self.swiglu(middle_output))  # (batch, seq_len1, dim_x1)
        
        return final_output