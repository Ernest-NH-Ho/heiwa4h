# self_attention.py
import torch
import torch.nn as nn
import math
from modules.model.basic_layers import Swish, SwiGLU_Simplified
from modules.functions.activation import SigmoidWindow

class SelfAttention(nn.Module):
    def __init__(self, dim_in, num_heads, dim_qk, dim_v, expansion_factor, num_ring, 
                 attention_activation='softmax', dropout_value=0.0):
        """
        Self-Attention with Swish and SwiGLU residual connections.
        
        Args:
            dim_in: Input embedding dimension
            num_heads: Number of attention heads
            dim_qk: Dimension per head for queries and keys
            dim_v: Dimension per head for values
            expansion_factor: Expansion factor for SwiGLU
            attention_activation: Activation for attention scores ('softmax', 'sigmoid', 'softplus', 'none')
            dropout_value: Dropout probability for Swish and SwiGLU layers
        """
        super(SelfAttention, self).__init__()
        
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.attention_activation = attention_activation

        # Sigmoid window
        # self.lower_limit = lower_limit
        # self.upper_limit = upper_limit
        self.num_ring = num_ring
        
        # Total dimensions
        self.total_dim_qk = num_heads * dim_qk
        self.total_dim_v = num_heads * dim_v
        
        # Q, K, V projections
        self.W_q = nn.Linear(dim_in, self.total_dim_qk, bias=False)
        self.W_k = nn.Linear(dim_in, self.total_dim_qk, bias=False)
        self.W_v = nn.Linear(dim_in, self.total_dim_v, bias=False)
        
        # Swish layer: projects from (num_heads * dim_v) to dim_in
        self.swish = Swish(in_dim=self.total_dim_v, out_dim=dim_in, dropout_value=dropout_value)
        
        # SwiGLU layer: dim_in to dim_in with internal expansion
        self.swiglu = SwiGLU_Simplified(in_dim=dim_in, expansion_factor=expansion_factor, dropout_value=dropout_value)
        
        # RMSNorm layers
        self.norm_middle = nn.RMSNorm(dim_in)
        self.norm_final = nn.RMSNorm(dim_in)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(dim_qk)
        
    def forward(self, x, m=1, c=None):
        """
        Args:
            x: (batch, seq_len, dim_in)
            
        Returns:
            output: (batch, seq_len, dim_in)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.W_q(x)  # (batch, seq_len, num_heads * dim_qk)
        k = self.W_k(x)  # (batch, seq_len, num_heads * dim_qk)
        v = self.W_v(x)  # (batch, seq_len, num_heads * dim_v)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.dim_qk).transpose(1, 2)  # (batch, num_heads, seq_len, dim_qk)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.dim_qk).transpose(1, 2)  # (batch, num_heads, seq_len, dim_qk)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.dim_v).transpose(1, 2)   # (batch, num_heads, seq_len, dim_v)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, num_heads, seq_len, seq_len)

        # Exclude self Attention   OR    Apply coordinate mask (if this is done, excluding Self Attention is unnecessary)
        if c != None:
            dist = torch.sqrt(((c.unsqueeze(-2) - c.unsqueeze(-3))**2).sum(axis=-1))
            # print("coord", c.size())
            # print(c)
            # print(dist)
            attention_scores = attention_scores * SigmoidWindow.calculate(x=dist, num_ring=self.num_ring) #  lower_limit=self.lower_limit, upper_limit=self.upper_limit
        else:
            attention_scores = attention_scores * (torch.ones(attention_scores.size()).cuda() - 99999999*torch.eye(attention_scores.size(-1)).cuda().unsqueeze(0).unsqueeze(0))

        
        # Apply activation
        if self.attention_activation == 'softmax':
            attention = torch.softmax(attention_scores, dim=-1)
        elif self.attention_activation == 'relu':
            attention = torch.relu(attention_scores, dim=-1)
        elif self.attention_activation == 'sigmoid':
            if self.training:
                attention_scores = torch.clamp(attention_scores, max=8, min=-8)
            attention = torch.sigmoid(attention_scores)
        elif self.attention_activation == 'softplus':
            attention = torch.nn.functional.softplus(attention_scores)
        elif self.attention_activation == 'none' or self.attention_activation is None:
            attention = attention_scores
        else:
            raise ValueError(f"Unknown attention_activation: {self.attention_activation}")

                
        # Compute message
        message = torch.matmul(attention, v)  # (batch, num_heads, seq_len, dim_v)
        
        # Concatenate heads
        message = message.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.total_dim_v)  # (batch, seq_len, num_heads * dim_v)
        
        # First residual: x + Swish(message), then RMSNorm
        middle_output = self.norm_middle(x + self.swish(message) * m.reshape(batch_size, seq_len, 1))  # (batch, seq_len, dim_in)
        
        # Second residual: middle_output + SwiGLU(middle_output), then RMSNorm
        final_output = self.norm_final(x + middle_output + self.swiglu(middle_output) * m.reshape(batch_size, seq_len, 1))  # (batch, seq_len, dim_in)
        
        return final_output # * m.reshape(batch_size, seq_len, 1)
