# self_attention_geometric.py
import torch
import torch.nn as nn
import math
from modules.model.basic_layers import Swish, SwiGLU_Simplified
from modules.functions.activation import SigmoidWindow

class SelfAttentionGeometric(nn.Module):
    def __init__(self, dim_in, num_heads_T, num_heads_R, 
                 dim_qk_T_position, dim_qk_T_direction, dim_v_T,
                 dim_qk_R, dim_v_R, 
                 expansion_factor, num_ring, 
                 attention_activation='softmax', dropout_value=0.0, verbose=False):
        """
        Geometric Self-Attention with local coordinate frame transformations.
        
        Args:
            dim_in: Input embedding dimension
            num_heads_T: Number of transformation heads (rotation + translation)
            num_heads_R: Number of rotation-only heads
            dim_qk_T_position: Position-like q/k dimension per T-head (in logical units, will be *3)
            dim_qk_T_direction: Direction-like q/k dimension per T-head (in logical units, will be *3)
            dim_v_T: Value dimension per T-head (in logical units, will be *3)
            dim_qk_R: q/k dimension per R-head (in logical units, will be *3)
            dim_v_R: Value dimension per R-head (in logical units, will be *3)
            expansion_factor: Expansion factor for SwiGLU
            num_ring: Number of rings for SigmoidWindow
            attention_activation: Activation for attention scores ('softmax', 'sigmoid', etc.)
            dropout_value: Dropout probability
            verbose: If True, print tensor shapes during forward pass
        """
        super(SelfAttentionGeometric, self).__init__()
        
        self.dim_in = dim_in
        self.num_heads_T = num_heads_T
        self.num_heads_R = num_heads_R
        self.num_heads = num_heads_T + num_heads_R
        
        # Logical dimensions (in units, will be multiplied by 3 for actual 3D vectors)
        self.dim_qk_T_position = dim_qk_T_position
        self.dim_qk_T_direction = dim_qk_T_direction
        self.dim_v_T = dim_v_T
        self.dim_qk_R = dim_qk_R
        self.dim_v_R = dim_v_R
        
        self.attention_activation = attention_activation
        self.num_ring = num_ring
        self.verbose = verbose
        
        # Actual dimensions (in 3D space)
        self.actual_dim_qk_T = (dim_qk_T_position + dim_qk_T_direction) * 3
        self.actual_dim_v_T = dim_v_T * 3
        self.actual_dim_qk_R = dim_qk_R * 3
        self.actual_dim_v_R = dim_v_R * 3
        
        # Total dimensions
        self.total_dim_qk = num_heads_T * self.actual_dim_qk_T + num_heads_R * self.actual_dim_qk_R
        self.total_dim_v = num_heads_T * self.actual_dim_v_T + num_heads_R * self.actual_dim_v_R
        
        # Q, K, V projections
        self.W_q = nn.Linear(dim_in, self.total_dim_qk, bias=False)
        self.W_k = nn.Linear(dim_in, self.total_dim_qk, bias=False)
        self.W_v = nn.Linear(dim_in, self.total_dim_v, bias=False)
        
        # Swish layer
        self.swish = Swish(in_dim=self.total_dim_v, out_dim=dim_in, dropout_value=dropout_value)
        
        # SwiGLU layer
        self.swiglu = SwiGLU_Simplified(in_dim=dim_in, expansion_factor=expansion_factor, dropout_value=dropout_value)
        
        # RMSNorm layers
        self.norm_middle = nn.RMSNorm(dim_in)
        self.norm_final = nn.RMSNorm(dim_in)
        
        # Scaling factors per head type
        self.scale_T = 1.0 / math.sqrt(self.actual_dim_qk_T)
        self.scale_R = 1.0 / math.sqrt(self.actual_dim_qk_R)
        
    def build_local_frames(self, c):
        """
        Build local coordinate frames for each residue.
        
        Args:
            c: (batch, seq_len, 3, 3) - coordinates for [N, Ca, C]
            
        Returns:
            R: (batch, seq_len, 3, 3) - rotation matrices
            t: (batch, seq_len, 3) - translation vectors (Ca positions)
        """
        N = c[:, :, 0, :]   # (batch, seq_len, 3)
        Ca = c[:, :, 1, :]  # (batch, seq_len, 3)
        C = c[:, :, 2, :]   # (batch, seq_len, 3)
        
        # Origin at Ca
        t = Ca
        
        # x-axis: from Ca to C (pointing in negative direction, so Ca - C)
        x = Ca - C
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        if self.verbose:
            print("x is unit vector:", ((x**2).sum(axis=-1)==1).all())
        
        # e1: from Ca to N
        e1 = N - Ca
        e1 = e1 / (torch.norm(e1, dim=-1, keepdim=True) + 1e-8)
        if self.verbose:
            print("e1 is unit vector:", ((e1**2).sum(axis=-1)==1).all())
        
        # y-axis: Gram-Schmidt orthogonalization
        y = e1 - (x * e1).sum(dim=-1, keepdim=True) * x
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        if self.verbose:
            print("y is unit vector:", ((y**2).sum(axis=-1)==1).all())
            
        # z-axis: cross product
        z = torch.cross(x, y, dim=-1)
        
        # R = [x, y, z] stacked as columns
        R = torch.stack([x, y, z], dim=-1)  # (batch, seq_len, 3, 3)
        
        return R, t
    
    def apply_rotation(self, features, R):
        """
        Apply rotation to features.
        
        Args:
            features: (..., num_vectors, 3)
            R: (..., 3, 3)
            
        Returns:
            rotated: (..., num_vectors, 3)
        """
        # Use einsum for matrix multiplication: R @ features
        return torch.einsum('...ij,...nj->...ni', R, features)
    
    def apply_transformation(self, features, R, t_diff):
        """
        Apply rotation and translation to features.
        
        Args:
            features: (..., num_vectors, 3)
            R: (..., 3, 3)
            t_diff: (..., 3) - translation difference
            
        Returns:
            transformed: (..., num_vectors, 3)
        """
        rotated = self.apply_rotation(features, R)
        # Add translation to each vector
        return rotated + t_diff.unsqueeze(-2)
    
    def forward(self, x, m=1, c=None):
        """
        Args:
            x: (batch, seq_len, dim_in)
            m: mask (batch, seq_len)
            c: (batch, seq_len, 3, 3) - coordinates [N, Ca, C]
            
        Returns:
            output: (batch, seq_len, dim_in)
        """
        if c is None:
            raise ValueError("Coordinates c must be provided for geometric attention")
            
        batch_size, seq_len, _ = x.shape
        
        if self.verbose:
            print("\n=== GeometricSelfAttention Forward Pass ===")
            print(f"x.size() = {x.size()}")
            print(f"c.size() = {c.size()}")
        
        # Build local frames
        R, t = self.build_local_frames(c)  # R: (batch, seq_len, 3, 3), t: (batch, seq_len, 3)
        
        if self.verbose:
            print(f"R.size() = {R.size()}")
            print(f"t.size() = {t.size()}")
        
        # Extract Ca coordinates for distance calculation
        Ca = c[:, :, 1, :]  # (batch, seq_len, 3)
        
        # Project to Q, K, V
        q = self.W_q(x)  # (batch, seq_len, total_dim_qk)
        k = self.W_k(x)  # (batch, seq_len, total_dim_qk)
        v = self.W_v(x)  # (batch, seq_len, total_dim_v)
        
        if self.verbose:
            print(f"\nAfter projection:")
            print(f"q.size() = {q.size()}")
            print(f"k.size() = {k.size()}")
            print(f"v.size() = {v.size()}")
        
        # Split into T-heads and R-heads
        # Q and K splitting
        q_T = q[:, :, :self.num_heads_T * self.actual_dim_qk_T]
        q_R = q[:, :, self.num_heads_T * self.actual_dim_qk_T:]
        k_T = k[:, :, :self.num_heads_T * self.actual_dim_qk_T]
        k_R = k[:, :, self.num_heads_T * self.actual_dim_qk_T:]
        
        # V splitting
        v_T = v[:, :, :self.num_heads_T * self.actual_dim_v_T]
        v_R = v[:, :, self.num_heads_T * self.actual_dim_v_T:]
        
        if self.verbose:
            print(f"\nAfter T/R split:")
            print(f"q_T.size() = {q_T.size()}, q_R.size() = {q_R.size()}")
            print(f"k_T.size() = {k_T.size()}, k_R.size() = {k_R.size()}")
            print(f"v_T.size() = {v_T.size()}, v_R.size() = {v_R.size()}")
        
        # Reshape for multi-head attention with 3D vectors
        if self.verbose:
            print(f"\nAfter reshape to 3D vectors:")
        # T-heads: (batch, seq_len, num_heads_T, num_vectors, 3)
        if self.num_heads_T > 0:
            q_T = q_T.reshape(batch_size, seq_len, self.num_heads_T, -1, 3).transpose(1, 2)
            k_T = k_T.reshape(batch_size, seq_len, self.num_heads_T, -1, 3).transpose(1, 2)
            v_T = v_T.reshape(batch_size, seq_len, self.num_heads_T, -1, 3).transpose(1, 2)
            if self.verbose:
                print(f"q_T.size() = {q_T.size()}")
                print(f"k_T.size() = {k_T.size()}")
                print(f"v_T.size() = {v_T.size()}")
        
        # R-heads
        if self.num_heads_R > 0:
            q_R = q_R.reshape(batch_size, seq_len, self.num_heads_R, -1, 3).transpose(1, 2)
            k_R = k_R.reshape(batch_size, seq_len, self.num_heads_R, -1, 3).transpose(1, 2)
            v_R = v_R.reshape(batch_size, seq_len, self.num_heads_R, -1, 3).transpose(1, 2)
            if self.verbose:
                print(f"q_R.size() = {q_R.size()}")
                print(f"k_R.size() = {k_R.size()}")
                print(f"v_R.size() = {v_R.size()}")
        
        # ====== Compute relative transformations ======
        # R_i @ R_j^T for transforming from frame j to frame i
        R_i = R.unsqueeze(2)  # (batch, seq_len_i, 1, 3, 3)
        R_j = R.unsqueeze(1)  # (batch, 1, seq_len_j, 3, 3)
        R_rel = torch.matmul(R_i, R_j.transpose(-2, -1))  # (batch, seq_len_i, seq_len_j, 3, 3)
        
        # Translation difference: t_j - t_i
        t_i = t.unsqueeze(2)  # (batch, seq_len_i, 1, 3)
        t_j = t.unsqueeze(1)  # (batch, 1, seq_len_j, 3)
        t_diff = t_j - t_i  # (batch, seq_len_i, seq_len_j, 3)
        
        if self.verbose:
            print(f"\nRelative transformations:")
            print(f"R_rel.size() = {R_rel.size()}")
            print(f"t_diff.size() = {t_diff.size()}")

        # ==== Prepare Rot matrices for T or R transform ==========
        R_expanded = R.unsqueeze(1)  # (batch, 1, seq_len, 3, 3)
        R_rel_expanded = R_rel.unsqueeze(1)  # (batch, 1, seq_len_i, seq_len_j, 3, 3)
        
        # ====== Transform T-heads ======
        if self.num_heads_T > 0:
            # Queries: apply R_i to all vectors at position i
            q_T_transformed = self.apply_rotation(q_T, R_expanded)
            
            if self.verbose:
                print(f"\nT-heads transformations:")
                print(f"q_T_transformed.size() = {q_T_transformed.size()}")
            
            # Keys: transform from frame j to frame i
            k_T_expanded = k_T.unsqueeze(2)  # (batch, num_heads_T, 1, seq_len_j, num_vectors, 3)
            
            # Split position and direction vectors
            k_T_position = k_T_expanded[:, :, :, :, :self.dim_qk_T_position, :]
            k_T_direction = k_T_expanded[:, :, :, :, self.dim_qk_T_position:, :]
            
            if self.verbose:
                print(f"k_T_position.size() = {k_T_position.size()}")
                print(f"k_T_direction.size() = {k_T_direction.size()}")
            
            # Position vectors: R_rel @ k + t_diff
            k_T_position_transformed = self.apply_rotation(k_T_position, R_rel_expanded)
            t_diff_expanded = t_diff.unsqueeze(1).unsqueeze(-2)  # (batch, 1, seq_i, seq_j, 1, 3)
            k_T_position_transformed = k_T_position_transformed + t_diff_expanded
            
            # Direction vectors: R_rel @ k (no translation)
            k_T_direction_transformed = self.apply_rotation(k_T_direction, R_rel_expanded)
            
            # Concatenate position and direction
            k_T_transformed = torch.cat([k_T_position_transformed, k_T_direction_transformed], dim=-2)
            
            if self.verbose:
                print(f"k_T_transformed.size() = {k_T_transformed.size()}")
            
            # Values: R_rel @ v + t_diff
            v_T_expanded = v_T.unsqueeze(2)  # (batch, num_heads_T, 1, seq_len_j, num_vectors, 3)
            v_T_transformed = self.apply_rotation(v_T_expanded, R_rel_expanded) + t_diff_expanded
            
            if self.verbose:
                print(f"v_T_transformed.size() = {v_T_transformed.size()}")

            # ====== Compute attention scores ======
            # Flatten 3D vectors for dot product computation
            # T-heads
            q_T_flat = q_T_transformed.reshape(batch_size, self.num_heads_T, seq_len, -1)
            k_T_flat = k_T_transformed.reshape(batch_size, self.num_heads_T, seq_len, seq_len, -1)
            q_T_expanded = q_T_flat.unsqueeze(3)  # (batch, heads, seq_i, 1, dim)
            attention_scores_T = (q_T_expanded * k_T_flat).sum(dim=-1) * self.scale_T
            
            if self.verbose:
                print(f"\nAttention computation:")
                print(f"q_T_flat.size() = {q_T_flat.size()}")
                print(f"k_T_flat.size() = {k_T_flat.size()}")
                print(f"attention_scores_T.size() = {attention_scores_T.size()}")
        
        # ====== Transform R-heads ======
        if self.num_heads_R > 0:
            # Queries: R_i @ q
            q_R_transformed = self.apply_rotation(q_R, R_expanded)
            
            if self.verbose:
                print(f"\nR-heads transformations:")
                print(f"q_R_transformed.size() = {q_R_transformed.size()}")
            
            # Keys: R_rel @ k
            k_R_expanded = k_R.unsqueeze(2)
            k_R_transformed = self.apply_rotation(k_R_expanded, R_rel_expanded)
            
            if self.verbose:
                print(f"k_R_transformed.size() = {k_R_transformed.size()}")
            
            # Values: R_i @ v (just rotate into recipient frame)
            # For each position i, apply R_i to all v_j
            R_i_for_v = R.unsqueeze(1).unsqueeze(3)  # (batch, 1, seq_len_i, 1, 3, 3)
            v_R_expanded = v_R.unsqueeze(2)  # (batch, num_heads_R, 1, seq_len_j, num_vectors, 3)
            v_R_transformed = self.apply_rotation(v_R_expanded, R_i_for_v)
            
            if self.verbose:
                print(f"v_R_transformed.size() = {v_R_transformed.size()}")
        
            # ====== Compute attention scores ======
            # Flatten 3D vectors for dot product computation
            # R-heads
            q_R_flat = q_R_transformed.reshape(batch_size, self.num_heads_R, seq_len, -1)
            k_R_flat = k_R_transformed.reshape(batch_size, self.num_heads_R, seq_len, seq_len, -1)
            q_R_expanded = q_R_flat.unsqueeze(3)
            attention_scores_R = (q_R_expanded * k_R_flat).sum(dim=-1) * self.scale_R
            
            if self.verbose:
                print(f"q_R_flat.size() = {q_R_flat.size()}")
                print(f"k_R_flat.size() = {k_R_flat.size()}")
                print(f"attention_scores_R.size() = {attention_scores_R.size()}")
        
        # =========== Concatenate attention scores from T and R heads ==========
        if min(self.num_heads_R, self.num_heads_T) > 0:
            attention_scores = torch.cat([attention_scores_T, attention_scores_R], dim=1)
        elif self.num_heads_R == 0:
            attention_scores = attention_scores_T
        else:
            attention_scores = attention_scores_R
        
        if self.verbose:
            print(f"attention_scores.size() (combined) = {attention_scores.size()}")
        
        # Apply distance-based masking
        dist = torch.sqrt(((Ca.unsqueeze(-2) - Ca.unsqueeze(-3))**2).sum(dim=-1) + 1e-8)
        dist_mask = SigmoidWindow.calculate(x=dist, num_ring=self.num_ring)
        attention_scores = attention_scores * dist_mask.unsqueeze(1)
        
        if self.verbose:
            print(f"dist.size() = {dist.size()}")
            print(f"dist_mask.size() = {dist_mask.size()}")
        
        # Apply activation
        if self.attention_activation == 'softmax':
            attention = torch.softmax(attention_scores, dim=-1)
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
        
        if self.verbose:
            print(f"attention.size() = {attention.size()}")
        
        # ====== Apply attention to values ======
        # Split attention for T and R heads
        attention_T = attention[:, :self.num_heads_T, :, :]
        attention_R = attention[:, self.num_heads_T:, :, :]
        
        # Flatten values and apply attention
        if self.verbose:
            print(f"\nMessages:")
            
        if self.num_heads_T > 0:
            v_T_flat = v_T_transformed.reshape(batch_size, self.num_heads_T, seq_len, seq_len, -1)
            message_T = torch.einsum('bhij,bhijd->bhid', attention_T, v_T_flat)
            if self.verbose:
                print(f"message_T.size() = {message_T.size()}")

        if self.num_heads_R > 0:
            v_R_flat = v_R_transformed.reshape(batch_size, self.num_heads_R, seq_len, seq_len, -1)
            message_R = torch.einsum('bhij,bhijd->bhid', attention_R, v_R_flat)
            if self.verbose:
                print(f"message_R.size() = {message_R.size()}")
        
        # Concatenate messages and reshape
        if min(self.num_heads_R, self.num_heads_T) > 0:
            message = torch.cat([message_T, message_R], dim=1)
        elif self.num_heads_R == 0:
            message = message_T
        else:
            message = message_R
        message = message.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.total_dim_v)
        
        if self.verbose:
            print(f"message.size() (combined) = {message.size()}")
        
        # Apply residual connections with Swish and SwiGLU
        middle_output = self.norm_middle(x + self.swish(message) * m.reshape(batch_size, seq_len, 1))
        final_output = self.norm_final(x + middle_output + self.swiglu(middle_output) * m.reshape(batch_size, seq_len, 1))
        
        if self.verbose:
            print(f"\nOutput:")
            print(f"final_output.size() = {final_output.size()}")
            print("=== End of Forward Pass ===\n")
        
        return final_output