import torch
import torch.nn as nn
import numpy as np

class DynamicStressPDE(nn.Module):
    """
    Enhanced PDE model with time-varying adjacency matrix.
    Models social withdrawal during high stress.
    """
    def __init__(self, n_users, adj_matrix, ude_model, diffusion_coeff=0.1, 
                 withdrawal_threshold=0.7, withdrawal_strength=0.5):
        super(DynamicStressPDE, self).__init__()
        self.n_users = n_users
        self.ude_model = ude_model
        self.diffusion_coeff = diffusion_coeff
        self.withdrawal_threshold = withdrawal_threshold
        self.withdrawal_strength = withdrawal_strength
        
        # Base adjacency matrix
        A_base = torch.tensor(adj_matrix, dtype=torch.float32)
        self.register_buffer('A_base', A_base)
        
    def compute_dynamic_adjacency(self, S_group):
        """
        Compute time-varying adjacency based on current stress levels.
        
        Rule: If S_i > threshold, reduce outgoing connections by withdrawal_strength
        This models social withdrawal during burnout.
        """
        # S_group is (n_users, 1)
        S_flat = S_group.squeeze()  # (n_users,)
        
        # Identify high-stress individuals
        high_stress = (S_flat > self.withdrawal_threshold).float()  # (n_users,)
        
        # Compute withdrawal factor for each user
        # withdrawal_factor = 1 - withdrawal_strength * high_stress
        # If high_stress=1, factor = 1 - 0.5 = 0.5 (50% reduction)
        # If high_stress=0, factor = 1.0 (no reduction)
        withdrawal_factor = 1.0 - self.withdrawal_strength * high_stress  # (n_users,)
        
        # Apply to outgoing edges (rows of adjacency matrix)
        # A_dynamic[i, j] = A_base[i, j] * withdrawal_factor[i]
        A_dynamic = self.A_base * withdrawal_factor.unsqueeze(1)  # (n_users, n_users)
        
        return A_dynamic
        
    def forward(self, t, S_group):
        """
        t: scalar time
        S_group: (n_users, 1) stress values
        """
        # 1. Individual Dynamics (Reaction)
        reaction = self.ude_model(t, S_group)
        
        # 2. Dynamic Group Dynamics
        # Compute current adjacency matrix
        A_t = self.compute_dynamic_adjacency(S_group)
        
        # Pairwise differences
        S_diff = S_group.T - S_group  # (n_users, n_users)
        
        # Gaussian kernel
        sigma = 1.5
        K = torch.exp(- (S_diff**2) / (sigma**2))
        
        # Weighted influence with dynamic adjacency
        Influence = A_t * K * S_diff
        
        # Sum over neighbors
        diffusion = self.diffusion_coeff * torch.sum(Influence, dim=1, keepdim=True)
        
        return reaction + diffusion
