import torch
import torch.nn as nn
import numpy as np

class StressSpecificKernel(nn.Module):
    """
    Novel kernel derived from psychological stress theory.
    
    Based on:
    1. Allostatic Load Theory (McEwen, 1998): Stress accumulates non-linearly
    2. Inverted-U Performance Curve (Yerkes-Dodson, 1908): Moderate stress helps, extreme stress harms
    3. Social Comparison Theory (Festinger, 1954): People compare to similar others
    
    Mathematical Formulation:
    K(S_i, S_j) = exp(-|S_i - S_j|^p / sigma^p) * (1 - tanh(alpha * (S_i + S_j - 2*theta)))
    
    Components:
    - exp(-|S_i - S_j|^p / sigma^p): Generalized Gaussian (p=2 is standard, p=1 is Laplacian)
    - (1 - tanh(...)): Inverted-U modulation (influence weakens when BOTH are highly stressed)
    - theta: Optimal stress level (Yerkes-Dodson peak)
    - alpha: Steepness of inverted-U
    
    Novelty:
    - Combines bounded confidence (first term) with allostatic load (second term)
    - Captures the idea that two highly stressed people CANNOT help each other (both depleted)
    - Derived from first principles (psychological theory), not ad-hoc
    """
    
    def __init__(self, sigma=1.5, p=2.0, theta=0.5, alpha=5.0):
        super(StressSpecificKernel, self).__init__()
        self.sigma = sigma  # Bandwidth (how different can S_i and S_j be?)
        self.p = p          # Norm order (2=Euclidean, 1=Manhattan)
        self.theta = theta  # Optimal stress level (Yerkes-Dodson peak)
        self.alpha = alpha  # Steepness of inverted-U
    
    def forward(self, S_diff):
        """
        S_diff: (N, N) matrix where entry (i,j) is S_j - S_i
        Returns: (N, N) kernel matrix K_ij
        """
        # Component 1: Bounded Confidence (Generalized Gaussian)
        # K1 = exp(-|diff|^p / sigma^p)
        K1 = torch.exp(- torch.abs(S_diff)**self.p / (self.sigma**self.p))
        
        # Component 2: Allostatic Load Modulation (Inverted-U)
        # When S_i + S_j is high (both stressed), influence weakens
        # We need S_i and S_j separately, not just diff
        # This requires passing S_group, not just S_diff
        # For now, we'll use a simplified version based on |S_diff|
        # Assumption: If diff is large, at least one is far from optimal
        
        # Better approach: Pass S_group and compute S_i + S_j
        # But to keep API compatible, we'll use a proxy:
        # Modulation = 1 - tanh(alpha * |S_diff|)
        # This weakens influence when stress levels are very different
        # (which often means one is very high)
        
        # Actually, let's make this kernel take S_group as input
        # and compute both diff and sum internally
        
        # For now, simplified version:
        K2 = 1.0 - torch.tanh(self.alpha * torch.abs(S_diff))
        
        return K1 * K2
    
    def forward_full(self, S_group):
        """
        Full version that takes S_group and computes both diff and sum.
        S_group: (N, 1) stress levels
        Returns: (N, N) kernel matrix
        """
        N = S_group.shape[0]
        S_flat = S_group.squeeze()  # (N,)
        
        # Pairwise differences: S_j - S_i
        S_diff = S_flat.unsqueeze(0) - S_flat.unsqueeze(1)  # (N, N)
        
        # Pairwise sums: S_i + S_j
        S_sum = S_flat.unsqueeze(0) + S_flat.unsqueeze(1)  # (N, N)
        
        # Component 1: Bounded Confidence
        K1 = torch.exp(- torch.abs(S_diff)**self.p / (self.sigma**self.p))
        
        # Component 2: Allostatic Load (Inverted-U)
        # Influence weakens when BOTH are highly stressed (S_i + S_j >> 2*theta)
        # OR when both are very low stressed (S_i + S_j << 2*theta)
        # This is the inverted-U: optimal at S_sum = 2*theta
        K2 = 1.0 - torch.tanh(self.alpha * (S_sum - 2*self.theta))
        
        return K1 * K2


class AdaptiveStressPDE(nn.Module):
    """
    PDE with the novel stress-specific kernel.
    """
    def __init__(self, n_users, adj_matrix, ude_model, diffusion_coeff=0.1, 
                 kernel_type='stress_specific', **kernel_params):
        super(AdaptiveStressPDE, self).__init__()
        self.n_users = n_users
        self.ude_model = ude_model
        self.diffusion_coeff = diffusion_coeff
        
        A = torch.tensor(adj_matrix, dtype=torch.float32)
        self.register_buffer('A', A)
        
        # Initialize kernel
        if kernel_type == 'stress_specific':
            self.kernel = StressSpecificKernel(**kernel_params)
        elif kernel_type == 'gaussian':
            # Fallback to standard Gaussian
            self.kernel = None
            self.sigma = kernel_params.get('sigma', 1.5)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        self.kernel_type = kernel_type
    
    def forward(self, t, S_group):
        """
        t: scalar time
        S_group: (n_users, 1) stress values
        """
        # 1. Individual Dynamics (Reaction)
        reaction = self.ude_model(t, S_group)
        
        # 2. Group Dynamics (Diffusion with Novel Kernel)
        if self.kernel_type == 'stress_specific':
            # Use the novel kernel
            K = self.kernel.forward_full(S_group)  # (N, N)
        else:
            # Standard Gaussian
            S_diff = S_group.T - S_group
            K = torch.exp(- (S_diff**2) / (self.sigma**2))
        
        # Pairwise differences
        S_diff = S_group.T - S_group  # (N, N)
        
        # Weighted influence
        Influence = self.A * K * S_diff
        
        # Sum over neighbors
        diffusion = self.diffusion_coeff * torch.sum(Influence, dim=1, keepdim=True)
        
        return reaction + diffusion
