"""
Multi-Coefficient UDE Model
Complete, tested, ready for Kaggle training

Model: dS/dt = -β·S + Σᵢ αᵢ·Fᵢ + NN(S, F)

Where:
- β: Single recovery rate
- αᵢ: 18 feature-specific sensitivities (one per feature)
- NN: Neural network for nonlinear dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UDEMultiCoeff(nn.Module):
    """
    Universal Differential Equation with Multi-Coefficient Feature Sensitivities
    
    This model learns:
    - 1 beta (recovery rate)
    - 18 alphas (feature-specific sensitivities)
    - Neural network weights
    
    Total interpretable parameters: 19
    """
    
    def __init__(self, hidden_dim=64, num_features=18):
        """
        Args:
            hidden_dim: Hidden layer size for neural network
            num_features: Number of input features (default 18 for WESAD)
        """
        super(UDEMultiCoeff, self).__init__()
        
        self.num_features = num_features
        
        # Neural Network: Takes stress + all features
        input_dim = 1 + num_features
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # === LEARNABLE PARAMETERS ===
        # Recovery rate (single, shared)
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))  # softplus(-2.9) ≈ 0.05
        
        # Feature-specific sensitivities (18 separate values)
        self._alphas_raw = nn.Parameter(torch.ones(num_features) * (-2.2))  # softplus(-2.2) ≈ 0.1
        
        # Batch data (set before ODE solving)
        self.current_features = None  # (batch, seq_len, num_features)
        self.current_t = None  # (seq_len,)
    
    @property
    def beta(self):
        """Recovery rate (always positive via softplus)"""
        return F.softplus(self._beta_raw)
    
    @property
    def alphas(self):
        """Feature-specific sensitivities (always positive via softplus)"""
        return F.softplus(self._alphas_raw)  # Shape: (18,)
    
    def set_current_batch(self, t, features):
        """
        Set current batch data for ODE integration
        
        Args:
            t: (seq_len,) time points
            features: (batch, seq_len, num_features) feature tensor
        """
        self.current_t = t
        self.current_features = features
    
    def get_features_at_t(self, t_scalar):
        """
        Interpolate features at specific time point
        
        Args:
            t_scalar: float, query time
            
        Returns:
            features: (batch, num_features) interpolated features
        """
        # Find surrounding indices
        idx_low = int(math.floor(t_scalar))
        idx_high = idx_low + 1
        
        # Clamp to valid range
        max_idx = self.current_features.shape[1] - 1
        idx_low = max(0, min(idx_low, max_idx))
        idx_high = max(0, min(idx_high, max_idx))
        
        # If at boundary, no interpolation needed
        if idx_low == idx_high:
            return self.current_features[:, idx_low, :]
        
        # Linear interpolation
        alpha = t_scalar - idx_low
        features_low = self.current_features[:, idx_low, :]
        features_high = self.current_features[:, idx_high, :]
        
        return (1 - alpha) * features_low + alpha * features_high
    
    def forward(self, t, y):
        """
        Compute dS/dt at time t given current state y
        
        Multi-Coefficient Equation:
        dS/dt = -β·S + Σᵢ αᵢ·Fᵢ + NN(S, F)
        
        Args:
            t: scalar tensor, current time
            y: (batch, 1) current stress levels
            
        Returns:
            dS/dt: (batch, 1) rate of change
        """
        # Handle input shape (batch, 1) or (batch,)
        S = y.squeeze(-1) if y.dim() > 1 else y  # Make (batch,)
        batch_size = S.shape[0]
        
        # Get features at current time
        features = self.get_features_at_t(t.item())  # (batch, num_features)
        
        # === MULTI-COEFFICIENT PHYSICS ===
        # Recovery term: -β·S
        recovery = -self.beta * S  # (batch,)
        
        # Feature-specific contributions: Σᵢ αᵢ·Fᵢ
        # alphas: (18,)
        # features: (batch, 18)
        # Result: (batch,) - sum over features dimension
        feature_contribution = torch.sum(self.alphas * features, dim=-1)  # (batch,)
        
        # Physics-based term
        f_physics = recovery + feature_contribution  # (batch,)
        
        # === NEURAL NETWORK CORRECTION ===
        # Input: [S, F1, F2, ..., F18]
        nn_input = torch.cat([S.unsqueeze(-1), features], dim=-1)  # (batch, 19)
        f_nn = self.net(nn_input).squeeze(-1)  # (batch,)
        
        # Total derivative
        dS_dt = f_physics + f_nn  # (batch,)
        
        # Return same shape as input
        return dS_dt.unsqueeze(-1) if y.dim() > 1 else dS_dt
    
    def get_learned_params(self):
        """
        Get interpretable learned parameters
        
        Returns:
            dict with beta and alphas
        """
        return {
            'beta': self.beta.detach().cpu().item(),
            'alphas': self.alphas.detach().cpu().numpy()
        }
    
    def print_equation(self, feature_names=None):
        """
        Print the discovered equation in human-readable form
        
        Args:
            feature_names: List of feature names (optional)
        """
        params = self.get_learned_params()
        beta = params['beta']
        alphas = params['alphas']
        
        if feature_names is None:
            feature_names = [f'F{i+1}' for i in range(self.num_features)]
        
        print(f"\nDiscovered Equation:")
        print(f"dS/dt = -{beta:.4f}·S", end='')
        
        for i, (alpha, fname) in enumerate(zip(alphas, feature_names)):
            if alpha > 0.01:  # Only show significant terms
                print(f" + {alpha:.4f}·{fname}", end='')
        
        print(" + NN(S, F)")
        
        # Show top 5 most important features
        sorted_idx = torch.argsort(torch.tensor(alphas), descending=True)
        print(f"\nTop 5 Important Features:")
        for i in range(min(5, len(alphas))):
            idx = sorted_idx[i]
            print(f"  {i+1}. {feature_names[idx]}: α = {alphas[idx]:.4f}")
