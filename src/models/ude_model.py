import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import math

class UDE(nn.Module):
    """
    Universal Differential Equation for stress dynamics.
    MULTI-COEFFICIENT VERSION: Feature-specific sensitivities for personalization.
    
    Model: dS/dt = -β*S + Σᵢ αᵢ*Fᵢ + NN(S, Features)
    
    Where:
    - S: Stress level (state variable)
    - Fᵢ: Individual physiological features (HRV, EDA, Temp, etc.)
    - αᵢ: Feature-specific sensitivities (18 parameters, one per feature)
    - β: Recovery rate (single parameter - universal across features)
    - NN: Neural network correction term for nonlinear dynamics
    
    KEY CHANGE: Instead of single α (workload sensitivity), we now have α₁...α₁₈
    (one sensitivity coefficient per physiological feature).
    
    This enables rich personalization:
    - Person A: High α_HRV, low α_EDA → "Cardiac stress responder"
    - Person B: Low α_HRV, high α_EDA → "Anxiety stress responder"
    """
    def __init__(self, hidden_dim=64, num_features=18):
        """
        Args:
            hidden_dim: Size of hidden layers in neural network
            num_features: Number of input features (default 18 for multi-modal WESAD)
        """
        super(UDE, self).__init__()
        
        # Neural Network: g(S, Features) -> dS_nn (nonlinear correction)
        # Input: Stress (1) + All Features (num_features)
        input_dim = 1 + num_features
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # === MULTI-COEFFICIENT PARAMETERS ===
        # Recovery rate (single, shared across all features)
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))  # softplus(-2.9) ≈ 0.05
        
        # Feature-specific sensitivities (vector of 18 parameters)
        # Each αᵢ controls how much feature i drives stress
        # Initialize all to same safe value, they'll diverge during training
        self._alphas_raw = nn.Parameter(torch.ones(num_features) * (-2.2))
        # This gives ~0.1 for each after softplus (stable start)
        
        # Placeholders for current batch data
        self.current_features = None  # (batch, seq_len, num_features)
        self.current_t = None  # (seq_len,)
        self.num_features = num_features

    @property
    def beta(self):
        """Recovery rate (always positive via softplus)"""
        return F.softplus(self._beta_raw)
    
    @property
    def alphas(self):
        """Feature-specific sensitivities (always positive via softplus)"""
        return F.softplus(self._alphas_raw)  # Shape: (num_features,)

    def set_current_batch(self, t, features):
        """
        Set the current batch of features for ODE integration.
        
        Args:
            t: (seq_len,) time points
            features: (batch, seq_len, num_features) all physiological features
        """
        self.current_t = t
        self.current_features = features

    def get_features_at_t(self, t_scalar):
        """
        Interpolate features at time t_scalar.
        
        Args:
            t_scalar: float, query time point
            
        Returns:
            Interpolated features tensor (batch, num_features)
        """
        idx_low = int(math.floor(t_scalar))
        idx_high = idx_low + 1
        
        # Clamp to valid range
        max_idx = self.current_features.shape[1] - 1
        idx_low = max(0, min(idx_low, max_idx))
        idx_high = max(0, min(idx_high, max_idx))
        
        features_low = self.current_features[:, idx_low, :]
        features_high = self.current_features[:, idx_high, :]
        
        # Linear interpolation weight
        weight = t_scalar - int(math.floor(t_scalar))
        
        features_interp = features_low * (1 - weight) + features_high * weight
        return features_interp

    def forward(self, t, y):
        """
        Compute dS/dt at time t given current state y.
        
        MULTI-COEFFICIENT UPDATE:
        Instead of: dS/dt = -β*S + α*W
        We now use: dS/dt = -β*S + Σᵢ αᵢ*Fᵢ
        
        Args:
            t: scalar tensor, current time
            y: (batch,), current stress levels (odeint passes without extra dim)
            
        Returns:
            dS/dt: (batch,), rate of change of stress
        """
        S = y  # Current stress level (batch,)
        
        # Get all features at current time
        features = self.get_features_at_t(t.item())  # (batch, num_features)
        
        # === MULTI-COEFFICIENT PHYSICS TERM ===
        # Old: f_known = -beta * S + alpha * W
        # New: f_known = -beta * S + sum(alphas[i] * features[i])
        
        # Recovery term (universal)
        recovery = -self.beta * S  # (batch,)
        
        # Feature-weighted accumulation
        # alphas: (num_features,)
        # features: (batch, num_features)
        # We want: sum over features of (alpha_i * feature_i) for each batch
        feature_contribution = torch.sum(self.alphas * features, dim=-1)  # (batch,)
        
        f_known = recovery + feature_contribution  # (batch,)
        
        # Neural network correction term (nonlinear dynamics)
        # Handle both (batch,) and (batch, 1) cases
        if S.dim() == 1:
            S_expanded = S.unsqueeze(-1)  # (batch, 1)
        else:
            S_expanded = S  # already (batch, 1)
        
        nn_in = torch.cat([S_expanded, features], dim=-1)  # (batch, 1 + num_features)
        f_nn = self.net(nn_in)
        
        # Return same shape as input
        if S.dim() == 1:
            f_nn = f_nn.squeeze(-1)  # (batch,)
        
        return f_known + f_nn
    
    def get_interpretable_params(self):
        """
        Get the interpretable (positive) parameter values.
        
        NEW: Returns all 19 parameters (18 alphas + 1 beta)
        
        Returns:
            dict with:
            - 'alphas': array of 18 feature sensitivities
            - 'beta': recovery rate
            - 'alpha_mean': average sensitivity (for backward compatibility)
            - 'risk_score': mean(alphas) / beta (simplified risk metric)
        """
        alphas_vals = self.alphas.detach().cpu().numpy()
        beta_val = self.beta.item()
        
        return {
            'alphas': alphas_vals,  # Array of 18 values
            'beta': beta_val,
            'alpha_mean': alphas_vals.mean(),  # Average for comparison
            'alpha': alphas_vals[0],  # Workload sensitivity (first feature)
            'risk_score': alphas_vals.mean() / (beta_val + 1e-6)
        }
