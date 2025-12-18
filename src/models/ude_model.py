import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import math

class UDE(nn.Module):
    """
    Universal Differential Equation for stress dynamics.
    ENHANCED: Multi-modal feature input support.
    
    Model: dS/dt = -β*S + α*W + NN(S, Features)
    
    Where:
    - S: Stress level (state variable)
    - W: Workload (heart rate proxy)
    - Features: Multi-modal physiological features (EDA, Temp, Resp, ACC, EMG, HRV)
    - β: Recovery rate (higher = faster stress decay, GOOD)
    - α: Sensitivity (higher = more stressed by workload, BAD)
    - NN: Neural network correction term for unmodeled dynamics
    
    Parameters are constrained to be positive via softplus transformation.
    """
    def __init__(self, hidden_dim=64, num_features=18):
        """
        Args:
            hidden_dim: Size of hidden layers (increased from 32 to 64 for richer features)
            num_features: Number of input features (default 18 for multi-modal)
        """
        super(UDE, self).__init__()
        
        # Neural Network: g(S, Features) -> dS_nn
        # Input: Stress (1) + All Features (num_features)
        input_dim = 1 + num_features  # S + all features
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Known parameters (learnable, raw values before softplus)
        # dS = -beta * S + alpha * W
        # Initialize such that softplus gives reasonable starting values
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))  # softplus(-2.9) ≈ 0.05
        self._alpha_raw = nn.Parameter(torch.tensor([-2.2]))  # softplus(-2.2) ≈ 0.1
        
        # Placeholders for current batch data
        self.current_features = None  # (batch, seq_len, num_features)
        self.current_t = None  # (seq_len,)
        self.num_features = num_features

    @property
    def beta(self):
        """Recovery rate (always positive via softplus)"""
        return F.softplus(self._beta_raw)
    
    @property
    def alpha(self):
        """Sensitivity to workload (always positive via softplus)"""
        return F.softplus(self._alpha_raw)

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
        # Use math.floor for efficiency (no tensor creation)
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
        
        Args:
            t: scalar tensor, current time
            y: (batch, 1), current stress levels
            
        Returns:
            dS/dt: (batch, 1), rate of change of stress
        """
        S = y
        
        # Get all features at current time
        features = self.get_features_at_t(t.item())  # (batch, num_features)
        
        # Extract workload (first feature, index 0)
        W = features[:, 0:1]  # (batch, 1) - workload
        
        # Known dynamics with positive parameter constraints
        # dS = -beta * S + alpha * W
        f_known = -self.beta * S + self.alpha * W
        
        # Neural network correction term using ALL features
        # Input: [S, all_features]
        nn_in = torch.cat([S, features], dim=-1)  # (batch, 1 + num_features)
        f_nn = self.net(nn_in)
        
        return f_known + f_nn
    
    def get_interpretable_params(self):
        """
        Get the interpretable (positive) parameter values.
        
        Returns:
            dict with 'alpha' (sensitivity) and 'beta' (recovery rate)
        """
        return {
            'alpha': self.alpha.item(),
            'beta': self.beta.item(),
            'risk_score': self.alpha.item() / (self.beta.item() + 1e-6)
        }
