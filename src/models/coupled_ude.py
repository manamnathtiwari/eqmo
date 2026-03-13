import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class CoupledUDE(nn.Module):
    """
    Coupled Universal Differential Equation System
    Models interdependent dynamics of Stress, HRV, and EDA
    
    System:
    dS/dt = -β₁·S + γ₁₁·HRV + γ₁₂·EDA + γ₁₃·S·HRV + NN_S(S,HRV,EDA)
    dHRV/dt = -β₂·HRV + γ₂₁·S + γ₂₂·(baseline-HRV) + NN_HRV(S,HRV,EDA)
    dEDA/dt = -β₃·EDA + γ₃₁·S + NN_EDA(S,HRV,EDA)
    """
    
    def __init__(self, selected_features=None, hidden_size=32):
        super(CoupledUDE, self).__init__()
        
        if selected_features is None:
            selected_features = ['HRV', 'EDA']
        self.selected_features = selected_features
        self.n_features = len(selected_features)
        self.n_vars = 1 + self.n_features  # Stress + features
        
        # Recovery rates (β parameters) - one per variable
        self._beta_stress = nn.Parameter(torch.tensor(-2.0))
        self._beta_hrv = nn.Parameter(torch.tensor(-2.1))
        self._beta_eda = nn.Parameter(torch.tensor(-1.9))
        
        # Coupling coefficients (γ parameters)
        # Features → Stress
        self._gamma_hrv_to_stress = nn.Parameter(torch.tensor(-2.0))
        self._gamma_eda_to_stress = nn.Parameter(torch.tensor(-2.1))
        
        # Stress → Features
        self._gamma_stress_to_hrv = nn.Parameter(torch.tensor(-1.8))
        self._gamma_stress_to_eda = nn.Parameter(torch.tensor(-1.9))
        
        # Interaction term (S × F1)
        self._gamma_interaction = nn.Parameter(torch.tensor(-3.0))
        
        # Homeostasis for HRV
        self._gamma_hrv_homeostasis = nn.Parameter(torch.tensor(-2.2))
        self.hrv_baseline = nn.Parameter(torch.tensor(60.0))
        
        # Neural network corrections
        input_dim = self.n_vars
        self.nn_stress = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.nn_hrv = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.nn_eda = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Current batch data
        self.current_features = None
        self.current_t = None
    
    @property
    def beta_stress(self):
        return F.softplus(self._beta_stress)
    
    @property
    def beta_hrv(self):
        return F.softplus(self._beta_hrv)
    
    @property
    def beta_eda(self):
        return F.softplus(self._beta_eda)
    
    @property
    def gamma_hrv_to_stress(self):
        return F.softplus(self._gamma_hrv_to_stress)
    
    @property
    def gamma_eda_to_stress(self):
        return F.softplus(self._gamma_eda_to_stress)
    
    @property
    def gamma_interaction(self):
        return F.softplus(self._gamma_interaction)
    
    @property
    def gamma_stress_to_hrv(self):
        return F.softplus(self._gamma_stress_to_hrv)
    
    @property
    def gamma_hrv_homeostasis(self):
        return F.softplus(self._gamma_hrv_homeostasis)
    
    @property
    def gamma_stress_to_eda(self):
        return F.softplus(self._gamma_stress_to_eda)
    
    def set_current_batch(self, t, features):
        """
        t: (seq_len,) time points
        features: (batch, seq_len, n_features) - HRV, EDA values
        """
        self.current_t = t
        self.current_features = features
    
    def get_features_at_t(self, t_val):
        """Get feature values at specific time point via interpolation"""
        if self.current_features is None:
            raise ValueError("Must call set_current_batch first")
        t_idx = torch.argmin(torch.abs(self.current_t - t_val))
        return self.current_features[:, t_idx, :]
    
    def forward(self, t, y):
        """
        Coupled ODE system
        t: scalar time
        y: (batch, n_vars) where n_vars = [Stress, HRV, EDA]
        
        Returns: dy/dt (batch, n_vars)
        """
        # Extract variables
        S = y[:, 0:1]
        HRV = y[:, 1:2]
        EDA = y[:, 2:3] if self.n_vars > 2 else torch.zeros_like(S)
        
        # === STRESS EQUATION ===
        dS_dt_known = (
            -self.beta_stress * S +
            self.gamma_hrv_to_stress * HRV +
            self.gamma_eda_to_stress * EDA +
            self.gamma_interaction * S * HRV
        )
        
        state = torch.cat([S, HRV, EDA], dim=1)
        dS_dt_nn = self.nn_stress(state)
        dS_dt = dS_dt_known + dS_dt_nn
        
        # === HRV EQUATION ===
        dHRV_dt_known = (
            -self.beta_hrv * HRV -
            self.gamma_stress_to_hrv * S +
            self.gamma_hrv_homeostasis * (self.hrv_baseline - HRV)
        )
        dHRV_dt_nn = self.nn_hrv(state)
        dHRV_dt = dHRV_dt_known + dHRV_dt_nn
        
        # === EDA EQUATION ===
        dEDA_dt_known = (
            -self.beta_eda * EDA +
            self.gamma_stress_to_eda * S
        )
        dEDA_dt_nn = self.nn_eda(state)
        dEDA_dt = dEDA_dt_known + dEDA_dt_nn
        
        dy_dt = torch.cat([dS_dt, dHRV_dt, dEDA_dt], dim=1)
        return dy_dt
    
    def get_interpretable_params(self):
        """Extract all interpretable parameters"""
        return {
            'beta_stress': self.beta_stress.item(),
            'beta_hrv': self.beta_hrv.item(),
            'beta_eda': self.beta_eda.item(),
            'gamma_hrv_to_stress': self.gamma_hrv_to_stress.item(),
            'gamma_eda_to_stress': self.gamma_eda_to_stress.item(),
            'gamma_interaction': self.gamma_interaction.item(),
            'gamma_stress_to_hrv': self.gamma_stress_to_hrv.item(),
            'gamma_hrv_homeostasis': self.gamma_hrv_homeostasis.item(),
            'gamma_stress_to_eda': self.gamma_stress_to_eda.item(),
            'hrv_baseline': self.hrv_baseline.item(),
            'feedback_S_HRV': (self.gamma_hrv_to_stress * self.gamma_stress_to_hrv).item(),
            'feedback_S_EDA': (self.gamma_eda_to_stress * self.gamma_stress_to_eda).item(),
        }
    
    def get_equations_str(self):
        """Get human-readable equations"""
        p = self.get_interpretable_params()
        
        eq_stress = (
            f"dS/dt = -{p['beta_stress']:.3f}·S "
            f"+ {p['gamma_hrv_to_stress']:.3f}·HRV "
            f"+ {p['gamma_eda_to_stress']:.3f}·EDA "
            f"+ {p['gamma_interaction']:.3f}·S·HRV "
            f"+ NN_S(S,HRV,EDA)"
        )
        eq_hrv = (
            f"dHRV/dt = -{p['beta_hrv']:.3f}·HRV "
            f"- {p['gamma_stress_to_hrv']:.3f}·S "
            f"+ {p['gamma_hrv_homeostasis']:.3f}·({p['hrv_baseline']:.1f} - HRV) "
            f"+ NN_HRV(S,HRV,EDA)"
        )
        eq_eda = (
            f"dEDA/dt = -{p['beta_eda']:.3f}·EDA "
            f"+ {p['gamma_stress_to_eda']:.3f}·S "
            f"+ NN_EDA(S,HRV,EDA)"
        )
        return {'stress': eq_stress, 'hrv': eq_hrv, 'eda': eq_eda}
