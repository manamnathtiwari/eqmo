import torch
import torch.nn as nn
import numpy as np

class CoupledUDE(nn.Module):
    """
    Coupled Universal Differential Equation System
    Models interdependent dynamics of Stress, HRV, and EDA
    
    System:
    dS/dt = -β₁·S + γ₁₁·HRV + γ₁₂·EDA + γ₁₃·S·HRV + NN_S(S,HRV,EDA)
    dHRV/dt = -β₂·HRV + γ₂₁·S + γ₂₂·(baseline-HRV) + NN_HRV(S,HRV,EDA)
    dEDA/dt = -β₃·EDA + γ₃₁·S + NN_EDA(S,HRV,EDA)
    """
    
    def __init__(self, selected_features=['HRV', 'EDA'], hidden_size=32):
        super(CoupledUDE, self).__init__()
        
        self.selected_features = selected_features
        self.n_features = len(selected_features)  # Number of features (2-6)
        self.n_vars = 1 + self.n_features  # Stress + features
        
        # Recovery rates (β parameters) - one per variable
        self._beta_stress = nn.Parameter(torch.tensor(-2.0))
        self._beta_features = nn.ParameterList([
            nn.Parameter(torch.tensor(-2.0 - 0.1*i)) for i in range(self.n_features)
        ])
        
        # Coupling coefficients (γ parameters)
        # Features → Stress
        self._gamma_to_stress = nn.ParameterList([
            nn.Parameter(torch.tensor(-2.0 - 0.1*i)) for i in range(self.n_features)
        ])
        
        # Stress → Features
        self._gamma_from_stress = nn.ParameterList([
            nn.Parameter(torch.tensor(-1.8 - 0.1*i)) for i in range(self.n_features)
        ])
        
        # Interaction term (S × F1)
        self._gamma_interaction = nn.Parameter(torch.tensor(-3.0))
        
        # Homeostasis for first feature (e.g., HRV)
        self._gamma_homeostasis = nn.Parameter(torch.tensor(-2.2))
        self.feature_baseline = nn.Parameter(torch.tensor(60.0))
        
        # Neural network corrections
        input_dim = self.n_vars  # [S, F1, F2, ..., Fn]
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
        
        # Baseline values (learned)
        self.hrv_baseline = nn.Parameter(torch.tensor(60.0))
        
        # Current batch data
        self.current_features = None
        self.current_t = None
    
    @property
    def beta_stress(self):
        return torch.nn.functional.softplus(self._beta_stress)
    
    @property
    def beta_hrv(self):
        return torch.nn.functional.softplus(self._beta_hrv)
    
    @property
    def beta_eda(self):
        return torch.nn.functional.softplus(self._beta_eda)
    
    @property
    def gamma_hrv_to_stress(self):
        return torch.nn.functional.softplus(self._gamma_hrv_to_stress)
    
    @property
    def gamma_eda_to_stress(self):
        return torch.nn.functional.softplus(self._gamma_eda_to_stress)
    
    @property
    def gamma_interaction(self):
        return torch.nn.functional.softplus(self._gamma_interaction)
    
    @property
    def gamma_stress_to_hrv(self):
        return torch.nn.functional.softplus(self._gamma_stress_to_hrv)
    
    @property
    def gamma_hrv_homeostasis(self):
        return torch.nn.functional.softplus(self._gamma_hrv_homeostasis)
    
    @property
    def gamma_stress_to_eda(self):
        return torch.nn.functional.softplus(self._gamma_stress_to_eda)
    
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
        batch_size = y.shape[0]
        
        # Extract variables
        S = y[:, 0:1]      # Stress
        HRV = y[:, 1:2]    # HRV
        EDA = y[:, 2:3] if self.n_vars > 2 else torch.zeros_like(S)
        
        # Get additional features if needed (static inputs)
        # features = self.get_features_at_t(t.item()) if needed
        
        # === STRESS EQUATION ===
        dS_dt_known = (
            -self.beta_stress * S +
            self.gamma_hrv_to_stress * HRV +
            self.gamma_eda_to_stress * EDA +
            self.gamma_interaction * S * HRV
        )
        
        # Neural correction
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
        
        # Stack derivatives
        dy_dt = torch.cat([dS_dt, dHRV_dt, dEDA_dt], dim=1)
        
        return dy_dt
    
    def get_interpretable_params(self):
        """Extract all interpretable parameters"""
        return {
            # Recovery rates
            'beta_stress': self.beta_stress.item(),
            'beta_hrv': self.beta_hrv.item(),
            'beta_eda': self.beta_eda.item(),
            
            # Coupling coefficients
            'gamma_hrv_to_stress': self.gamma_hrv_to_stress.item(),
            'gamma_eda_to_stress': self.gamma_eda_to_stress.item(),
            'gamma_interaction': self.gamma_interaction.item(),
            'gamma_stress_to_hrv': self.gamma_stress_to_hrv.item(),
            'gamma_hrv_homeostasis': self.gamma_hrv_homeostasis.item(),
            'gamma_stress_to_eda': self.gamma_stress_to_eda.item(),
            
            # Baselines
            'hrv_baseline': self.hrv_baseline.item(),
            
            # Feedback loop strengths
            'feedback_S_HRV': (self.gamma_hrv_to_stress * self.gamma_stress_to_hrv).item(),
            'feedback_S_EDA': (self.gamma_eda_to_stress * self.gamma_stress_to_eda).item(),
        }
    
    def get_equations_str(self):
        """Get human-readable equations"""
        params = self.get_interpretable_params()
        
        eq_stress = f"""
dS/dt = -{params['beta_stress']:.3f}·S 
        + {params['gamma_hrv_to_stress']:.3f}·HRV 
        + {params['gamma_eda_to_stress']:.3f}·EDA 
        + {params['gamma_interaction']:.3f}·S·HRV 
        + NN_S(S,HRV,EDA)
"""
        
        eq_hrv = f"""
dHRV/dt = -{params['beta_hrv']:.3f}·HRV 
          - {params['gamma_stress_to_hrv']:.3f}·S 
          + {params['gamma_hrv_homeostasis']:.3f}·({params['hrv_baseline']:.1f} - HRV)
          + NN_HRV(S,HRV,EDA)
"""
        
        eq_eda = f"""
dEDA/dt = -{params['beta_eda']:.3f}·EDA 
          + {params['gamma_stress_to_eda']:.3f}·S 
          + NN_EDA(S,HRV,EDA)
"""
        
        return {
            'stress': eq_stress,
            'hrv': eq_hrv,
            'eda': eq_eda
        }
