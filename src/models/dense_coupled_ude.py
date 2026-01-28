import torch
import torch.nn as nn
import numpy as np

class DenseCoupledUDE(nn.Module):
    """
    Dense Coupled Universal Differential Equation System
    Uses ALL 18 features with full coupling between stress and key physiological variables
    
    System models:
    - Stress (S)
    - HRV (primary cardiac indicator)
    - EDA (primary arousal indicator)
    - Workload (cognitive load)
    - Plus 14 other features as inputs
    
    Total: 18 features, 4 coupled variables
    """
    
    def __init__(self, n_features=18, hidden_size=64):
        super(DenseCoupledUDE, self).__init__()
        
        self.n_features = n_features
        self.n_coupled = 4  # Stress, HRV, EDA, Workload (main coupled variables)
        
        # Recovery rates (β parameters) for coupled variables
        self._beta_stress = nn.Parameter(torch.tensor(-2.0))
        self._beta_hrv = nn.Parameter(torch.tensor(-2.3))
        self._beta_eda = nn.Parameter(torch.tensor(-1.5))
        self._beta_workload = nn.Parameter(torch.tensor(-2.1))
        
        # Coupling matrix: how each coupled variable affects others
        # Shape: (4, 4) for [S, HRV, EDA, Workload]
        self._coupling_matrix = nn.Parameter(torch.randn(4, 4) * 0.1)
        
        # Feature sensitivities: how all 18 features affect each coupled variable
        # Shape: (4, 18) - each coupled var has sensitivity to all 18 features
        self._feature_sensitivities = nn.Parameter(torch.randn(4, n_features) * 0.1)
        
        # Neural network for residual dynamics
        # Input: 4 coupled vars + 18 features = 22 total
        input_dim = self.n_coupled + n_features
        
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
        
        self.nn_workload = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Baselines
        self.hrv_baseline = nn.Parameter(torch.tensor(60.0))
        self.eda_baseline = nn.Parameter(torch.tensor(0.5))
    
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
    def beta_workload(self):
        return torch.nn.functional.softplus(self._beta_workload)
    
    @property
    def coupling_matrix(self):
        return torch.tanh(self._coupling_matrix)  # Bounded coupling
    
    @property
    def feature_sensitivities(self):
        return torch.tanh(self._feature_sensitivities)  # Bounded sensitivities
    
    def forward(self, t, y, features):
        """
        y: (batch, 4) - [Stress, HRV, EDA, Workload]
        features: (batch, 18) - all 18 physiological features
        
        Returns: dy/dt (batch, 4)
        """
        batch_size = y.shape[0]
        
        # Extract coupled variables
        S = y[:, 0:1]
        HRV = y[:, 1:2]
        EDA = y[:, 2:3]
        WL = y[:, 3:4]
        
        # Stack for coupling
        coupled_vars = torch.cat([S, HRV, EDA, WL], dim=1)  # (batch, 4)
        
        # Full state for neural networks
        full_state = torch.cat([coupled_vars, features], dim=1)  # (batch, 22)
        
        # === STRESS EQUATION ===
        # Recovery
        dS_recovery = -self.beta_stress * S
        
        # Coupling from other variables
        coupling_S = self.coupling_matrix[0:1, :] @ coupled_vars.T  # (1, batch)
        dS_coupling = coupling_S.T
        
        # Feature effects
        feature_effect_S = (self.feature_sensitivities[0:1, :] @ features.T).T
        
        # Neural correction
        dS_nn = self.nn_stress(full_state)
        
        dS_dt = dS_recovery + dS_coupling + feature_effect_S + dS_nn
        
        # === HRV EQUATION ===
        dHRV_recovery = -self.beta_hrv * HRV
        coupling_HRV = (self.coupling_matrix[1:2, :] @ coupled_vars.T).T
        feature_effect_HRV = (self.feature_sensitivities[1:2, :] @ features.T).T
        homeostasis_HRV = 0.1 * (self.hrv_baseline - HRV)
        dHRV_nn = self.nn_hrv(full_state)
        
        dHRV_dt = dHRV_recovery + coupling_HRV + feature_effect_HRV + homeostasis_HRV + dHRV_nn
        
        # === EDA EQUATION ===
        dEDA_recovery = -self.beta_eda * EDA
        coupling_EDA = (self.coupling_matrix[2:3, :] @ coupled_vars.T).T
        feature_effect_EDA = (self.feature_sensitivities[2:3, :] @ features.T).T
        homeostasis_EDA = 0.1 * (self.eda_baseline - EDA)
        dEDA_nn = self.nn_eda(full_state)
        
        dEDA_dt = dEDA_recovery + coupling_EDA + feature_effect_EDA + homeostasis_EDA + dEDA_nn
        
        # === WORKLOAD EQUATION ===
        dWL_recovery = -self.beta_workload * WL
        coupling_WL = (self.coupling_matrix[3:4, :] @ coupled_vars.T).T
        feature_effect_WL = (self.feature_sensitivities[3:4, :] @ features.T).T
        dWL_nn = self.nn_workload(full_state)
        
        dWL_dt = dWL_recovery + coupling_WL + feature_effect_WL + dWL_nn
        
        # Stack derivatives
        dy_dt = torch.cat([dS_dt, dHRV_dt, dEDA_dt, dWL_dt], dim=1)
        
        return dy_dt
    
    def get_interpretable_params(self):
        """Extract all interpretable parameters"""
        coupling = self.coupling_matrix.detach().cpu().numpy()
        sensitivities = self.feature_sensitivities.detach().cpu().numpy()
        
        return {
            # Recovery rates
            'beta_stress': self.beta_stress.item(),
            'beta_hrv': self.beta_hrv.item(),
            'beta_eda': self.beta_eda.item(),
            'beta_workload': self.beta_workload.item(),
            
            # Coupling matrix (4x4)
            'coupling_S_to_S': coupling[0, 0],
            'coupling_HRV_to_S': coupling[0, 1],
            'coupling_EDA_to_S': coupling[0, 2],
            'coupling_WL_to_S': coupling[0, 3],
            
            'coupling_S_to_HRV': coupling[1, 0],
            'coupling_HRV_to_HRV': coupling[1, 1],
            'coupling_EDA_to_HRV': coupling[1, 2],
            'coupling_WL_to_HRV': coupling[1, 3],
            
            'coupling_S_to_EDA': coupling[2, 0],
            'coupling_HRV_to_EDA': coupling[2, 1],
            'coupling_EDA_to_EDA': coupling[2, 2],
            'coupling_WL_to_EDA': coupling[2, 3],
            
            'coupling_S_to_WL': coupling[3, 0],
            'coupling_HRV_to_WL': coupling[3, 1],
            'coupling_EDA_to_WL': coupling[3, 2],
            'coupling_WL_to_WL': coupling[3, 3],
            
            # Feature sensitivities (4x18 = 72 parameters)
            'feature_sensitivities': sensitivities,
            
            # Baselines
            'hrv_baseline': self.hrv_baseline.item(),
            'eda_baseline': self.eda_baseline.item(),
            
            # Feedback loops
            'feedback_S_HRV': coupling[0, 1] * coupling[1, 0],
            'feedback_S_EDA': coupling[0, 2] * coupling[2, 0],
            'feedback_S_WL': coupling[0, 3] * coupling[3, 0],
        }
    
    def get_equations_str(self):
        """Get human-readable equations"""
        params = self.get_interpretable_params()
        sens = params['feature_sensitivities']
        
        # Get top 5 features for each variable
        feature_names = [
            'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50', 'hrv_lf_hf',
            'hr_mean', 'hr_std', 'eda_mean', 'eda_std', 'eda_peaks',
            'temp_mean', 'temp_std', 'resp_mean', 'resp_std',
            'activity_mean', 'activity_std', 'emg_mean', 'emg_std', 'workload'
        ]
        
        eq_stress = f"""
dS/dt = -{params['beta_stress']:.3f}·S
        + {params['coupling_HRV_to_S']:.3f}·HRV
        + {params['coupling_EDA_to_S']:.3f}·EDA  
        + {params['coupling_WL_to_S']:.3f}·Workload
        + Σ(γᵢ·Fᵢ) [18 features]
        + NN(S,HRV,EDA,WL,F₁...F₁₈)
"""
        
        eq_hrv = f"""
dHRV/dt = -{params['beta_hrv']:.3f}·HRV
          + {params['coupling_S_to_HRV']:.3f}·S
          + {params['coupling_EDA_to_HRV']:.3f}·EDA
          + {params['coupling_WL_to_HRV']:.3f}·Workload
          + 0.1·({params['hrv_baseline']:.1f} - HRV)
          + Σ(γᵢ·Fᵢ) [18 features]
          + NN(...)
"""
        
        return {
            'stress': eq_stress,
            'hrv': eq_hrv,
            'summary': f"Dense Coupled UDE: 4 coupled variables × 18 features = 72 feature sensitivities + 16 coupling terms"
        }
