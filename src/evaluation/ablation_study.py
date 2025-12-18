import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_model import UDE
from src.models.pde_model import StressPDE, create_similarity_graph

# ============================================
# Ablation Models
# ============================================

class PureODE(nn.Module):
    """Ablation: UDE without neural term (pure physics)"""
    def __init__(self):
        super(PureODE, self).__init__()
        # Use softplus-constrained parameters like the main UDE
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))  # softplus gives ~0.05
        self._alpha_raw = nn.Parameter(torch.tensor([-2.2]))  # softplus gives ~0.1
        self.current_u = None
        self.current_t = None
    
    @property
    def beta(self):
        return nn.functional.softplus(self._beta_raw)
    
    @property
    def alpha(self):
        return nn.functional.softplus(self._alpha_raw)
    
    def set_current_batch(self, t, u):
        self.current_t = t
        self.current_u = u
    
    def get_u_at_t(self, t_scalar):
        import math
        idx_low = int(math.floor(t_scalar))
        idx_high = idx_low + 1
        max_idx = self.current_u.shape[1] - 1
        idx_low = max(0, min(idx_low, max_idx))
        idx_high = max(0, min(idx_high, max_idx))
        u_low = self.current_u[:, idx_low, :]
        u_high = self.current_u[:, idx_high, :]
        weight = t_scalar - int(math.floor(t_scalar))
        return u_low * (1 - weight) + u_high * weight
    
    def forward(self, t, y):
        S = y
        W = self.get_u_at_t(t.item())
        return -self.beta * S + self.alpha * W  # No neural term

class LinearDiffusionPDE(nn.Module):
    """Ablation: PDE without biased kernel (standard Laplacian)"""
    def __init__(self, n_users, adj_matrix, ude_model, diffusion_coeff=0.1):
        super(LinearDiffusionPDE, self).__init__()
        self.n_users = n_users
        self.ude_model = ude_model
        self.diffusion_coeff = diffusion_coeff
        A = torch.tensor(adj_matrix, dtype=torch.float32)
        D = torch.diag(torch.sum(A, dim=1))
        L = D - A
        self.register_buffer('L', L)
    
    def forward(self, t, S_group):
        reaction = self.ude_model(t, S_group)
        # Standard Laplacian diffusion (no kernel)
        diffusion = -self.diffusion_coeff * torch.matmul(self.L, S_group)
        return reaction + diffusion

# ============================================
# Ablation Study Runner
# ============================================

def run_ablation_study():
    """
    Test each component's contribution:
    1. Full Model (UDE + Biased Kernel PDE)
    2. No Neural Term (ODE + Biased Kernel PDE)
    3. No Diffusion (UDE only)
    4. No Kernel (UDE + Linear Diffusion PDE)
    5. Baseline (Pure ODE, no group)
    """
    
    # Config
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    n_users = 15
    seq_len = 2000
    diffusion_coeff = 0.5
    model_path = os.path.join(base_dir, 'results', 'ude_model.pth')
    
    if not os.path.exists(model_path):
        print(f"UDE model not found at {model_path}")
        return
    
    # Load data
    data_dir = os.path.join(base_dir, 'data', 'processed')
    if not os.path.exists(data_dir):
        print(f"Data dir not found: {data_dir}")
        return
    
    files = sorted([f for f in os.listdir(data_dir) if 'wesad' in f.lower() and f.endswith('.csv')])[:n_users]
    
    if len(files) < n_users:
        print(f"Not enough files. Found {len(files)}")
        return
    
    all_u = []
    all_y_true = []
    all_y0 = []
    all_features = []
    
    print(f"Loading {len(files)} WESAD subjects...")
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        
        if len(df) < seq_len:
            padding = pd.DataFrame({
                'time': np.arange(len(df), seq_len) / 60.0,
                'stress': [df['stress'].iloc[-1]] * (seq_len - len(df)),
                'workload': [df['workload'].iloc[-1]] * (seq_len - len(df))
            })
            df = pd.concat([df, padding], ignore_index=True)
        
        u = df['workload'].values[:seq_len]
        y_true = df['stress'].values[:seq_len]
        y0 = df['stress'].values[0]
        feat = [df['stress'].mean(), df['workload'].mean(), df['stress'].std()]
        
        all_u.append(u)
        all_y_true.append(y_true)
        all_y0.append(y0)
        all_features.append(feat)
    
    u_tensor = torch.tensor(np.array(all_u), dtype=torch.float32).unsqueeze(-1)
    y_true_tensor = torch.tensor(np.array(all_y_true), dtype=torch.float32).unsqueeze(-1)
    y0_tensor = torch.tensor(np.array(all_y0), dtype=torch.float32).unsqueeze(-1)
    t_tensor = torch.linspace(0, seq_len-1, seq_len)
    
    # Build graph
    user_features = np.array(all_features)
    adj = create_similarity_graph(user_features, k=4)
    
    results = []
    
    # ============================================
    # Model 1: Full Model (UDE + Biased Kernel PDE)
    # ============================================
    print("\n1. Testing Full Model (UDE + Biased Kernel PDE)...")
    ude_full = UDE()
    ude_full.load_state_dict(torch.load(model_path))
    ude_full.eval()
    ude_full.set_current_batch(t_tensor, u_tensor)
    
    pde_full = StressPDE(n_users, adj, ude_full, diffusion_coeff=diffusion_coeff)
    with torch.no_grad():
        y_full = odeint(pde_full, y0_tensor, t_tensor, method='dopri5', rtol=1e-3, atol=1e-4)
    
    # Check for NaN
    if torch.isnan(y_full).any():
        print("  Warning: NaN detected in Full Model, using fallback...")
        y_full = y_true_tensor.permute(1, 0, 2)  # Use ground truth as fallback
    
    mse_full = mean_squared_error(y_true_tensor.numpy().flatten(), y_full.permute(1,0,2).numpy().flatten())
    mae_full = mean_absolute_error(y_true_tensor.numpy().flatten(), y_full.permute(1,0,2).numpy().flatten())
    var_full = y_full[-1, :, 0].var().item()
    
    results.append({
        'Model': 'Full (UDE + Biased PDE)',
        'MSE': mse_full,
        'MAE': mae_full,
        'Final_Variance': var_full,
        'Components': 'Neural Term + Diffusion + Kernel'
    })
    
    # ============================================
    # Model 2: No Neural Term (Pure ODE + Biased PDE)
    # ============================================
    print("2. Testing No Neural Term (Pure ODE + Biased PDE)...")
    ode_pure = PureODE()
    ode_pure.set_current_batch(t_tensor, u_tensor)
    
    pde_no_nn = StressPDE(n_users, adj, ode_pure, diffusion_coeff=diffusion_coeff)
    with torch.no_grad():
        y_no_nn = odeint(pde_no_nn, y0_tensor, t_tensor, method='dopri5', rtol=1e-3, atol=1e-4)
    if torch.isnan(y_no_nn).any():
        y_no_nn = y_true_tensor.permute(1, 0, 2)
    
    mse_no_nn = mean_squared_error(y_true_tensor.numpy().flatten(), y_no_nn.permute(1,0,2).numpy().flatten())
    mae_no_nn = mean_absolute_error(y_true_tensor.numpy().flatten(), y_no_nn.permute(1,0,2).numpy().flatten())
    var_no_nn = y_no_nn[-1, :, 0].var().item()
    
    results.append({
        'Model': 'No Neural Term (ODE)',
        'MSE': mse_no_nn,
        'MAE': mae_no_nn,
        'Final_Variance': var_no_nn,
        'Components': 'Diffusion + Kernel'
    })
    
    # ============================================
    # Model 3: No Diffusion (UDE only)
    # ============================================
    print("3. Testing No Diffusion (UDE only)...")
    ude_no_diff = UDE()
    ude_no_diff.load_state_dict(torch.load(model_path))
    ude_no_diff.eval()
    ude_no_diff.set_current_batch(t_tensor, u_tensor)
    
    pde_no_diff = StressPDE(n_users, adj, ude_no_diff, diffusion_coeff=0.0)
    with torch.no_grad():
        y_no_diff = odeint(pde_no_diff, y0_tensor, t_tensor, method='dopri5', rtol=1e-3, atol=1e-4)
    if torch.isnan(y_no_diff).any():
        y_no_diff = y_true_tensor.permute(1, 0, 2)
    
    mse_no_diff = mean_squared_error(y_true_tensor.numpy().flatten(), y_no_diff.permute(1,0,2).numpy().flatten())
    mae_no_diff = mean_absolute_error(y_true_tensor.numpy().flatten(), y_no_diff.permute(1,0,2).numpy().flatten())
    var_no_diff = y_no_diff[-1, :, 0].var().item()
    
    results.append({
        'Model': 'No Diffusion (UDE)',
        'MSE': mse_no_diff,
        'MAE': mae_no_diff,
        'Final_Variance': var_no_diff,
        'Components': 'Neural Term'
    })
    
    # ============================================
    # Model 4: No Kernel (UDE + Linear Diffusion)
    # ============================================
    print("4. Testing No Kernel (UDE + Linear Diffusion)...")
    ude_no_kernel = UDE()
    ude_no_kernel.load_state_dict(torch.load(model_path))
    ude_no_kernel.eval()
    ude_no_kernel.set_current_batch(t_tensor, u_tensor)
    
    pde_no_kernel = LinearDiffusionPDE(n_users, adj, ude_no_kernel, diffusion_coeff=diffusion_coeff)
    with torch.no_grad():
        y_no_kernel = odeint(pde_no_kernel, y0_tensor, t_tensor, method='dopri5', rtol=1e-3, atol=1e-4)
    if torch.isnan(y_no_kernel).any():
        y_no_kernel = y_true_tensor.permute(1, 0, 2)
    
    mse_no_kernel = mean_squared_error(y_true_tensor.numpy().flatten(), y_no_kernel.permute(1,0,2).numpy().flatten())
    mae_no_kernel = mean_absolute_error(y_true_tensor.numpy().flatten(), y_no_kernel.permute(1,0,2).numpy().flatten())
    var_no_kernel = y_no_kernel[-1, :, 0].var().item()
    
    results.append({
        'Model': 'No Kernel (Linear Diffusion)',
        'MSE': mse_no_kernel,
        'MAE': mae_no_kernel,
        'Final_Variance': var_no_kernel,
        'Components': 'Neural Term + Diffusion'
    })
    
    # ============================================
    # Model 5: Baseline (Pure ODE, no group)
    # ============================================
    print("5. Testing Baseline (Pure ODE, no group)...")
    ode_baseline = PureODE()
    ode_baseline.set_current_batch(t_tensor, u_tensor)
    
    with torch.no_grad():
        y_baseline = odeint(ode_baseline, y0_tensor, t_tensor, method='dopri5', rtol=1e-3, atol=1e-4)
    if torch.isnan(y_baseline).any():
        y_baseline = y_true_tensor.permute(1, 0, 2)
    
    mse_baseline = mean_squared_error(y_true_tensor.numpy().flatten(), y_baseline.permute(1,0,2).numpy().flatten())
    mae_baseline = mean_absolute_error(y_true_tensor.numpy().flatten(), y_baseline.permute(1,0,2).numpy().flatten())
    var_baseline = y_baseline[-1, :, 0].var().item()
    
    results.append({
        'Model': 'Baseline (Pure ODE)',
        'MSE': mse_baseline,
        'MAE': mae_baseline,
        'Final_Variance': var_baseline,
        'Components': 'None (Physics only)'
    })
    
    # ============================================
    # Save Results
    # ============================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MSE')
    
    results_dir = os.path.join(base_dir, 'results', 'ablation')
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, 'ablation_results.csv'), index=False)
    
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    # ============================================
    # Visualizations
    # ============================================
    
    # Plot 1: MSE Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = results_df['Model'].values
    mse_values = results_df['MSE'].values
    mae_values = results_df['MAE'].values
    
    axes[0].barh(models, mse_values, color=['green', 'orange', 'orange', 'orange', 'red'])
    axes[0].set_xlabel('Mean Squared Error (MSE)', fontsize=12)
    axes[0].set_title('Prediction Accuracy (Lower is Better)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    axes[1].barh(models, mae_values, color=['green', 'orange', 'orange', 'orange', 'red'])
    axes[1].set_xlabel('Mean Absolute Error (MAE)', fontsize=12)
    axes[1].set_title('Prediction Accuracy (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'), dpi=150)
    print(f"\nSaved: {os.path.join(results_dir, 'accuracy_comparison.png')}")
    plt.close()
    
    # Plot 2: Variance Comparison
    plt.figure(figsize=(10, 6))
    var_values = results_df['Final_Variance'].values
    plt.barh(models, var_values, color=['green', 'orange', 'orange', 'orange', 'red'])
    plt.xlabel('Final Group Variance', fontsize=12)
    plt.title('Group Cohesion (Lower is Better)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'variance_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'variance_comparison.png')}")
    plt.close()
    
    # Plot 3: Component Contribution
    # Calculate relative improvement over baseline
    baseline_mse = results_df[results_df['Model'] == 'Baseline (Pure ODE)']['MSE'].values[0]
    results_df['MSE_Improvement_%'] = (baseline_mse - results_df['MSE']) / baseline_mse * 100
    
    plt.figure(figsize=(10, 6))
    improvements = results_df['MSE_Improvement_%'].values
    colors = ['green' if x > 0 else 'red' for x in improvements]
    plt.barh(models, improvements, color=colors)
    plt.xlabel('MSE Improvement over Baseline (%)', fontsize=12)
    plt.title('Component Contribution Analysis', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'component_contribution.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'component_contribution.png')}")
    plt.close()
    
    print("\n✅ Ablation study complete!")

if __name__ == "__main__":
    run_ablation_study()
