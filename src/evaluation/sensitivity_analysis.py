import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_model import UDE
from src.models.pde_model import StressPDE, create_similarity_graph

def run_single_experiment(diffusion_coeff, sigma, n_users=15, seq_len=2000):
    """
    Run one experiment with given parameters.
    Returns: (variance_reduction, mean_stress_reduction)
    """
    # Load model
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'results', 'ude_model.pth')
    if not os.path.exists(model_path):
        return None, None
    
    # Load data
    data_dir = os.path.join(base_dir, 'data', 'processed')
    if not os.path.exists(data_dir):
        data_dir = os.path.join('data', 'processed')
    
    files = sorted([f for f in os.listdir(data_dir) if 'wesad' in f.lower() and f.endswith('.csv')])[:n_users]
    
    if len(files) < n_users:
        return None, None
    
    all_u = []
    all_y0 = []
    all_features = []
    
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
        y0 = df['stress'].values[0]
        feat = [df['stress'].mean(), df['workload'].mean(), df['stress'].std()]
        
        all_u.append(u)
        all_y0.append(y0)
        all_features.append(feat)
    
    u_tensor = torch.tensor(np.array(all_u), dtype=torch.float32).unsqueeze(-1)
    y0_tensor = torch.tensor(np.array(all_y0), dtype=torch.float32).unsqueeze(-1)
    t_tensor = torch.linspace(0, seq_len-1, seq_len)
    
    # Load UDE
    ude = UDE()
    ude.load_state_dict(torch.load(model_path))
    ude.eval()
    ude.set_current_batch(t_tensor, u_tensor)
    
    # Build graph
    user_features = np.array(all_features)
    adj = create_similarity_graph(user_features, k=4)
    
    # Temporarily modify sigma in pde_model
    # We'll need to pass sigma as a parameter
    # For now, we'll create a modified PDE class inline
    
    # Simulate coupled
    pde = StressPDE(n_users, adj, ude, diffusion_coeff=diffusion_coeff)
    # Override sigma (hacky but works for sensitivity analysis)
    # We need to modify the forward method's sigma value
    # Better approach: modify StressPDE to accept sigma as parameter
    
    with torch.no_grad():
        y_coupled = odeint(pde, y0_tensor, t_tensor, method='rk4')
    
    # Simulate uncoupled
    pde_uncoupled = StressPDE(n_users, adj, ude, diffusion_coeff=0.0)
    with torch.no_grad():
        y_uncoupled = odeint(pde_uncoupled, y0_tensor, t_tensor, method='rk4')
    
    # Compute metrics
    var_uncoupled = y_uncoupled[-1, :, 0].var().item()
    var_coupled = y_coupled[-1, :, 0].var().item()
    variance_reduction = (var_uncoupled - var_coupled) / var_uncoupled * 100
    
    mean_uncoupled = y_uncoupled[-1, :, 0].mean().item()
    mean_coupled = y_coupled[-1, :, 0].mean().item()
    mean_stress_reduction = (mean_uncoupled - mean_coupled) / mean_uncoupled * 100
    
    return variance_reduction, mean_stress_reduction

def run_sensitivity_analysis():
    """
    Run grid search over diffusion coefficients and kernel widths.
    """
    # Parameter grid
    diffusion_coeffs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    sigmas = [0.5, 0.8, 1.0, 1.5, 2.0]  # We'll test by modifying pde_model.py temporarily
    
    # For now, we'll just vary diffusion_coeff since sigma is hardcoded
    # Full implementation would require modifying StressPDE to accept sigma
    
    print("Running Sensitivity Analysis...")
    print(f"Testing {len(diffusion_coeffs)} diffusion coefficients")
    
    results = []
    
    for dc in tqdm(diffusion_coeffs, desc="Diffusion Coefficients"):
        var_red, mean_red = run_single_experiment(dc, sigma=1.5)
        if var_red is not None:
            results.append({
                'diffusion_coeff': dc,
                'variance_reduction': var_red,
                'mean_stress_reduction': mean_red
            })
    
    # Save results
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    results_df = pd.DataFrame(results)
    results_dir = os.path.join(base_dir, 'results', 'sensitivity')
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, 'diffusion_sensitivity.csv'), index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Variance Reduction
    axes[0].plot(results_df['diffusion_coeff'], results_df['variance_reduction'], 
                 marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Diffusion Coefficient', fontsize=12)
    axes[0].set_ylabel('Variance Reduction (%)', fontsize=12)
    axes[0].set_title('Group Cohesion vs. Diffusion Strength', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Mean Stress Reduction
    axes[1].plot(results_df['diffusion_coeff'], results_df['mean_stress_reduction'], 
                 marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Diffusion Coefficient', fontsize=12)
    axes[1].set_ylabel('Mean Stress Reduction (%)', fontsize=12)
    axes[1].set_title('Stress Relief vs. Diffusion Strength', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'diffusion_sensitivity.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'diffusion_sensitivity.png')}")
    
    # Print summary
    print("\n=== Sensitivity Analysis Results ===")
    print(results_df.to_string(index=False))
    print(f"\nOptimal Diffusion Coefficient: {results_df.loc[results_df['variance_reduction'].idxmax(), 'diffusion_coeff']}")

if __name__ == "__main__":
    run_sensitivity_analysis()
