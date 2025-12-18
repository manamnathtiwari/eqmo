import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_model import UDE
from src.models.pde_model import StressPDE, create_similarity_graph
from src.models.dynamic_pde_model import DynamicStressPDE

def run_dynamic_graph_comparison():
    """
    Compare static vs. dynamic graph models.
    """
    # Config
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    n_users = 15
    seq_len = 3000
    diffusion_coeff = 0.5
    model_path = os.path.join(base_dir, 'results', 'ude_model.pth')
    
    if not os.path.exists(model_path):
        print("UDE model not found.")
        return
    
    # Load data
    data_dir = os.path.join(base_dir, 'data', 'processed')
    if not os.path.exists(data_dir):
        data_dir = os.path.join('data', 'processed')
    
    files = sorted([f for f in os.listdir(data_dir) if 'wesad' in f.lower() and f.endswith('.csv')])[:n_users]
    
    if len(files) < n_users:
        print(f"Not enough files. Found {len(files)}")
        return
    
    all_u = []
    all_y0 = []
    all_features = []
    subject_ids = []
    
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
        y0 = df['stress'].values[0]
        feat = [df['stress'].mean(), df['workload'].mean(), df['stress'].std()]
        
        all_u.append(u)
        all_y0.append(y0)
        all_features.append(feat)
        subject_ids.append(f.replace('.csv', ''))
    
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
    
    print(f"Graph: {adj.sum()} edges")
    
    # 1. Static Graph PDE
    print("Simulating Static Graph PDE...")
    pde_static = StressPDE(n_users, adj, ude, diffusion_coeff=diffusion_coeff)
    with torch.no_grad():
        y_static = odeint(pde_static, y0_tensor, t_tensor, method='rk4')
    
    # 2. Dynamic Graph PDE
    print("Simulating Dynamic Graph PDE (with withdrawal)...")
    pde_dynamic = DynamicStressPDE(n_users, adj, ude, diffusion_coeff=diffusion_coeff,
                                   withdrawal_threshold=0.7, withdrawal_strength=0.6)
    with torch.no_grad():
        y_dynamic = odeint(pde_dynamic, y0_tensor, t_tensor, method='rk4')
    
    # 3. Uncoupled
    print("Simulating Uncoupled (baseline)...")
    pde_uncoupled = StressPDE(n_users, adj, ude, diffusion_coeff=0.0)
    with torch.no_grad():
        y_uncoupled = odeint(pde_uncoupled, y0_tensor, t_tensor, method='rk4')
    
    # Visualizations
    # Visualizations
    results_dir = os.path.join(base_dir, 'results', 'dynamic_graph')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Average Stress Comparison
    avg_uncoupled = y_uncoupled.mean(dim=1).squeeze().numpy()
    avg_static = y_static.mean(dim=1).squeeze().numpy()
    avg_dynamic = y_dynamic.mean(dim=1).squeeze().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_tensor, avg_uncoupled, label='Uncoupled (No Group)', linestyle=':', linewidth=2, alpha=0.7)
    plt.plot(t_tensor, avg_static, label='Static Graph', linestyle='--', linewidth=2)
    plt.plot(t_tensor, avg_dynamic, label='Dynamic Graph (Withdrawal)', linewidth=2.5)
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Average Stress', fontsize=12)
    plt.title('Dynamic vs. Static Graph: Average Stress Evolution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'average_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'average_comparison.png')}")
    plt.close()
    
    # Plot 2: Variance Over Time
    var_uncoupled = y_uncoupled.var(dim=1).squeeze().numpy()
    var_static = y_static.var(dim=1).squeeze().numpy()
    var_dynamic = y_dynamic.var(dim=1).squeeze().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_tensor, var_uncoupled, label='Uncoupled', linestyle=':', linewidth=2, alpha=0.7)
    plt.plot(t_tensor, var_static, label='Static Graph', linestyle='--', linewidth=2)
    plt.plot(t_tensor, var_dynamic, label='Dynamic Graph', linewidth=2.5)
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Stress Variance', fontsize=12)
    plt.title('Group Cohesion: Variance Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'variance_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'variance_comparison.png')}")
    plt.close()
    
    # Plot 3: Individual Trajectories (Sample)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(6, n_users)):
        axes[i].plot(t_tensor, y_uncoupled[:, i, 0], label='Uncoupled', linestyle=':', alpha=0.6)
        axes[i].plot(t_tensor, y_static[:, i, 0], label='Static', linestyle='--', alpha=0.8)
        axes[i].plot(t_tensor, y_dynamic[:, i, 0], label='Dynamic', linewidth=2)
        axes[i].set_title(f'{subject_ids[i]}', fontsize=10)
        axes[i].set_xlabel('Time (min)', fontsize=9)
        axes[i].set_ylabel('Stress', fontsize=9)
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'individual_trajectories.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'individual_trajectories.png')}")
    plt.close()
    
    # Summary Statistics
    print("\n=== Dynamic Graph Comparison ===")
    print(f"\nFinal Average Stress:")
    print(f"  Uncoupled: {avg_uncoupled[-1]:.4f}")
    print(f"  Static:    {avg_static[-1]:.4f} ({(avg_static[-1]-avg_uncoupled[-1])/avg_uncoupled[-1]*100:+.1f}%)")
    print(f"  Dynamic:   {avg_dynamic[-1]:.4f} ({(avg_dynamic[-1]-avg_uncoupled[-1])/avg_uncoupled[-1]*100:+.1f}%)")
    
    print(f"\nFinal Variance:")
    print(f"  Uncoupled: {var_uncoupled[-1]:.4f}")
    print(f"  Static:    {var_static[-1]:.4f} ({(1-var_static[-1]/var_uncoupled[-1])*100:.1f}% reduction)")
    print(f"  Dynamic:   {var_dynamic[-1]:.4f} ({(1-var_dynamic[-1]/var_uncoupled[-1])*100:.1f}% reduction)")
    
    # Save results
    results = pd.DataFrame({
        'Model': ['Uncoupled', 'Static Graph', 'Dynamic Graph'],
        'Final_Mean_Stress': [avg_uncoupled[-1], avg_static[-1], avg_dynamic[-1]],
        'Final_Variance': [var_uncoupled[-1], var_static[-1], var_dynamic[-1]],
        'Variance_Reduction_%': [0, (1-var_static[-1]/var_uncoupled[-1])*100, (1-var_dynamic[-1]/var_uncoupled[-1])*100]
    })
    results.to_csv(os.path.join(results_dir, 'comparison_summary.csv'), index=False)
    print(f"\nSaved: {os.path.join(results_dir, 'comparison_summary.csv')}")

if __name__ == "__main__":
    run_dynamic_graph_comparison()
