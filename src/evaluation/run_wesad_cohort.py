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

def run_wesad_cohort_simulation():
    # Config
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    n_users = 15  # All WESAD subjects
    seq_len = 3000  # Extended from 1000 to 3000 minutes (~50 hours)
    diffusion_coeff = 0.5  # Increased from 0.08 for stronger group influence
    model_path = os.path.join(base_dir, 'results', 'ude_model.pth')
    
    if not os.path.exists(model_path):
        print(f"UDE model not found at {model_path}. Run training first.")
        return

    # 1. Load Data for all WESAD subjects
    # 1. Load Data for all WESAD subjects
    data_dir = os.path.join(base_dir, 'data', 'processed')
    if not os.path.exists(data_dir):
        print(f"Data dir not found: {data_dir}")
        return
    
    files = sorted([f for f in os.listdir(data_dir) if 'wesad' in f.lower() and f.endswith('.csv')])[:n_users]
    
    if len(files) < n_users:
        print(f"Not enough WESAD files. Found {len(files)}, need {n_users}.")
        print(f"Available files: {files}")
        return
        
    all_u = []
    all_y0 = []
    all_features = []
    subject_ids = []
    
    print(f"Loading {len(files)} WESAD subjects...")
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        
        # Ensure we have enough data
        if len(df) < seq_len:
            print(f"  Warning: {f} has only {len(df)} points, padding...")
            # Pad with last value
            padding = pd.DataFrame({
                'time': np.arange(len(df), seq_len) / 60.0,
                'stress': [df['stress'].iloc[-1]] * (seq_len - len(df)),
                'workload': [df['workload'].iloc[-1]] * (seq_len - len(df))
            })
            df = pd.concat([df, padding], ignore_index=True)
        
        u = df['workload'].values[:seq_len]
        y0 = df['stress'].values[0]
        
        # Extract features for grouping
        feat = [df['stress'].mean(), df['workload'].mean(), df['stress'].std()]
        
        all_u.append(u)
        all_y0.append(y0)
        all_features.append(feat)
        subject_ids.append(f.replace('.csv', ''))
        
    # Convert to tensors
    u_tensor = torch.tensor(np.array(all_u), dtype=torch.float32).unsqueeze(-1)
    y0_tensor = torch.tensor(np.array(all_y0), dtype=torch.float32).unsqueeze(-1)
    t_tensor = torch.linspace(0, seq_len-1, seq_len)
    
    # 2. Load UDE
    ude = UDE()
    ude.load_state_dict(torch.load(model_path))
    ude.eval()
    
    # Set context for the WHOLE group
    ude.set_current_batch(t_tensor, u_tensor)
    
    # 3. Setup PDE with COHORT GROUPING (Similarity Graph)
    user_features = np.array(all_features)
    adj = create_similarity_graph(user_features, k=4)  # Each person connected to 4 most similar
    
    print(f"Constructed Cohort Graph with {adj.sum()} edges (Similarity-based).")
    print(f"Average connections per person: {adj.sum() / n_users:.1f}")
    
    # 4. Simulate Coupled (PDE)
    print("Simulating Coupled Group (PDE with Biased Kernel)...")
    pde = StressPDE(n_users, adj, ude, diffusion_coeff=diffusion_coeff)
    
    with torch.no_grad():
        y_coupled = odeint(pde, y0_tensor, t_tensor, method='rk4')
        
    # 5. Simulate Uncoupled (Individual UDEs only)
    print("Simulating Uncoupled Group (Independent)...")
    pde_uncoupled = StressPDE(n_users, adj, ude, diffusion_coeff=0.0)
    
    with torch.no_grad():
        y_uncoupled = odeint(pde_uncoupled, y0_tensor, t_tensor, method='rk4')

    # 6. Visualize Results
    # 6. Visualize Results
    results_dir = os.path.join(base_dir, 'results', 'wesad_cohort')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Average Group Stress
    avg_coupled = y_coupled.mean(dim=1).squeeze().numpy()
    avg_uncoupled = y_uncoupled.mean(dim=1).squeeze().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_tensor, avg_uncoupled, label='Uncoupled (Individual)', linestyle='--', linewidth=2)
    plt.plot(t_tensor, avg_coupled, label='Coupled (Group Influence)', linewidth=2)
    plt.title('WESAD Cohort: Average Stress Evolution')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Average Stress Level')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'average_stress.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'average_stress.png')}")
    plt.close()
    
    # Plot 2: Individual Trajectories (Sample)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i in range(min(6, n_users)):
        axes[i].plot(t_tensor, y_uncoupled[:, i, 0], label='Uncoupled', linestyle='--', alpha=0.7)
        axes[i].plot(t_tensor, y_coupled[:, i, 0], label='Coupled', alpha=0.9)
        axes[i].set_title(f'{subject_ids[i]}')
        axes[i].set_xlabel('Time (min)')
        axes[i].set_ylabel('Stress')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'individual_trajectories.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'individual_trajectories.png')}")
    plt.close()
    
    # Plot 3: Stress Distribution at End
    final_coupled = y_coupled[-1, :, 0].numpy()
    final_uncoupled = y_uncoupled[-1, :, 0].numpy()
    
    plt.figure(figsize=(10, 6))
    x = np.arange(n_users)
    width = 0.35
    plt.bar(x - width/2, final_uncoupled, width, label='Uncoupled', alpha=0.7)
    plt.bar(x + width/2, final_coupled, width, label='Coupled', alpha=0.7)
    plt.xlabel('Subject')
    plt.ylabel('Final Stress Level')
    plt.title('Final Stress Levels: Coupled vs Uncoupled')
    plt.xticks(x, [s.replace('u_wesad_', 'S') for s in subject_ids], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'final_stress_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'final_stress_comparison.png')}")
    plt.close()
    
    # Plot 4: Stress Variance Over Time
    var_coupled = y_coupled.var(dim=1).squeeze().numpy()
    var_uncoupled = y_uncoupled.var(dim=1).squeeze().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_tensor, var_uncoupled, label='Uncoupled (Higher Variance)', linestyle='--')
    plt.plot(t_tensor, var_coupled, label='Coupled (Smoothed by Diffusion)')
    plt.title('Group Stress Variance Over Time')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'stress_variance.png'), dpi=150)
    print(f"Saved: {os.path.join(results_dir, 'stress_variance.png')}")
    plt.close()
    
    # Summary Statistics
    print("\n=== WESAD Cohort Simulation Summary ===")
    print(f"Number of subjects: {n_users}")
    print(f"Simulation duration: {seq_len} minutes")
    print(f"Graph edges: {adj.sum()}")
    print(f"\nFinal Average Stress:")
    print(f"  Uncoupled: {final_uncoupled.mean():.4f} ± {final_uncoupled.std():.4f}")
    print(f"  Coupled:   {final_coupled.mean():.4f} ± {final_coupled.std():.4f}")
    print(f"\nStress Variance Reduction:")
    print(f"  Initial: {var_uncoupled[0]:.4f} -> {var_coupled[0]:.4f} ({(1-var_coupled[0]/var_uncoupled[0])*100:.1f}% reduction)")
    print(f"  Final:   {var_uncoupled[-1]:.4f} -> {var_coupled[-1]:.4f} ({(1-var_coupled[-1]/var_uncoupled[-1])*100:.1f}% reduction)")

if __name__ == "__main__":
    run_wesad_cohort_simulation()
