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
from src.models.pde_model import StressPDE, create_grid_graph, create_similarity_graph

def run_simulation():
    # Config
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    n_users = 15 
    seq_len = 1000
    model_path = os.path.join(base_dir, 'results', 'ude_model.pth')
    
    if not os.path.exists(model_path):
        print(f"UDE model not found at {model_path}. Run training first.")
        return

    # 1. Load Data for N users
    # We'll just take the first N files
    data_dir = os.path.join(base_dir, 'data', 'processed')
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])[:n_users]
    
    if len(files) < n_users:
        print(f"Not enough data files. Found {len(files)}, need {n_users}.")
        return
        
    all_u = []
    all_y0 = []
    all_features = [] # For similarity graph
    
    print("Loading data...")
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        u = df['workload'].values[:seq_len]
        y0 = df['stress'].values[0]
        
        # Extract simple features for grouping: [mean_stress, mean_workload]
        # In a real app, this would be from the 'activity_class' or metadata
        feat = [df['stress'].mean(), df['workload'].mean()]
        
        all_u.append(u)
        all_y0.append(y0)
        all_features.append(feat)
        
    # (n_users, seq_len, 1)
    u_tensor = torch.tensor(np.array(all_u), dtype=torch.float32).unsqueeze(-1)
    # (n_users, 1)
    y0_tensor = torch.tensor(np.array(all_y0), dtype=torch.float32).unsqueeze(-1)
    
    t_tensor = torch.linspace(0, seq_len-1, seq_len)
    
    # 2. Load UDE
    ude = UDE()
    ude.load_state_dict(torch.load(model_path))
    ude.eval()
    
    # Set context for the WHOLE group
    ude.set_current_batch(t_tensor, u_tensor)
    
    # 3. Setup PDE with COHORT GROUPING (Similarity Graph)
    # Use the features we collected
    user_features = np.array(all_features)
    # Create graph based on similarity (k=3 neighbors)
    adj = create_similarity_graph(user_features, k=3)
    
    print(f"Constructed Cohort Graph with {adj.sum()} edges (Similarity-based).")
    
    pde = StressPDE(n_users, adj, ude, diffusion_coeff=0.05)
    
    # 4. Simulate Coupled (PDE)
    print("Simulating Coupled Group (PDE)...")
    with torch.no_grad():
        # odeint expects func(t, y)
        # pde.forward matches this signature
        y_coupled = odeint(pde, y0_tensor, t_tensor, method='rk4') # (seq_len, n_users, 1)
        
    # 5. Simulate Uncoupled (Individual UDEs only)
    print("Simulating Uncoupled Group (Independent)...")
    pde_uncoupled = StressPDE(n_users, adj, ude, diffusion_coeff=0.0)
    with torch.no_grad():
        y_uncoupled = odeint(pde_uncoupled, y0_tensor, t_tensor, method='rk4')

    # 6. Visualize
    # Pick a central user and a corner user to compare
    user_idx = 5 # Some user in the middle
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_tensor, y_uncoupled[:, user_idx, 0], label='Uncoupled (Individual)', linestyle='--')
    plt.plot(t_tensor, y_coupled[:, user_idx, 0], label='Coupled (Group Influence)')
    plt.plot(t_tensor, u_tensor[user_idx, :, 0] * 0.2, label='Workload', alpha=0.2)
    plt.title(f'Effect of Group Diffusion on User {user_idx}')
    plt.xlabel('Time')
    plt.ylabel('Stress')
    plt.legend()
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'pde_comparison.png'))
    print(f"Saved comparison plot to {os.path.join(results_dir, 'pde_comparison.png')}")
    
    # Heatmap animation (optional, saving just start/end for now)
    # Reshape to grid 3x5 (for 15 users)
    stress_end = y_coupled[-1, :, 0].reshape(3, 5).numpy()
    plt.figure()
    plt.imshow(stress_end, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Stress Level')
    plt.colorbar(label='Stress Level')
    plt.title('Group Stress Map at End (t=1000)')
    plt.savefig(os.path.join(results_dir, 'group_heatmap_end.png'))
    print(f"Saved heatmap to {os.path.join(results_dir, 'group_heatmap_end.png')}")

if __name__ == "__main__":
    run_simulation()
