"""
Cohort Optimization & Social Simulation
---------------------------------------
Uses real physiological parameters (Alpha, Beta) extracted from WESAD subjects
to simulate and optimize group stress dynamics.

Key Features:
1. Extracts Alpha (Sensitivity) and Beta (Recovery) from trained UDE models.
2. Clusters users into 'Vulnerable' and 'Resilient' groups.
3. Simulates a 'Virtual Office' with Social Diffusion (PDE).
4. Compares 'High Risk Cohort' vs 'Optimized Cohort' to prove Social Buffering.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_model import UDE

def load_subject_parameters(models_dir, device):
    """Load Alpha/Beta from all trained models"""
    model_files = [f for f in os.listdir(models_dir) if f.startswith('ude_fold_') and f.endswith('.pth')]
    model_files.sort()
    
    params = []
    print(f"Loading parameters from {len(model_files)} subjects...")
    
    for f in model_files:
        # Load model
        model = UDE().to(device)
        model.load_state_dict(torch.load(os.path.join(models_dir, f), map_location=device))
        
        # Extract
        alpha = model.alpha.item() # Sensitivity
        beta = model.beta.item()   # Recovery
        
        # ID
        fold_idx = int(f.split('_')[-1].split('.')[0])
        subject_id = f"S{fold_idx+1}" # S2, S3, etc. (Approximate mapping)
        
        params.append({
            'Subject': subject_id,
            'Alpha': alpha,
            'Beta': beta,
            'Risk': alpha / (beta + 1e-6)
        })
        
    return pd.DataFrame(params)

def simulate_group_dynamics(participants, duration_mins=60, dt=0.1, diffusion_coef=0.5):
    """
    Simulate stress evolution of a group using UDE+PDE dynamics.
    dS_i/dt = -beta_i * S_i + alpha_i * W(t) + D * sum(A_ij * (S_j - S_i))
    """
    n_agents = len(participants)
    steps = int(duration_mins * 60 / dt)
    time = np.linspace(0, duration_mins, steps)
    
    # Initialize State
    S = np.zeros((n_agents, steps))
    S[:, 0] = 0.2 # Initial low stress
    
    # Parameters
    alphas = participants['Alpha'].values
    betas = participants['Beta'].values
    
    # Workload Scenario: High Stress Event (e.g., Deadline)
    # Workload = 1.0 from t=10 to t=40
    W = np.zeros(steps)
    W[int(10*60/dt) : int(40*60/dt)] = 1.0
    
    # Adjacency Matrix (Fully Connected Office)
    A = np.ones((n_agents, n_agents)) - np.eye(n_agents)
    # Normalize by degree
    D_matrix = A / (n_agents - 1)
    
    # Simulation Loop
    for t in range(steps - 1):
        current_S = S[:, t]
        current_W = W[t]
        
        # 1. Individual Dynamics (UDE)
        # dS = -beta * S + alpha * W
        dS_ind = -betas * current_S + alphas * current_W
        
        # 2. Social Dynamics (PDE/Diffusion)
        # dS_soc = D * sum(S_j - S_i)
        # Vectorized: D * (A @ S - S)
        # Using normalized D_matrix: mean field influence
        avg_S = np.mean(current_S)
        dS_soc = diffusion_coef * (avg_S - current_S)
        
        # Update
        S[:, t+1] = current_S + (dS_ind + dS_soc) * dt
        
        # Clamp to [0, 1]
        S[:, t+1] = np.clip(S[:, t+1], 0, 1)
        
    return time, S, W

def run_cohort_optimization():
    print("="*70)
    print("SMART COHORT OPTIMIZATION (Virtual Office Simulation)")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, 'results', 'loso_models')
    out_dir = os.path.join(base_dir, 'results', 'simulation')
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device('cpu') # Simulation is fast on CPU
    
    # 1. Load Real Parameters
    df = load_subject_parameters(models_dir, device)
    
    if len(df) < 2:
        print("Error: Need at least 2 trained models to run simulation.")
        return

    # 2. Identify Groups
    # Median split on Risk
    median_risk = df['Risk'].median()
    df['Group'] = df['Risk'].apply(lambda x: 'Vulnerable' if x > median_risk else 'Resilient')
    
    vulnerable = df[df['Group'] == 'Vulnerable']
    resilient = df[df['Group'] == 'Resilient']
    
    print(f"\nIdentified Groups:")
    print(f"  Vulnerable (High Risk): {len(vulnerable)} subjects")
    print(f"  Resilient (Low Risk):   {len(resilient)} subjects")
    
    # 3. Scenario A: The "Burnout Factory" (All Vulnerable)
    # If we don't have enough vulnerable, duplicate them
    cohort_A = pd.concat([vulnerable] * 3).iloc[:5] # Create a team of 5
    print(f"\nSimulating Scenario A: High Risk Cohort (5 Vulnerable)")
    
    t, S_A, W = simulate_group_dynamics(cohort_A, diffusion_coef=0.5)
    mean_stress_A = np.mean(S_A, axis=0)
    peak_stress_A = np.max(mean_stress_A)
    
    # 4. Scenario B: The "Optimized Team" (Mixed)
    # Mix 2 Vulnerable + 3 Resilient
    cohort_B = pd.concat([vulnerable.iloc[:2], resilient.iloc[:3]])
    # If not enough, just mix whatever we have
    if len(cohort_B) < 5:
         cohort_B = pd.concat([vulnerable, resilient]).iloc[:5]
         
    print(f"Simulating Scenario B: Optimized Cohort ({len(cohort_B)} Mixed)")
    
    t, S_B, W = simulate_group_dynamics(cohort_B, diffusion_coef=0.5)
    mean_stress_B = np.mean(S_B, axis=0)
    peak_stress_B = np.max(mean_stress_B)
    
    # 5. Results
    reduction = (peak_stress_A - peak_stress_B) / peak_stress_A * 100
    print(f"\nRESULTS:")
    print(f"  Peak Stress (High Risk Team): {peak_stress_A:.4f}")
    print(f"  Peak Stress (Optimized Team): {peak_stress_B:.4f}")
    print(f"  Stress Reduction: {reduction:.2f}%")
    
    # 6. Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot Workload
    plt.fill_between(t, 0, W*0.2, color='gray', alpha=0.1, label='Workload Event')
    
    # Plot Team A
    plt.plot(t, mean_stress_A, 'r-', linewidth=3, label=f'High Risk Team (Peak: {peak_stress_A:.2f})')
    # Plot Team B
    plt.plot(t, mean_stress_B, 'g-', linewidth=3, label=f'Optimized Team (Peak: {peak_stress_B:.2f})')
    
    plt.title(f'Social Buffering Effect: {reduction:.1f}% Stress Reduction', fontsize=14, fontweight='bold')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Average Team Stress')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(out_dir, 'cohort_optimization.png'), dpi=150)
    print(f"\nPlot saved to: {os.path.join(out_dir, 'cohort_optimization.png')}")
    
    # Save stats
    with open(os.path.join(out_dir, 'simulation_results.txt'), 'w') as f:
        f.write(f"Scenario A Peak: {peak_stress_A}\n")
        f.write(f"Scenario B Peak: {peak_stress_B}\n")
        f.write(f"Reduction: {reduction}%\n")

if __name__ == "__main__":
    run_cohort_optimization()
