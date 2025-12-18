"""
Visualize Discovered Equations vs. Actual Predictions
Shows: Ground Truth, UDE Prediction, Symbolic Approximation
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.ude_model import UDE
from src.utils import StressDataset
from torch.utils.data import DataLoader
from torchdiffeq import odeint

def symbolic_ode(t, y, alpha, beta, symbolic_terms):
    """
    ODE using only the discovered symbolic equation (no neural network)
    
    symbolic_terms: dict of coefficients like {'S': -0.29, 'W': 0.01, 'S^2': -0.03}
    """
    S = y[0]
    # We need W at this time - for now use a dummy constant
    # In real use, we'd need the workload schedule
    W = 0.5  # Placeholder
    
    # Physics terms
    dS = -beta * S + alpha * W
    
    # Add symbolic correction
    if 'S' in symbolic_terms:
        dS += symbolic_terms['S'] * S
    if 'W' in symbolic_terms:
        dS += symbolic_terms['W'] * W
    if 'S^2' in symbolic_terms:
        dS += symbolic_terms['S^2'] * (S**2)
    if 'S^2 W' in symbolic_terms:
        dS += symbolic_terms['S^2 W'] * (S**2 * W)
    
    return torch.tensor([dS])

def parse_correction_terms(correction_str):
    """Parse correction string like '-0.2893*S +0.0073*W -0.0315*S^2' into dict"""
    terms = {}
    if correction_str == '0':
        return terms
    
    # Split by spaces, keeping signs
    parts = correction_str.replace('+', ' +').replace('-', ' -').split()
    
    for part in parts:
        if '*' in part:
            coef, var = part.split('*')
            terms[var] = float(coef)
    
    return terms

def visualize_fold(fold_num=3):
    """Visualize predictions for a specific fold"""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Load model
    model_path = os.path.join(base_dir, 'results', 'loso_models', f'ude_fold_{fold_num}.pth')
    model = UDE()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    alpha = model.alpha.item()
    beta = model.beta.item()
    
    # Load symbolic results
    symbolic_path = os.path.join(base_dir, 'results', 'symbolic', 'discovered_equations.csv')
    symbolic_df = pd.read_csv(symbolic_path)
    fold_row = symbolic_df[symbolic_df['fold'] == fold_num].iloc[0]
    correction = fold_row['correction']
    
    print(f"Visualizing Fold {fold_num}")
    print(f"α = {alpha:.4f}, β = {beta:.4f}")
    print(f"Discovered correction: {correction}")
    
    # Load test data
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    test_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    if fold_num - 1 >= len(test_files):
        print(f"Fold {fold_num} test data not found")
        return
    
    test_file = os.path.join(data_dir, test_files[fold_num - 1])
    test_dataset = StressDataset(test_file, seq_len=60)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Get one batch
    batch = next(iter(test_loader))
    t_grid = batch['t'][0]
    y_true = batch['y'][0].numpy()  # (seq_len, 1)
    u = batch['u'][0]
    y0 = y_true[0]
    
    # UDE Prediction (with neural network)
    model.set_current_batch(t_grid, u.unsqueeze(0))
    with torch.no_grad():
        y_ude = odeint(model, torch.tensor(y0, dtype=torch.float32).unsqueeze(0), 
                       t_grid, method='dopri5')
        y_ude = y_ude.squeeze().numpy()
    
    # Symbolic Prediction (physics + discovered terms only, NO neural network)
    symbolic_terms = parse_correction_terms(correction)
    y_symbolic = np.zeros_like(y_true)
    y_symbolic[0] = y0
    
    # Simple Euler integration for symbolic equation
    dt = 1.0  # 1 second steps
    for i in range(len(t_grid) - 1):
        S_curr = y_symbolic[i, 0]
        W_curr = u[i].item()  # Current workload
        
        # Physics
        dS = -beta * S_curr + alpha * W_curr
        
        # Symbolic corrections
        if 'S' in symbolic_terms:
            dS += symbolic_terms['S'] * S_curr
        if 'W' in symbolic_terms:
            dS += symbolic_terms['W'] * W_curr
        if 'S^2' in symbolic_terms:
            dS += symbolic_terms['S^2'] * (S_curr**2)
        if 'S^2 W' in symbolic_terms:
            dS += symbolic_terms['S^2 W'] * (S_curr**2 * W_curr)
        
        y_symbolic[i+1, 0] = S_curr + dS * dt
        y_symbolic[i+1, 0] = np.clip(y_symbolic[i+1, 0], 0, 1)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Stress predictions
    ax1 = axes[0]
    time = t_grid.numpy()
    
    ax1.plot(time, y_true, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax1.plot(time, y_ude, 'b--', linewidth=2, label='UDE (Physics + Neural Net)')
    ax1.plot(time, y_symbolic, 'r:', linewidth=2.5, label='Symbolic (Physics + Discovered Equation)')
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Stress Level', fontsize=12)
    ax1.set_title(f'Fold {fold_num}: UDE vs. Symbolic Regression Approximation\n'
                  f'α={alpha:.3f}, β={beta:.3f}, Correction: {correction}', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Bottom: Errors
    ax2 = axes[1]
    error_ude = np.abs(y_ude - y_true.flatten())
    error_symbolic = np.abs(y_symbolic.flatten() - y_true.flatten())
    
    ax2.plot(time, error_ude, 'b-', linewidth=2, label=f'UDE Error (MAE={error_ude.mean():.4f})')
    ax2.plot(time, error_symbolic, 'r-', linewidth=2, label=f'Symbolic Error (MAE={error_symbolic.mean():.4f})')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Prediction Errors Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(base_dir, 'results', 'symbolic')
    output_file = os.path.join(output_dir, f'visualization_fold_{fold_num}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to: {output_file}")
    
    plt.show()
    
    # Print statistics
    print(f"\n📊 Error Statistics:")
    print(f"  UDE MAE:      {error_ude.mean():.4f} ± {error_ude.std():.4f}")
    print(f"  Symbolic MAE: {error_symbolic.mean():.4f} ± {error_symbolic.std():.4f}")
    print(f"  Approximation Quality: {(1 - error_symbolic.mean()/error_ude.mean())*100:.1f}% as good as UDE")

def visualize_all_folds():
    """Create comparison visualization for all folds"""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    symbolic_path = os.path.join(base_dir, 'results', 'symbolic', 'discovered_equations.csv')
    symbolic_df = pd.read_csv(symbolic_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, row in symbolic_df.iterrows():
        fold_num = int(row['fold'])
        if fold_num > 6:
            break
            
        ax = axes[i]
        
        alpha = row['alpha']
        beta = row['beta']
        risk = row['risk']
        
        # Simple phase portrait
        S = np.linspace(0, 1, 100)
        W_low = 0.2
        W_high = 0.8
        
        # dS/dt for different workloads
        dS_low = -beta * S + alpha * W_low
        dS_high = -beta * S + alpha * W_high
        
        ax.plot(S, dS_low, 'b-', label=f'W={W_low} (low)')
        ax.plot(S, dS_high, 'r-', label=f'W={W_high} (high)')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Stress (S)')
        ax.set_ylabel('dS/dt')
        ax.set_title(f'Fold {fold_num}: α={alpha:.3f}, β={beta:.3f}\nRisk={risk:.2f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(base_dir, 'results', 'symbolic', 'phase_portraits_all_folds.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved phase portraits to: {output_file}")
    plt.show()

if __name__ == "__main__":
    print("="*70)
    print("SYMBOLIC REGRESSION VISUALIZATION")
    print("="*70)
    
    # Visualize one fold in detail
    visualize_fold(fold_num=3)
    
    # Overview of all folds
    print("\nCreating overview of all folds...")
    visualize_all_folds()
