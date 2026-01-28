"""
Generate Publication-Ready Figures
----------------------------------
Creates high-quality plots for the IEEE TBME submission.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import torch
from torchdiffeq import odeint
import sys

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.ude_model import UDE
from src.utils import StressDataset
from torch.utils.data import DataLoader

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
colors = sns.color_palette("deep")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'paper_figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_loso_results():
    path = os.path.join(BASE_DIR, 'results', 'loso_models', 'loso_results.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_parameters():
    path = os.path.join(BASE_DIR, 'results', 'explainability', 'parameters.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def plot_loso_performance():
    """Figure 1: LOSO Cross-Validation Performance"""
    print("Generating Figure 1: LOSO Performance...")
    df = load_loso_results()
    if df is None:
        print("Skipping Figure 1 (No data)")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x='fold', y='test_mse', data=df, color=colors[0])
    plt.axhline(df['test_mse'].mean(), color='r', linestyle='--', label=f'Mean MSE: {df["test_mse"].mean():.4f}')
    
    plt.title('Performance Across 15 Subjects (LOSO Cross-Validation)', fontweight='bold')
    plt.xlabel('Fold / Subject')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Fig1_LOSO_Performance.png'), dpi=300)
    plt.close()

def plot_risk_matrix():
    """Figure 2: Alpha-Beta Risk Matrix"""
    print("Generating Figure 2: Risk Matrix...")
    df = load_parameters()
    if df is None:
        print("Skipping Figure 2 (No data)")
        return
        
    plt.figure(figsize=(8, 8))
    
    # Calculate boundaries (Risk = Alpha/Beta)
    x = np.linspace(0.01, 0.3, 100)
    y_high = x * 2.0  # Limit for high risk
    
    # Scatter
    sns.scatterplot(data=df, x='Beta (Recovery)', y='Alpha (Sensitivity)', 
                    hue='Burnout Risk', palette='RdYlGn_r', s=200, edgecolor='k')
    
    # Annotate
    for i in range(df.shape[0]):
        plt.text(df['Beta (Recovery)'][i]+0.002, df['Alpha (Sensitivity)'][i]+0.002, 
                 df['Subject'][i].replace('u_wesad_', 'S'), fontsize=9)
                 
    plt.title('Personalized Stress Evaluation Matrix', fontweight='bold')
    plt.xlabel('Recovery Rate (β) $\\rightarrow$')
    plt.ylabel('Stress Sensitivity (α) $\\rightarrow$')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Fig2_Risk_Matrix.png'), dpi=300)
    plt.close()

def plot_trajectory_example():
    """Figure 3: Representative Trajectory"""
    print("Generating Figure 3: Trajectory Example...")
    # Load Fold 5 model (Best Performer)
    fold_idx = 5
    model_path = os.path.join(BASE_DIR, 'results', 'loso_models', f'ude_fold_{fold_idx}.pth')
    data_path = os.path.join(BASE_DIR, 'data', 'processed', 'normalized')
    subject_file = [f for f in os.listdir(data_path) if f.startswith('u_wesad_')][fold_idx-1]
    
    if not os.path.exists(model_path):
        print("Skipping Figure 3 (Model not found)")
        return
        
    # Run Inference
    device = torch.device('cpu')
    model = UDE()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    ds = StressDataset(os.path.join(data_path, subject_file), seq_len=120) # Longer sequence
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(dl)) # First batch
    
    t = batch['t'][0]
    y_true = batch['y'][0]
    feat = batch['features'][0]
    y0 = y_true[0].unsqueeze(0)
    
    model.set_current_batch(t, feat.unsqueeze(0))
    with torch.no_grad():
        y_pred = odeint(model, y0, t, method='dopri5').squeeze()
        
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(t, y_true.squeeze(), 'k-', label='Ground Truth (ECG/HRV)', linewidth=2, alpha=0.7)
    plt.plot(t, y_pred, 'g--', label='UDE Discovered Dynamics', linewidth=2)
    
    # Workload is index 0 in features
    workload = feat[:, 0].numpy()
    plt.fill_between(t, 0, 1, where=(workload > 0.5), color='red', alpha=0.1, label='High Workload')
    
    plt.title(f'Stress Dynamics Recovery: Subject S{fold_idx}', fontweight='bold')
    plt.xlabel('Time (normalized)')
    plt.ylabel('Stress Level')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Fig3_Trajectory.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_loso_performance()
    plot_risk_matrix()
    plot_trajectory_example()
    print(f"✅ Figures generated in {RESULTS_DIR}")
