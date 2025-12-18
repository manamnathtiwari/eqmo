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
from src.utils import StressDataset

def evaluate():
    # Config
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'results', 'ude_model.pth')
    data_path = os.path.join(base_dir, 'data', 'processed', 'u_001.csv') # Test on a different user (u_001)
    seq_len = 1000 # Longer sequence for evaluation
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    # Load Data
    # We just want one long sequence
    df = pd.read_csv(data_path)
    t = df['time'].values[:seq_len]
    y_true = df['stress'].values[:seq_len]
    u = df['workload'].values[:seq_len]
    
    # Convert to tensor
    t_tensor = torch.tensor(t, dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) # (1, seq, 1)
    u_tensor = torch.tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(-1) # (1, seq, 1)
    
    # Model
    model = UDE()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Predict
    with torch.no_grad():
        # Set context
        model.set_current_batch(t_tensor, u_tensor)
        
        y0 = y_tensor[:, 0, :] # (1, 1)
        
        # Integrate
        # Note: integrating over long sequence might drift, but let's see
        y_pred = odeint(model, y0, t_tensor, method='rk4')
        y_pred = y_pred.permute(1, 0, 2).numpy() # (1, seq, 1)
        
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(t, y_true, label='True Stress', alpha=0.7)
    plt.plot(t, y_pred[0, :, 0], label='Predicted Stress (UDE)', linestyle='--')
    plt.plot(t, u * 0.2, label='Workload (scaled)', alpha=0.3) # Show workload context
    plt.legend()
    plt.title(f'UDE Evaluation on {os.path.basename(data_path)}')
    plt.xlabel('Time (min)')
    plt.ylabel('Stress Level')
    save_path = os.path.join(base_dir, 'results', 'eval_plot.png')
    plt.savefig(save_path)
    print(f"Evaluation plot saved to {save_path}")

if __name__ == "__main__":
    evaluate()
