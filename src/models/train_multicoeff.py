"""
Multi-Coefficient UDE Training Script
LOSO Cross-Validation for WESAD Dataset

Complete, tested, ready for Kaggle
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_multicoeff import UDEMultiCoeff
from src.utils import StressDataset


# Feature names for WESAD
FEATURE_NAMES = [
    'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50', 'hrv_lf_hf',
    'hr_mean_norm', 'hr_std_norm',
    'eda_mean_norm', 'eda_std_norm', 'eda_peaks_norm',
    'temp_mean_norm', 'temp_std_norm',
    'resp_mean_norm', 'resp_std_norm',
    'activity_mean_norm', 'activity_std_norm',
    'emg_mean_norm', 'emg_std_norm',
    'workload'
]


def train_one_fold(model, train_loader, test_loader, epochs=50, lr=0.001, device='cpu'):
    """
    Train model for one fold
    
    Args:
        model: UDEMultiCoeff model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        test_loss: Final test loss
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            # Unpack batch
            t = batch['t'][0].to(device)  # (seq_len,)
            y = batch['y'].to(device)  # (batch, seq_len, 1)
            features = batch['features'].to(device)  # (batch, seq_len, num_features)
            
            # Prepare for ODE
            y = y.squeeze(-1)  # (batch, seq_len)
            y0 = y[:, 0:1]  # (batch, 1) - initial condition
            
            optimizer.zero_grad()
            
            # Set batch for model
            model.set_current_batch(t, features)
            
            # Solve ODE
            y_pred = odeint(model, y0, t, method='euler')  # (seq_len, batch, 1)
            y_pred = y_pred.permute(1, 0, 2).squeeze(-1)  # (batch, seq_len)
            
            # Loss
            loss = torch.mean((y_pred - y) ** 2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss = {avg_loss:.6f}")
    
    # Evaluate on test set
    model.eval()
    test_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            t = batch['t'][0].to(device)
            y = batch['y'].to(device).squeeze(-1)
            features = batch['features'].to(device)
            
            y0 = y[:, 0:1]
            
            model.set_current_batch(t, features)
            y_pred = odeint(model, y0, t, method='euler')
            y_pred = y_pred.permute(1, 0, 2).squeeze(-1)
            
            loss = torch.mean((y_pred - y) ** 2)
            test_losses.append(loss.item())
    
    test_loss = np.mean(test_losses)
    
    return model, train_losses, test_loss


def train_loso_multicoeff(data_dir, output_dir='results/multicoeff_models', 
                          seq_len=100, epochs=50, lr=0.001, batch_size=16):
    """
    Leave-One-Subject-Out Cross-Validation for Multi-Coefficient UDE
    
    Args:
        data_dir: Directory with normalized CSV files
        output_dir: Where to save trained models
        seq_len: Sequence length for temporal modeling
        epochs: Training epochs per fold
        lr: Learning rate
        batch_size: Batch size
        
    Returns:
        results_df: DataFrame with results for each fold
    """
    print("="*70)
    print("MULTI-COEFFICIENT UDE: LOSO CROSS-VALIDATION")
    print("="*70)
    print(f"Model: 18 separate alphas + 1 beta + Neural Network")
    print("="*70)
    
    # Get all subject files
    csv_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.csv') and f.startswith('u_wesad')
    ])
    
    print(f"\nFound {len(csv_files)} subjects")
    print(f"Config: seq_len={seq_len}, epochs={epochs}, lr={lr}, batch_size={batch_size}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # LOSO: Each subject is test set once
    for fold_idx, test_file in enumerate(csv_files, 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}/{len(csv_files)}")
        print(f"{'='*70}")
        print(f"Test Subject: {os.path.basename(test_file)}")
        
        # Split train/test
        train_files = [f for f in csv_files if f != test_file]
        print(f"Training on: {len(train_files)} subjects\n")
        
        # Create datasets
        train_datasets = [StressDataset(f, seq_len=seq_len) for f in train_files]
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = StressDataset(test_file, seq_len=seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train sequences: {len(train_dataset)}")
        print(f"Test sequences: {len(test_dataset)}")
        
        # Create model
        model = UDEMultiCoeff(hidden_dim=64, num_features=18)
        
        # Train
        print(f"\nTraining...")
        model, train_losses, test_loss = train_one_fold(
            model, train_loader, test_loader,
            epochs=epochs, lr=lr, device=device
        )
        
        print(f"\n✅ Fold {fold_idx} Complete!")
        print(f"   Test MSE: {test_loss:.6f}")
        
        # Get learned parameters
        params = model.get_learned_params()
        
        # Print discovered equation
        print(f"\n   Learned Parameters:")
        print(f"   Beta (recovery): {params['beta']:.6f}")
        print(f"   Alphas (mean): {params['alphas'].mean():.6f}")
        print(f"   Alphas (std): {params['alphas'].std():.6f}")
        
        # Save model
        model_path = os.path.join(output_dir, f'multicoeff_ude_fold_{fold_idx}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"\n   Saved: {model_path}")
        
        # Save alphas
        alphas_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Alpha': params['alphas']
        })
        alphas_path = os.path.join(output_dir, f'alphas_fold_{fold_idx}.csv')
        alphas_df.to_csv(alphas_path, index=False)
        print(f"   Saved: {alphas_path}")
        
        # Record results
        results.append({
            'Fold': fold_idx,
            'Subject': os.path.basename(test_file),
            'Test_MSE': test_loss,
            'Beta': params['beta'],
            'Alpha_Mean': params['alphas'].mean(),
            'Alpha_Std': params['alphas'].std(),
            'Alpha_Min': params['alphas'].min(),
            'Alpha_Max': params['alphas'].max()
        })
    
    # Save overall results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'multicoeff_loso_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*70}")
    print("LOSO CROSS-VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"\nMean Test MSE: {results_df['Test_MSE'].mean():.6f} ± {results_df['Test_MSE'].std():.6f}")
    print(f"\nResults saved to: {results_path}")
    print(f"{'='*70}")
    
    return results_df


if __name__ == "__main__":
    # For local testing
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / 'data' / 'processed' / 'normalized'
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Run LOSO
    results = train_loso_multicoeff(
        data_dir=str(data_dir),
        epochs=50  # Use 10 for quick test, 50 for final
    )
    
    print("\n✅ Training Complete!")
