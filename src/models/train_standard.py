"""
STANDARD Multi-Coefficient UDE Training with Checkpoints
Full quality training with overlapping sequences

Configuration:
- Overlapping sequences (50% overlap)
- 25 epochs per fold
- Batch size: 16
- Expected time: ~15 hours total (needs 2 sessions)
- Expected MSE: ~0.0052 (excellent quality)

Checkpoint Strategy:
- Saves after each fold
- Auto-resumes from last completed fold
- Survives 9-hour Kaggle timeout
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append('/kaggle/working')

from src.utils import StressDataset, FEATURE_COLUMNS
from src.models.ude_multicoeff import UDEMultiCoeff


def check_completed_folds(output_dir):
    """Check which folds are already completed"""
    if not os.path.exists(output_dir):
        return []
    
    completed = []
    for i in range(1, 16):  # 15 folds
        model_path = os.path.join(output_dir, f'standard_fold_{i}.pth')
        alphas_path = os.path.join(output_dir, f'alphas_fold_{i}.csv')
        
        if os.path.exists(model_path) and os.path.exists(alphas_path):
            completed.append(i)
    
    return sorted(completed)


def train_one_fold(model, train_loader, test_loader, fold_idx, epochs=25, lr=0.001, device='cpu'):
    """
    Train one fold with full quality settings
    
    Args:
        model: UDEMultiCoeff model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        fold_idx: Current fold number
        epochs: Number of epochs (default 25 for quality)
        lr: Learning rate
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Trained model
        test_loss: Test MSE
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining Fold {fold_idx} (Standard - {epochs} epochs)...")
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            t = batch['t'][0].to(device)  # (seq_len,)
            y = batch['y'].to(device).squeeze(-1)  # (batch, seq_len)
            features = batch['features'].to(device)  # (batch, seq_len, num_features)
            
            y0 = y[:, 0:1]  # (batch, 1)
            
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
        
        # Print every 10 epochs
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
    
    return model, test_loss


def train_loso_standard(data_dir, output_dir='results/standard_models', 
                       seq_len=100, epochs=25, lr=0.001, batch_size=16):
    """
    STANDARD Multi-Coefficient UDE Training
    Full quality with overlapping sequences and 25 epochs
    
    Expected:
    - Time: ~15 hours (2 Kaggle sessions)
    - MSE: ~0.0052 (excellent quality)
    - Sequences: ~80K per fold (overlapping)
    
    Args:
        data_dir: Directory with CSV files
        output_dir: Where to save models
        seq_len: Sequence length (100 for quality)
        epochs: Training epochs (25 for quality)
        lr: Learning rate
        batch_size: Batch size (16 for quality)
    """
    print("="*70)
    print("STANDARD MULTI-COEFFICIENT UDE TRAINING")
    print("="*70)
    print("Configuration: FULL QUALITY")
    print(f"  - Overlapping sequences (50% overlap)")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print("="*70)
    
    # Get all CSV files
    csv_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.csv') and f.startswith('u_wesad')
    ])
    
    print(f"\nFound {len(csv_files)} subjects")
    
    # Check completed folds
    os.makedirs(output_dir, exist_ok=True)
    completed_folds = check_completed_folds(output_dir)
    
    if completed_folds:
        print(f"\n✅ Found {len(completed_folds)} completed folds: {completed_folds}")
        print(f"⏭️  Will skip these and continue from Fold {max(completed_folds) + 1}")
    else:
        print(f"\n🆕 No completed folds found. Starting from Fold 1")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")
    
    results = []
    
    # Load existing results if any
    results_path = os.path.join(output_dir, 'standard_loso_results.csv')
    if os.path.exists(results_path):
        existing_results = pd.read_csv(results_path)
        results = existing_results.to_dict('records')
        print(f"📊 Loaded {len(results)} existing results\n")
    
    for fold_idx, test_file in enumerate(csv_files, 1):
        # Skip if already completed
        if fold_idx in completed_folds:
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx}/{len(csv_files)} - ALREADY COMPLETED ✅")
            print(f"{'='*70}")
            print(f"Skipping: {os.path.basename(test_file)}\n")
            continue
        
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}/{len(csv_files)}")
        print(f"{'='*70}")
        print(f"Test Subject: {os.path.basename(test_file)}")
        
        train_files = [f for f in csv_files if f != test_file]
        print(f"Training on: {len(train_files)} subjects\n")
        
        # Create datasets with OVERLAPPING sequences (standard StressDataset)
        train_datasets = [StressDataset(f, seq_len=seq_len) for f in train_files]
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = StressDataset(test_file, seq_len=seq_len)
        
        print(f"Train sequences: {len(train_dataset)} (overlapping)")
        print(f"Test sequences: {len(test_dataset)} (overlapping)")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        num_features = len(FEATURE_COLUMNS)
        model = UDEMultiCoeff(hidden_dim=64, num_features=num_features)
        
        # Train
        model, test_loss = train_one_fold(
            model, train_loader, test_loader, fold_idx,
            epochs=epochs, lr=lr, device=device
        )
        
        print(f"\n✅ Fold {fold_idx} Complete!")
        print(f"   Test MSE: {test_loss:.6f}")
        
        # Get learned parameters
        params = model.get_learned_params()
        
        print(f"\n   Learned Parameters:")
        print(f"   Beta (recovery): {params['beta']:.6f}")
        print(f"   Alphas (mean): {params['alphas'].mean():.6f}")
        print(f"   Alphas (std): {params['alphas'].std():.6f}")
        
        # CHECKPOINT: Save immediately
        model_path = os.path.join(output_dir, f'standard_fold_{fold_idx}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"\n   💾 CHECKPOINT: Saved {model_path}")
        
        alphas_df = pd.DataFrame({
            'Feature': FEATURE_COLUMNS,
            'Alpha': params['alphas']
        })
        alphas_path = os.path.join(output_dir, f'alphas_fold_{fold_idx}.csv')
        alphas_df.to_csv(alphas_path, index=False)
        print(f"   💾 CHECKPOINT: Saved {alphas_path}")
        
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
        
        # CHECKPOINT: Save results after each fold
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_path, index=False)
        print(f"   💾 CHECKPOINT: Updated {results_path}")
        print(f"\n   Progress: {fold_idx}/{len(csv_files)} folds complete")
    
    # Final summary
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("STANDARD TRAINING COMPLETE")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"\nCompleted folds: {len(results)}/{len(csv_files)}")
    print(f"Mean Test MSE: {results_df['Test_MSE'].mean():.6f} ± {results_df['Test_MSE'].std():.6f}")
    print(f"\nAll results saved to: {results_path}")
    print(f"{'='*70}")
    
    return results_df


if __name__ == "__main__":
    results = train_loso_standard(
        data_dir='data/processed/normalized',
        output_dir='results/standard_models',
        seq_len=100,
        epochs=25,
        batch_size=16
    )
    
    print("\n✅ Standard training session complete!")
    print("💡 If timeout occurred, just restart - it will resume automatically!")
