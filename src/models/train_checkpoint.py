"""
CHECKPOINT-ENABLED Multi-Coefficient UDE Training
Survives Kaggle 9-hour timeout by saving progress and resuming

Features:
- Saves after each fold
- Checks for existing models on startup
- Skips completed folds
- Resumes from where it stopped
- Can run across multiple sessions
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import os
from glob import glob


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


class FastStressDataset(Dataset):
    """Fast dataset with NO sequence overlap"""
    def __init__(self, csv_path, seq_len=50):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        
        self.features = self.df[FEATURE_NAMES].values.astype(np.float32)
        self.stress = self.df['stress'].values.astype(np.float32)
        self.time = self.df['time'].values.astype(np.float32)
        
        self.num_sequences = (len(self.df) - seq_len) // seq_len
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        t = self.time[start_idx:end_idx]
        y = self.stress[start_idx:end_idx]
        features = self.features[start_idx:end_idx, :]
        
        t = t - t[0]
        
        return {
            't': torch.tensor(t, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
            'features': torch.tensor(features, dtype=torch.float32)
        }


def check_completed_folds(output_dir):
    """
    Check which folds are already completed
    Returns list of completed fold numbers
    """
    if not os.path.exists(output_dir):
        return []
    
    completed = []
    for i in range(1, 16):  # 15 folds
        model_path = os.path.join(output_dir, f'multicoeff_ude_fold_{i}.pth')
        alphas_path = os.path.join(output_dir, f'alphas_fold_{i}.csv')
        
        if os.path.exists(model_path) and os.path.exists(alphas_path):
            completed.append(i)
    
    return sorted(completed)


def train_one_fold_fast(model, train_loader, test_loader, epochs=20, lr=0.001, device='cpu'):
    """Fast training with fewer epochs"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            t = batch['t'][0].to(device)
            y = batch['y'].to(device).squeeze(-1)
            features = batch['features'].to(device)
            
            y0 = y[:, 0:1]
            
            optimizer.zero_grad()
            
            model.set_current_batch(t, features)
            y_pred = odeint(model, y0, t, method='euler')
            y_pred = y_pred.permute(1, 0, 2).squeeze(-1)
            
            loss = torch.mean((y_pred - y) ** 2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss = {avg_loss:.6f}")
    
    # Evaluate
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


def train_loso_checkpoint(data_dir, output_dir='results/multicoeff_models', 
                          seq_len=50, epochs=20, lr=0.001, batch_size=32):
    """
    CHECKPOINT-ENABLED LOSO Cross-Validation
    
    Automatically resumes from last completed fold
    Can survive multiple Kaggle sessions
    """
    print("="*70)
    print("CHECKPOINT-ENABLED MULTI-COEFFICIENT UDE")
    print("="*70)
    print("Automatically resumes from last checkpoint")
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
    
    print(f"\nConfig: seq_len={seq_len}, epochs={epochs}, lr={lr}, batch_size={batch_size}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Import model
    from ude_multicoeff import UDEMultiCoeff
    
    results = []
    
    # Load existing results if any
    results_path = os.path.join(output_dir, 'loso_results.csv')
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
        
        # Create datasets
        train_datasets = [FastStressDataset(f, seq_len=seq_len) for f in train_files]
        test_dataset = FastStressDataset(test_file, seq_len=seq_len)
        
        train_data = []
        for ds in train_datasets:
            train_data.extend([ds[i] for i in range(len(ds))])
        
        print(f"Train sequences: {len(train_data)} (non-overlapping)")
        print(f"Test sequences: {len(test_dataset)} (non-overlapping)")
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = UDEMultiCoeff(hidden_dim=64, num_features=18)
        
        # Train
        print(f"\nTraining...")
        model, test_loss = train_one_fold_fast(
            model, train_loader, test_loader,
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
        model_path = os.path.join(output_dir, f'multicoeff_ude_fold_{fold_idx}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"\n   💾 CHECKPOINT: Saved {model_path}")
        
        alphas_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
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
    print("LOSO CROSS-VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"\nCompleted folds: {len(results)}/{len(csv_files)}")
    print(f"Mean Test MSE: {results_df['Test_MSE'].mean():.6f} ± {results_df['Test_MSE'].std():.6f}")
    print(f"\nAll results saved to: {results_path}")
    print(f"{'='*70}")
    
    return results_df


if __name__ == "__main__":
    results = train_loso_checkpoint(
        data_dir='data/processed/normalized',
        output_dir='results/multicoeff_models',
        seq_len=50,
        epochs=20,
        batch_size=32
    )
    
    print("\n✅ Training session complete!")
    print("💡 If timeout occurred, just restart - it will resume automatically!")
