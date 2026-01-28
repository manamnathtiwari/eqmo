"""
COMBINED Multi-Coefficient UDE Training
Runs BOTH Standard and Fast versions sequentially in one notebook

Total Expected Time: ~27 hours (3-4 Kaggle sessions)
- Standard: ~15 hours (Folds 1-15)
- Fast: ~12 hours (Folds 1-15)

Both have checkpoint support - survives 9-hour timeouts
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import os
import sys

sys.path.append('/kaggle/working')

from src.utils import StressDataset, FEATURE_COLUMNS
from src.models.ude_multicoeff import UDEMultiCoeff


# ============================================================================
# FAST DATASET (Non-overlapping)
# ============================================================================

class FastStressDataset(Dataset):
    """Non-overlapping sequences for fast training"""
    
    def __init__(self, csv_path, seq_len=50):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        
        self.feature_columns = [col for col in FEATURE_COLUMNS if col in self.df.columns]
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        self.stress = self.df['stress'].values.astype(np.float32)
        self.time = self.df['time'].values.astype(np.float32)
        
        self.num_sequences = (len(self.df) - seq_len) // seq_len
        self.num_features = len(self.feature_columns)
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        t = self.time[start_idx:end_idx] - self.time[start_idx]
        y = self.stress[start_idx:end_idx]
        features = self.features[start_idx:end_idx, :]
        
        return {
            't': torch.tensor(t, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
            'features': torch.tensor(features, dtype=torch.float32),
            'num_features': self.num_features
        }


# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def check_completed_folds(output_dir, prefix):
    """Check which folds are completed for a given prefix (standard/fast)"""
    if not os.path.exists(output_dir):
        return []
    
    completed = []
    for i in range(1, 16):
        model_path = os.path.join(output_dir, f'{prefix}_fold_{i}.pth')
        alphas_path = os.path.join(output_dir, f'alphas_{prefix}_fold_{i}.csv')
        
        if os.path.exists(model_path) and os.path.exists(alphas_path):
            completed.append(i)
    
    return sorted(completed)


# ============================================================================
# TRAINING FUNCTION (Used by both Standard and Fast)
# ============================================================================

def train_one_fold(model, train_loader, test_loader, fold_idx, config_name, 
                   epochs=25, lr=0.001, device='cpu'):
    """
    Train one fold
    
    Args:
        model: UDEMultiCoeff model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        fold_idx: Current fold number
        config_name: 'Standard' or 'Fast'
        epochs: Number of epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining Fold {fold_idx} ({config_name} - {epochs} epochs)...")
    
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
    
    return model, np.mean(test_losses)


# ============================================================================
# STANDARD TRAINING (Overlapping sequences)
# ============================================================================

def train_standard(data_dir, output_dir='results/standard_models', 
                   seq_len=100, epochs=25, batch_size=16):
    """Train STANDARD version with overlapping sequences"""
    
    print("\n" + "="*70)
    print("STANDARD TRAINING (Overlapping Sequences)")
    print("="*70)
    print(f"Config: seq_len={seq_len}, epochs={epochs}, batch_size={batch_size}")
    print("="*70)
    
    csv_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.csv') and f.startswith('u_wesad')
    ])
    
    os.makedirs(output_dir, exist_ok=True)
    completed = check_completed_folds(output_dir, 'standard')
    
    if completed:
        print(f"\n✅ Found {len(completed)} completed folds: {completed}")
    else:
        print(f"\n🆕 Starting from Fold 1")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    results = []
    results_path = os.path.join(output_dir, 'standard_loso_results.csv')
    
    if os.path.exists(results_path):
        results = pd.read_csv(results_path).to_dict('records')
    
    for fold_idx, test_file in enumerate(csv_files, 1):
        if fold_idx in completed:
            print(f"\nFOLD {fold_idx}/15 - ALREADY COMPLETED ✅")
            continue
        
        print(f"\n{'='*70}")
        print(f"STANDARD - FOLD {fold_idx}/15")
        print(f"{'='*70}")
        print(f"Test: {os.path.basename(test_file)}")
        
        train_files = [f for f in csv_files if f != test_file]
        
        # Overlapping sequences
        train_datasets = [StressDataset(f, seq_len=seq_len) for f in train_files]
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = StressDataset(test_file, seq_len=seq_len)
        
        print(f"Train: {len(train_dataset)} sequences (overlapping)")
        print(f"Test: {len(test_dataset)} sequences")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        model = UDEMultiCoeff(hidden_dim=64, num_features=len(FEATURE_COLUMNS))
        
        model, test_loss = train_one_fold(
            model, train_loader, test_loader, fold_idx, 'Standard',
            epochs=epochs, device=device
        )
        
        print(f"\n✅ Fold {fold_idx} Complete! MSE: {test_loss:.6f}")
        
        params = model.get_learned_params()
        
        # ATOMIC SAVE: Save to temp files first, then rename (all-or-nothing)
        try:
            model_path = os.path.join(output_dir, f'standard_fold_{fold_idx}.pth')
            alphas_path = os.path.join(output_dir, f'alphas_standard_fold_{fold_idx}.csv')
            
            # Save to temporary files
            model_temp = model_path + '.tmp'
            alphas_temp = alphas_path + '.tmp'
            results_temp = results_path + '.tmp'
            
            torch.save(model.state_dict(), model_temp)
            pd.DataFrame({'Feature': FEATURE_COLUMNS, 'Alpha': params['alphas']}).to_csv(
                alphas_temp, index=False)
            
            results.append({
                'Fold': fold_idx,
                'Subject': os.path.basename(test_file),
                'Test_MSE': test_loss,
                'Beta': params['beta']
            })
            
            pd.DataFrame(results).to_csv(results_temp, index=False)
            
            # All saves successful - atomically rename temp files
            os.replace(model_temp, model_path)
            os.replace(alphas_temp, alphas_path)
            os.replace(results_temp, results_path)
            
            print(f"💾 Checkpoint saved! Progress: {fold_idx}/15")
            
        except Exception as e:
            # Rollback: Remove temp files and pop from results
            print(f"⚠️  Save failed: {e}")
            print(f"🔄 Rolling back Fold {fold_idx}...")
            
            if os.path.exists(model_temp):
                os.remove(model_temp)
            if os.path.exists(alphas_temp):
                os.remove(alphas_temp)
            if os.path.exists(results_temp):
                os.remove(results_temp)
            
            # Remove from results list
            if results and results[-1]['Fold'] == fold_idx:
                results.pop()
            
            print(f"❌ Fold {fold_idx} NOT saved - will retry on next run")
            raise  # Re-raise to stop execution
    
    df = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print(f"STANDARD COMPLETE: {len(results)}/15 folds")
    print(f"Mean MSE: {df['Test_MSE'].mean():.6f} ± {df['Test_MSE'].std():.6f}")
    print(f"{'='*70}\n")
    
    return df


# ============================================================================
# FAST TRAINING (Non-overlapping sequences)
# ============================================================================

def train_fast(data_dir, output_dir='results/fast_models', 
               seq_len=50, epochs=25, batch_size=32):
    """Train FAST version with non-overlapping sequences"""
    
    print("\n" + "="*70)
    print("FAST TRAINING (Non-overlapping Sequences)")
    print("="*70)
    print(f"Config: seq_len={seq_len}, epochs={epochs}, batch_size={batch_size}")
    print("="*70)
    
    csv_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.csv') and f.startswith('u_wesad')
    ])
    
    os.makedirs(output_dir, exist_ok=True)
    completed = check_completed_folds(output_dir, 'fast')
    
    if completed:
        print(f"\n✅ Found {len(completed)} completed folds: {completed}")
    else:
        print(f"\n🆕 Starting from Fold 1")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    results = []
    results_path = os.path.join(output_dir, 'fast_loso_results.csv')
    
    if os.path.exists(results_path):
        results = pd.read_csv(results_path).to_dict('records')
    
    for fold_idx, test_file in enumerate(csv_files, 1):
        if fold_idx in completed:
            print(f"\nFOLD {fold_idx}/15 - ALREADY COMPLETED ✅")
            continue
        
        print(f"\n{'='*70}")
        print(f"FAST - FOLD {fold_idx}/15")
        print(f"{'='*70}")
        print(f"Test: {os.path.basename(test_file)}")
        
        train_files = [f for f in csv_files if f != test_file]
        
        # Non-overlapping sequences
        train_datasets = [FastStressDataset(f, seq_len=seq_len) for f in train_files]
        test_dataset = FastStressDataset(test_file, seq_len=seq_len)
        
        train_data = []
        for ds in train_datasets:
            train_data.extend([ds[i] for i in range(len(ds))])
        
        print(f"Train: {len(train_data)} sequences (non-overlapping)")
        print(f"Test: {len(test_dataset)} sequences")
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        model = UDEMultiCoeff(hidden_dim=64, num_features=len(FEATURE_COLUMNS))
        
        model, test_loss = train_one_fold(
            model, train_loader, test_loader, fold_idx, 'Fast',
            epochs=epochs, device=device
        )
        
        print(f"\n✅ Fold {fold_idx} Complete! MSE: {test_loss:.6f}")
        
        params = model.get_learned_params()
        
        # ATOMIC SAVE: Save to temp files first, then rename (all-or-nothing)
        try:
            model_path = os.path.join(output_dir, f'fast_fold_{fold_idx}.pth')
            alphas_path = os.path.join(output_dir, f'alphas_fast_fold_{fold_idx}.csv')
            
            # Save to temporary files
            model_temp = model_path + '.tmp'
            alphas_temp = alphas_path + '.tmp'
            results_temp = results_path + '.tmp'
            
            torch.save(model.state_dict(), model_temp)
            pd.DataFrame({'Feature': FEATURE_COLUMNS, 'Alpha': params['alphas']}).to_csv(
                alphas_temp, index=False)
            
            results.append({
                'Fold': fold_idx,
                'Subject': os.path.basename(test_file),
                'Test_MSE': test_loss,
                'Beta': params['beta']
            })
            
            pd.DataFrame(results).to_csv(results_temp, index=False)
            
            # All saves successful - atomically rename temp files
            os.replace(model_temp, model_path)
            os.replace(alphas_temp, alphas_path)
            os.replace(results_temp, results_path)
            
            print(f"💾 Checkpoint saved! Progress: {fold_idx}/15")
            
        except Exception as e:
            # Rollback: Remove temp files and pop from results
            print(f"⚠️  Save failed: {e}")
            print(f"🔄 Rolling back Fold {fold_idx}...")
            
            if os.path.exists(model_temp):
                os.remove(model_temp)
            if os.path.exists(alphas_temp):
                os.remove(alphas_temp)
            if os.path.exists(results_temp):
                os.remove(results_temp)
            
            # Remove from results list
            if results and results[-1]['Fold'] == fold_idx:
                results.pop()
            
            print(f"❌ Fold {fold_idx} NOT saved - will retry on next run")
            raise  # Re-raise to stop execution
    
    df = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print(f"FAST COMPLETE: {len(results)}/15 folds")
    print(f"Mean MSE: {df['Test_MSE'].mean():.6f} ± {df['Test_MSE'].std():.6f}")
    print(f"{'='*70}\n")
    
    return df


# ============================================================================
# MAIN: Run BOTH sequentially
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMBINED TRAINING: STANDARD + FAST")
    print("="*70)
    print("This will train BOTH versions sequentially")
    print("Total expected time: ~27 hours (3-4 sessions)")
    print("="*70)
    
    data_dir = 'data/processed/normalized'
    
    # Train STANDARD first
    print("\n\n🚀 Starting STANDARD training...")
    standard_results = train_standard(
        data_dir=data_dir,
        output_dir='results/standard_models',
        seq_len=100,
        epochs=25,
        batch_size=16
    )
    
    # Train FAST second
    print("\n\n🚀 Starting FAST training...")
    fast_results = train_fast(
        data_dir=data_dir,
        output_dir='results/fast_models',
        seq_len=50,
        epochs=25,
        batch_size=32
    )
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 ALL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nSTANDARD Results:")
    print(f"  Mean MSE: {standard_results['Test_MSE'].mean():.6f} ± {standard_results['Test_MSE'].std():.6f}")
    print(f"  Completed: {len(standard_results)}/15 folds")
    
    print(f"\nFAST Results:")
    print(f"  Mean MSE: {fast_results['Test_MSE'].mean():.6f} ± {fast_results['Test_MSE'].std():.6f}")
    print(f"  Completed: {len(fast_results)}/15 folds")
    
    print(f"\nDifference: {abs(standard_results['Test_MSE'].mean() - fast_results['Test_MSE'].mean()):.6f}")
    print("\n✅ Both models saved in results/ directory")
    print("="*70)
