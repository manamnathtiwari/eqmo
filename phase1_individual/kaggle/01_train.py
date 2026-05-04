"""
KAGGLE NOTEBOOK: MC-UDE Training with L1-Sparse Regularization
================================================================
Self-contained script for training Multi-Coefficient UDE models
with all novelty components.

Upload DATA: Upload your normalized WESAD CSV files to Kaggle as a dataset.
Run this notebook with GPU enabled.

Expected runtime: ~4-6 hours on P100 GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import os
import json
from glob import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'DATA_DIR': '/kaggle/input/wesad-normalized/normalized',  # Adjust to your Kaggle dataset path
    'OUTPUT_DIR': '/kaggle/working/mc_ude_results',
    'SEQ_LEN': 60,
    'EPOCHS': 50,
    'BATCH_SIZE': 32,
    'LR': 0.001,
    'LAMBDA_L1': 0.001,         # L1 sparsity strength
    'LAMBDA_PHYSICS': 0.01,     # Physics constraint strength 
    'HIDDEN_DIM': 64,
    'NUM_FEATURES': 18,
}

# Feature columns (normalized versions in CSV)
FEATURE_COLUMNS = [
    'workload_norm',
    'hrv_rmssd_norm', 'hrv_sdnn_norm', 'hrv_pnn50_norm', 'hrv_lf_hf_norm',
    'heart_rate_norm',
    'eda_mean_norm', 'eda_std_norm', 'eda_peaks_norm',
    'resp_mean_norm', 'resp_std_norm', 'resp_rate_norm',
    'temp_mean_norm', 'temp_std_norm',
    'activity_level_norm', 'activity_std_norm',
    'emg_mean_norm', 'emg_std_norm'
]

FEATURE_DISPLAY_NAMES = [
    'Workload', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF/HF',
    'Heart Rate', 'EDA_Mean', 'EDA_Std', 'EDA_Peaks',
    'Resp_Mean', 'Resp_Std', 'Resp_Rate',
    'Temp_Mean', 'Temp_Std', 'Activity_Mean', 'Activity_Std',
    'EMG_Mean', 'EMG_Std'
]

# ============================================================================
# DATASET
# ============================================================================
class StressDataset(Dataset):
    """Non-overlapping stress sequences for UDE training"""
    def __init__(self, csv_path, seq_len=60):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        
        # Use normalized features
        self.feature_columns = [c for c in FEATURE_COLUMNS if c in self.df.columns]
        if len(self.feature_columns) == 0:
            # Fallback: try raw columns
            raw_cols = [c.replace('_norm', '') for c in FEATURE_COLUMNS]
            self.feature_columns = [c for c in raw_cols if c in self.df.columns]
            print(f"  WARNING: Using raw features in {csv_path}")
        
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        self.stress = self.df['stress'].values.astype(np.float32)
        self.time = self.df['time'].values.astype(np.float32)
        self.num_features = len(self.feature_columns)
        self.n_sequences = max(0, (len(self.df) - seq_len) // (seq_len // 2))
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start = idx * (self.seq_len // 2)
        end = start + self.seq_len
        
        t = self.time[start:end] - self.time[start]
        y = self.stress[start:end]
        features = self.features[start:end]
        
        return {
            't': torch.tensor(t, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
            'features': torch.tensor(features, dtype=torch.float32),
            'num_features': self.num_features
        }


# ============================================================================
# MC-UDE MODEL (with L1-Sparse + Physics Constraints)
# ============================================================================
class MCUDE(nn.Module):
    """
    Multi-Coefficient Universal Differential Equation
    
    Model: dS/dt = -β·S + Σᵢ αᵢ·Fᵢ + NN(S, Features)
    
    Novelties:
    1. L1-sparse αᵢ for automatic feature selection
    2. Physics constraints on β (bounded recovery)
    3. Per-subject interpretable equations
    """
    def __init__(self, hidden_dim=64, num_features=18):
        super().__init__()
        
        input_dim = 1 + num_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))
        self._alphas_raw = nn.Parameter(torch.ones(num_features) * (-2.2))
        
        self.current_features = None
        self.current_t = None
        self.num_features = num_features
    
    @property
    def beta(self):
        return F.softplus(self._beta_raw)
    
    @property
    def alphas(self):
        return F.softplus(self._alphas_raw)
    
    def set_current_batch(self, t, features):
        self.current_t = t
        self.current_features = features
    
    def forward(self, t, y):
        S = y
        t_idx = torch.argmin(torch.abs(self.current_t - t))
        features = self.current_features[:, t_idx, :]
        
        recovery = -self.beta * S
        feature_contribution = torch.sum(self.alphas * features, dim=-1)
        f_known = recovery + feature_contribution
        
        if S.dim() == 1:
            S_expanded = S.unsqueeze(-1)
        else:
            S_expanded = S
        
        nn_in = torch.cat([S_expanded, features], dim=-1)
        f_nn = self.net(nn_in)
        
        if S.dim() == 1:
            f_nn = f_nn.squeeze(-1)
        
        return f_known + f_nn
    
    def l1_loss(self, lambda_l1=0.001):
        """L1 sparsity penalty on alphas"""
        return lambda_l1 * torch.sum(torch.abs(self.alphas))
    
    def physics_loss(self, lambda_physics=0.01):
        """Physics constraint: β must be in reasonable range [0.01, 5.0]"""
        beta = self.beta
        low = F.relu(0.01 - beta)
        high = F.relu(beta - 5.0)
        return lambda_physics * (low + high).squeeze()
    
    def get_equation_string(self, threshold=0.01):
        """Extract learned equation as human-readable string"""
        beta = self.beta.item()
        alphas = self.alphas.detach().cpu().numpy()
        
        terms = [f"-{beta:.4f}·S"]
        for i, (alpha, name) in enumerate(zip(alphas, FEATURE_DISPLAY_NAMES)):
            if alpha > threshold:
                terms.append(f"+{alpha:.4f}·{name}")
        terms.append("+ NN(S, F)")
        
        return "dS/dt = " + " ".join(terms)
    
    def get_sparse_profile(self, threshold=0.01):
        """Get active features after L1 training"""
        alphas = self.alphas.detach().cpu().numpy()
        active = [(FEATURE_DISPLAY_NAMES[i], float(a)) 
                  for i, a in enumerate(alphas) if a > threshold]
        active.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'active_features': active,
            'n_active': len(active),
            'sparsity_pct': 100 * (1 - len(active) / len(alphas)),
            'all_alphas': alphas.tolist(),
            'beta': self.beta.item(),
        }


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_one_fold(model, train_loader, test_loader, epochs, lr, device, config):
    """Train one LOSO fold with L1 + physics constraints"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    lambda_l1 = config['LAMBDA_L1']
    lambda_physics = config['LAMBDA_PHYSICS']
    
    history = {'train_loss': [], 'val_loss': [], 'l1_loss': [], 'physics_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            t = batch['t'][0].to(device)
            y = batch['y'].to(device).squeeze(-1)
            features = batch['features'].to(device)
            y0 = y[:, 0]
            
            optimizer.zero_grad()
            model.set_current_batch(t, features)
            
            y_pred = odeint(model, y0, t, method='euler')
            y_pred = y_pred.permute(1, 0)
            
            # Combined loss: trajectory MSE + L1 sparsity + physics
            mse_loss = torch.mean((y_pred - y) ** 2)
            l1_loss = model.l1_loss(lambda_l1)
            phys_loss = model.physics_loss(lambda_physics)
            
            total_loss = mse_loss + l1_loss + phys_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(mse_loss.item())
        
        avg_train = np.mean(epoch_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in test_loader:
                t = batch['t'][0].to(device)
                y = batch['y'].to(device).squeeze(-1)
                features = batch['features'].to(device)
                y0 = y[:, 0]
                
                model.set_current_batch(t, features)
                y_pred = odeint(model, y0, t, method='euler')
                y_pred = y_pred.permute(1, 0)
                
                val_loss = torch.mean((y_pred - y) ** 2)
                val_losses.append(val_loss.item())
        
        avg_val = np.mean(val_losses)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        if (epoch + 1) % 10 == 0:
            n_active = sum(1 for a in model.alphas.detach().cpu().numpy() if a > 0.01)
            print(f"  Epoch {epoch+1}/{epochs}: Train={avg_train:.6f} Val={avg_val:.6f} "
                  f"β={model.beta.item():.4f} Active={n_active}/18")
    
    return history


# ============================================================================
# LOSO CROSS-VALIDATION
# ============================================================================
def run_loso_training():
    """Run full LOSO cross-validation with L1-sparse MC-UDE"""
    config = CONFIG
    
    print("=" * 70)
    print("MC-UDE TRAINING WITH L1-SPARSE REGULARIZATION")
    print("=" * 70)
    print(f"Config: {json.dumps(config, indent=2)}")
    
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    
    # Find data files
    csv_files = sorted(glob(os.path.join(config['DATA_DIR'], '*.csv')))
    if not csv_files:
        # Try alternate path
        alt_dir = '/kaggle/input/wesad-processed/processed/normalized'
        csv_files = sorted(glob(os.path.join(alt_dir, '*.csv')))
    
    print(f"\nFound {len(csv_files)} subject files")
    if len(csv_files) == 0:
        print("ERROR: No CSV files found. Check DATA_DIR path.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    results = []
    all_profiles = {}
    
    for fold_idx, test_file in enumerate(csv_files, 1):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx}/{len(csv_files)}: Test = {os.path.basename(test_file)}")
        print(f"{'=' * 70}")
        
        # Check if already done
        model_path = os.path.join(config['OUTPUT_DIR'], f'mcude_fold_{fold_idx}.pth')
        if os.path.exists(model_path):
            print("  Already completed, skipping.")
            continue
        
        train_files = [f for f in csv_files if f != test_file]
        
        # Create datasets
        train_datasets = [StressDataset(f, seq_len=config['SEQ_LEN']) for f in train_files]
        test_dataset = StressDataset(test_file, seq_len=config['SEQ_LEN'])
        
        # Flatten train datasets
        train_data = []
        for ds in train_datasets:
            for i in range(len(ds)):
                train_data.append(ds[i])
        
        train_loader = DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
        
        print(f"  Train sequences: {len(train_data)}")
        print(f"  Test sequences: {len(test_dataset)}")
        
        # Create model
        model = MCUDE(
            hidden_dim=config['HIDDEN_DIM'],
            num_features=config['NUM_FEATURES']
        ).to(device)
        
        # Train
        history = train_one_fold(
            model, train_loader, test_loader,
            epochs=config['EPOCHS'], lr=config['LR'],
            device=device, config=config
        )
        
        # Final evaluation
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                t = batch['t'][0].to(device)
                y = batch['y'].to(device).squeeze(-1)
                features = batch['features'].to(device)
                y0 = y[:, 0]
                
                model.set_current_batch(t, features)
                y_pred = odeint(model, y0, t, method='dopri5', rtol=1e-3, atol=1e-4)
                y_pred = y_pred.permute(1, 0)
                
                test_losses.append(torch.mean((y_pred - y) ** 2).item())
        
        test_mse = np.mean(test_losses)
        
        # Extract profile
        profile = model.get_sparse_profile()
        equation = model.get_equation_string()
        
        print(f"\n  ✅ Test MSE: {test_mse:.6f}")
        print(f"  Equation: {equation}")
        print(f"  Active features: {profile['n_active']}/18 (sparsity: {profile['sparsity_pct']:.0f}%)")
        for feat, alpha in profile['active_features'][:5]:
            print(f"    {feat}: α = {alpha:.4f}")
        
        # Save model
        torch.save(model.state_dict(), model_path)
        
        # Save profile
        profile_path = os.path.join(config['OUTPUT_DIR'], f'profile_fold_{fold_idx}.json')
        save_profile = {k: v for k, v in profile.items()}
        save_profile['equation'] = equation
        save_profile['test_mse'] = test_mse
        with open(profile_path, 'w') as f:
            json.dump(save_profile, f, indent=2)
        
        results.append({
            'Fold': fold_idx,
            'Subject': os.path.basename(test_file),
            'Test_MSE': test_mse,
            'Beta': profile['beta'],
            'N_Active': profile['n_active'],
            'Sparsity': profile['sparsity_pct'],
        })
        
        all_profiles[os.path.basename(test_file)] = profile
        
        # Save checkpoint
        pd.DataFrame(results).to_csv(
            os.path.join(config['OUTPUT_DIR'], 'loso_results.csv'), index=False
        )
    
    # Final summary
    df = pd.DataFrame(results)
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))
    print(f"\nMean Test MSE: {df['Test_MSE'].mean():.6f} ± {df['Test_MSE'].std():.6f}")
    print(f"Mean Sparsity: {df['Sparsity'].mean():.0f}%")
    print(f"Mean Active Features: {df['N_Active'].mean():.1f}/18")
    
    return df


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    results = run_loso_training()
    print("\n✅ Training complete! Download results from /kaggle/working/mc_ude_results/")
