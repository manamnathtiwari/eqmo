"""
KAGGLE TRAINING SCRIPT - Multi-Coefficient UDE
Train 15 models with 18 separate alpha coefficients (one per feature)

Upload this to Kaggle and run!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from glob import glob

# ============================================================================
# MULTI-COEFFICIENT UDE MODEL
# ============================================================================

class MultiCoeffUDE(nn.Module):
    """
    UDE with 18 separate alpha coefficients (one per feature)
    
    Model: dS/dt = -β*S + Σᵢ αᵢ*Fᵢ + NN(S, Features)
    
    Parameters:
    - β: Single recovery rate
    - α₁...α₁₈: 18 feature-specific sensitivities
    - NN: Neural network for nonlinear dynamics
    """
    def __init__(self, hidden_dim=64, num_features=18):
        super(MultiCoeffUDE, self).__init__()
        
        input_dim = 1 + num_features
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Single recovery rate
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))
        
        # 18 SEPARATE alphas (one per feature)
        self._alphas_raw = nn.Parameter(torch.ones(num_features) * (-2.2))
        
        self.current_features = None
        self.current_t = None
        self.num_features = num_features
    
    @property
    def beta(self):
        return F.softplus(self._beta_raw)
    
    @property
    def alphas(self):
        """Returns 18 separate alpha values"""
        return F.softplus(self._alphas_raw)  # Shape: (18,)
    
    def set_current_batch(self, t, features):
        self.current_t = t
        self.current_features = features
    
    def forward(self, t, y):
        """
        ODE function: dS/dt
        
        Args:
            t: time point
            y: stress level (batch,)
        
        Returns:
            dS/dt: stress derivative (batch,)
        """
        batch_size = y.shape[0]
        
        # Get features at current time
        t_idx = torch.argmin(torch.abs(self.current_t - t))
        features = self.current_features[:, t_idx, :]  # (batch, 18)
        
        # Linear dynamics with 18 SEPARATE alphas
        stress_decay = -self.beta * y
        feature_drive = torch.sum(self.alphas * features, dim=1)  # Element-wise multiply then sum
        
        # Neural network correction
        nn_input = torch.cat([y.unsqueeze(1), features], dim=1)
        nn_correction = self.net(nn_input).squeeze()
        
        return stress_decay + feature_drive + nn_correction

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_multicoeff_ude(train_data, val_data, epochs=50, lr=0.001, device='cpu'):
    """
    Train multi-coefficient UDE model
    
    Args:
        train_data: dict with 'features', 'stress', 'time'
        val_data: dict with 'features', 'stress', 'time'
        epochs: number of training epochs
        lr: learning rate
        device: 'cpu' or 'cuda'
    
    Returns:
        trained model, training history
    """
    model = MultiCoeffUDE(hidden_dim=64, num_features=18).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Prepare data
    X_train = torch.FloatTensor(train_data['features']).to(device)  # (batch, seq, 18)
    y_train = torch.FloatTensor(train_data['stress']).to(device)    # (batch, seq)
    t_train = torch.FloatTensor(train_data['time']).to(device)      # (seq,)
    
    X_val = torch.FloatTensor(val_data['features']).to(device)
    y_val = torch.FloatTensor(val_data['stress']).to(device)
    t_val = torch.FloatTensor(val_data['time']).to(device)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Set batch
        model.set_current_batch(t_train, X_train)
        
        # Initial stress
        y0 = y_train[:, 0]
        
        # Solve ODE
        pred = odeint(model, y0, t_train, method='dopri5')  # (seq, batch)
        pred = pred.transpose(0, 1)  # (batch, seq)
        
        # Loss
        loss = torch.mean((pred - y_train) ** 2)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            model.set_current_batch(t_val, X_val)
            y0_val = y_val[:, 0]
            pred_val = odeint(model, y0_val, t_val, method='dopri5')
            pred_val = pred_val.transpose(0, 1)
            val_loss = torch.mean((pred_val - y_val) ** 2)
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
            
            # Print learned alphas
            alphas = model.alphas.detach().cpu().numpy()
            print(f"  Alpha range: [{alphas.min():.4f}, {alphas.max():.4f}]")
    
    return model, history

# ============================================================================
# LOSO CROSS-VALIDATION
# ============================================================================

def run_loso_multicoeff(data_dir, output_dir, epochs=50):
    """
    Run Leave-One-Subject-Out cross-validation with multi-coefficient UDE
    
    Args:
        data_dir: directory with normalized CSV files
        output_dir: where to save models
        epochs: training epochs per fold
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature names
    feature_names = [
        'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50', 'hrv_lf_hf',
        'hr_mean_norm', 'hr_std_norm',
        'eda_mean_norm', 'eda_std_norm', 'eda_peaks_norm',
        'temp_mean_norm', 'temp_std_norm',
        'resp_mean_norm', 'resp_std_norm',
        'activity_mean_norm', 'activity_std_norm',
        'emg_mean_norm', 'emg_std_norm',
        'workload'
    ]
    
    # Load all subjects
    csv_files = sorted(glob(os.path.join(data_dir, '*.csv')))
    print(f"Found {len(csv_files)} subject files")
    
    results = []
    
    for fold, test_file in enumerate(csv_files, 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{len(csv_files)} - Test: {os.path.basename(test_file)}")
        print(f"{'='*70}")
        
        # Prepare train/test split
        train_files = [f for f in csv_files if f != test_file]
        
        # Load training data
        train_dfs = [pd.read_csv(f) for f in train_files]
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        # Load test data
        test_df = pd.read_csv(test_file)
        
        # Prepare sequences
        def prepare_sequences(df, seq_len=100):
            features_list = []
            stress_list = []
            
            for i in range(0, len(df) - seq_len, seq_len // 2):
                seq = df.iloc[i:i+seq_len]
                
                features = seq[feature_names].values
                stress = seq['stress'].values
                
                features_list.append(features)
                stress_list.append(stress)
            
            return {
                'features': np.array(features_list),
                'stress': np.array(stress_list),
                'time': np.arange(seq_len)
            }
        
        train_data = prepare_sequences(train_df)
        test_data = prepare_sequences(test_df)
        
        print(f"Train sequences: {train_data['features'].shape[0]}")
        print(f"Test sequences: {test_data['features'].shape[0]}")
        
        # Train model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        model, history = train_multicoeff_ude(
            train_data, test_data,
            epochs=epochs,
            lr=0.001,
            device=device
        )
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(test_data['features']).to(device)
            y_test = torch.FloatTensor(test_data['stress']).to(device)
            t_test = torch.FloatTensor(test_data['time']).to(device)
            
            model.set_current_batch(t_test, X_test)
            y0_test = y_test[:, 0]
            pred_test = odeint(model, y0_test, t_test, method='dopri5')
            pred_test = pred_test.transpose(0, 1)
            
            test_mse = torch.mean((pred_test - y_test) ** 2).item()
        
        print(f"\nFinal Test MSE: {test_mse:.6f}")
        
        # Save model
        model_path = os.path.join(output_dir, f'multicoeff_ude_fold_{fold}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Saved: {model_path}")
        
        # Save learned alphas
        alphas = model.alphas.detach().cpu().numpy()
        beta = model.beta.detach().cpu().item()
        
        alphas_df = pd.DataFrame({
            'Feature': feature_names,
            'Alpha': alphas
        })
        alphas_df.to_csv(os.path.join(output_dir, f'alphas_fold_{fold}.csv'), index=False)
        
        # Record results
        results.append({
            'Fold': fold,
            'Subject': os.path.basename(test_file),
            'Test_MSE': test_mse,
            'Beta': beta,
            'Alpha_Mean': alphas.mean(),
            'Alpha_Std': alphas.std(),
            'Alpha_Min': alphas.min(),
            'Alpha_Max': alphas.max()
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'multicoeff_loso_results.csv'), index=False)
    
    print(f"\n{'='*70}")
    print("LOSO CROSS-VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(results_df)
    print(f"\nMean Test MSE: {results_df['Test_MSE'].mean():.6f} ± {results_df['Test_MSE'].std():.6f}")
    
    return results_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Kaggle paths
    DATA_DIR = '/kaggle/input/wesad-normalized/normalized'  # Adjust to your dataset path
    OUTPUT_DIR = '/kaggle/working/multicoeff_models'
    
    print("="*70)
    print("MULTI-COEFFICIENT UDE TRAINING")
    print("="*70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run LOSO
    results = run_loso_multicoeff(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        epochs=50  # Adjust as needed
    )
    
    print("\n✅ Training complete!")
    print(f"Models saved to: {OUTPUT_DIR}")
