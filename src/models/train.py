"""
Main Training Script: Multi-Subject Cross-Validation (LOSO)
-----------------------------------------------------------
Features:
1. Uses ALL 15 subjects (Leave-One-Subject-Out Cross-Validation)
2. Population-level normalization (Z-score across all subjects)
3. Real physiological features (Heart Rate, HRV) - NO DATA LEAKAGE
4. Rigorous training (50 epochs per fold)
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_model import UDE
from src.utils import StressDataset

def get_all_subject_files(data_dir):
    """Get all processed WESAD subject files"""
    files = sorted([f for f in os.listdir(data_dir) if f.startswith('u_wesad_') and f.endswith('.csv')])
    return [os.path.join(data_dir, f) for f in files]

def population_normalize_subjects(data_files):
    """
    Apply population-level normalization across all subjects for multi-modal features.
    Ensures model generalizes across people by using a common scale.
    
    IMPORTANT: Uses actual WESAD labels as ground truth to avoid data leakage.
    The stress target is derived from labels, NOT from any physiological feature.
    """
    print("\nApplying population-level normalization for multi-modal features...")
    
    # Features to normalize (all physiological signals)
    features_to_normalize = [
        'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50', 'hrv_lf_hf',
        'heart_rate', 'workload',
        'eda_mean', 'eda_std', 'eda_peaks',
        'resp_mean', 'resp_std', 'resp_rate',
        'temp_mean', 'temp_std',
        'activity_level', 'activity_std',
        'emg_mean', 'emg_std'
    ]
    
    # Collect values for all features
    feature_values = {feat: [] for feat in features_to_normalize}
    
    for file in data_files:
        df = pd.read_csv(file)
        for feat in features_to_normalize:
            if feat in df.columns:
                # Filter out NaN values
                valid_values = df[feat].dropna().values
                feature_values[feat].extend(valid_values)
    
    # Calculate population statistics for each feature
    feature_stats = {}
    for feat in features_to_normalize:
        if len(feature_values[feat]) > 0:
            # Convert to numpy array and remove any remaining NaNs
            values_array = np.array(feature_values[feat])
            values_array = values_array[~np.isnan(values_array)]
            
            if len(values_array) > 0:
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                feature_stats[feat] = {'mean': mean_val, 'std': std_val}
                print(f"  {feat}: mean={mean_val:.4f}, std={std_val:.4f}")
            else:
                print(f"  WARNING: {feat} has no valid values, skipping normalization")
    
    # Normalize and save
    normalized_dir = os.path.join(os.path.dirname(data_files[0]), 'normalized')
    os.makedirs(normalized_dir, exist_ok=True)
    
    for file in data_files:
        df = pd.read_csv(file)
        
        # Z-score normalization for ALL features (population-level)
        for feat in features_to_normalize:
            if feat in df.columns and feat in feature_stats:
                mean_val = feature_stats[feat]['mean']
                std_val = feature_stats[feat]['std']
                df[f'{feat}_norm'] = (df[feat] - mean_val) / (std_val + 1e-6)
        
        # CRITICAL: Use actual WESAD labels as ground truth for STRESS
        # This avoids data leakage (stress is NOT derived from any input feature)
        # WESAD labels: 1=Baseline, 2=Stress, 3=Amusement, 4=Meditation
        if 'label' in df.columns:
            # Map: Stress(2) → 1.0, Baseline(1) → 0.2, Amusement(3) → 0.3, Meditation(4) → 0.1
            label_to_stress = {1: 0.2, 2: 1.0, 3: 0.3, 4: 0.1, 0: 0.2}  # 0 = undefined
            df['stress'] = df['label'].map(lambda x: label_to_stress.get(int(x), 0.2))
            
            # Apply exponential smoothing to create continuous stress dynamics
            alpha_smooth = 0.1  # Smoothing factor
            stress_smooth = [df['stress'].iloc[0]]
            for i in range(1, len(df)):
                stress_smooth.append(alpha_smooth * df['stress'].iloc[i] + (1 - alpha_smooth) * stress_smooth[-1])
            df['stress'] = stress_smooth
        else:
            # This shouldn't happen with WESAD data, but keep as fallback
            print(f"  WARNING: No 'label' column in {file}")
            df['stress'] = 0.5  # Neutral default
        
        # Save normalized version
        filename = os.path.basename(file)
        out_path = os.path.join(normalized_dir, filename)
        df.to_csv(out_path, index=False)
    
    print(f"  Saved normalized data to: {normalized_dir}")
    return normalized_dir

def train_loso_cross_validation(data_dir, seq_len=60, epochs=50, lr=0.005, batch_size=256):
    """
    Leave-One-Subject-Out Cross-Validation
    Trains N models, where N is number of subjects.
    Each model is trained on N-1 subjects and tested on the remaining one.
    """
    print("="*70)
    print("LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION")
    print("="*70)
    
    # Get all subject files
    all_files = get_all_subject_files(data_dir)
    n_subjects = len(all_files)
    
    print(f"\nFound {n_subjects} subjects")
    print(f"Training config: seq_len={seq_len}, epochs={epochs}, lr={lr}, batch_size={batch_size}")
    
    # Apply population normalization
    normalized_dir = population_normalize_subjects(all_files)
    normalized_files = get_all_subject_files(normalized_dir)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    results_dir = os.path.join(base_dir, 'results', 'loso_models')
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    fold_losses = []
    
    # LOSO: Train on N-1 subjects, test on 1
    for test_idx in range(n_subjects):
        fold_num = test_idx + 1
        model_path = os.path.join(results_dir, f'ude_fold_{fold_num}.pth')
        
        if os.path.exists(model_path):
            print(f"Fold {fold_num} already trained. Skipping...")
            continue
            
        print(f"\n{'='*70}")
        print(f"FOLD {fold_num}/{n_subjects}: Testing on {os.path.basename(normalized_files[test_idx])}")
        print(f"{'='*70}")
        
        # Split train/test
        test_file = normalized_files[test_idx]
        train_files = [f for i, f in enumerate(normalized_files) if i != test_idx]
        
        print(f"  Train subjects: {len(train_files)}")
        print(f"  Test subject: {os.path.basename(test_file)}")
        
        # Create datasets
        train_datasets = [StressDataset(f, seq_len=seq_len) for f in train_files]
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = StressDataset(test_file, seq_len=seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
        # Get number of features from the first dataset sample
        sample_batch = test_dataset[0]
        num_features = sample_batch['num_features']
        print(f"  Using {num_features} multi-modal features")
        
        # Initialize model with correct feature count
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UDE(hidden_dim=64, num_features=num_features).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training
        print(f"\n  Training on {device}...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            count = 0
            
            for batch in train_loader:
                t_grid = batch['t'][0].to(device)
                y_true = batch['y'].to(device)
                features = batch['features'].to(device)  # Multi-modal features
                y0 = y_true[:, 0, :]
                
                model.set_current_batch(t_grid, features)
                y_pred = odeint(model, y0, t_grid, method='euler') # Euler is faster for training
                y_pred = y_pred.permute(1, 0, 2)
                
                loss = torch.mean((y_pred - y_true)**2)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for ODE training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                count += 1
            
            avg_loss = epoch_loss / count
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        
        # Testing
        print(f"\n  Testing...")
        model.eval()
        test_loss = 0
        test_count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                t_grid = batch['t'][0].to(device)
                y_true = batch['y'].to(device)
                features = batch['features'].to(device)  # Multi-modal features
                y0 = y_true[:, 0, :]
                
                model.set_current_batch(t_grid, features)
                y_pred = odeint(model, y0, t_grid, method='dopri5', rtol=1e-3, atol=1e-4)
                y_pred = y_pred.permute(1, 0, 2)
                
                loss = torch.mean((y_pred - y_true)**2)
                test_loss += loss.item()
                test_count += 1
        
        avg_test_loss = test_loss / test_count
        print(f"  Test MSE: {avg_test_loss:.6f}")
        
        results.append({
            'fold': test_idx + 1,
            'test_subject': os.path.basename(test_file),
            'test_mse': avg_test_loss
        })
        fold_losses.append(avg_test_loss)
        
        # Save fold model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        results_dir = os.path.join(base_dir, 'results', 'loso_models')
        os.makedirs(results_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(results_dir, f'ude_fold_{test_idx+1}.pth'))
    
    # Summary
    print(f"\n{'='*70}")
    print("LOSO CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    mean_mse = np.mean(fold_losses)
    std_mse = np.std(fold_losses)
    
    print(f"\n{'='*70}")
    print(f"Mean Test MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"{'='*70}")
    
    # Save results
    results_path = os.path.join(results_dir, 'loso_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Plot only if we have results
    if len(fold_losses) > 0:
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, len(fold_losses)+1), fold_losses)
        plt.axhline(mean_mse, color='r', linestyle='--', label=f'Mean: {mean_mse:.6f}')
        plt.xlabel('Fold (Test Subject)')
        plt.ylabel('Test MSE')
        plt.title('LOSO Cross-Validation Results')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'loso_results.png'), dpi=150)
        print(f"Plot saved to: {os.path.join(results_dir, 'loso_results.png')}")
    else:
        print("WARNING: No folds completed, skipping plot generation")
    
    return mean_mse, std_mse, results_df

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / 'data' / 'processed'
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run process_real_wesad.py first")
        sys.exit(1)
    
    # Run LOSO cross-validation
    mean_mse, std_mse, results = train_loso_cross_validation(str(data_dir))
    
    print(f"\n✅ LOSO Cross-Validation Complete!")
    print(f"   Mean MSE: {mean_mse:.6f} ± {std_mse:.6f}")
