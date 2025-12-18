"""
LOSO SOTA Comparison - Multi-Modal UDE vs Baselines
----------------------------------------------------
Evaluates UDE vs RF/Ridge/LSTM on all 15 folds
Uses pre-trained UDE models, trains baselines from scratch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_model import UDE
from src.models.lstm_baseline import LSTMBaseline
from src.utils import StressDataset

def get_all_subject_files(data_dir):
    """Get all processed subject files"""
    files = sorted([f for f in os.listdir(data_dir) if f.startswith('u_wesad_') and f.endswith('.csv')])
    return [os.path.join(data_dir, f) for f in files]

def prepare_ml_features(data_files):
    """Prepare simple features for ML baselines (last 10 points)"""
    all_features = []
    all_targets = []
    for file in data_files:
        df = pd.read_csv(file)
        for i in range(10, len(df)):
            # Use last 10 stress + workload values
            feat = df['stress'].values[i-10:i].tolist() + df['workload'].values[i-10:i].tolist()
            all_features.append(feat)
            all_targets.append(df['stress'].values[i])
    return np.array(all_features), np.array(all_targets)

def prepare_lstm_data(data_files, seq_len=60):
    """Prepare sequences for LSTM"""
    all_inputs = []
    all_targets = []
    for file in data_files:
        df = pd.read_csv(file)
        for i in range(seq_len, len(df)):
            seq = np.column_stack([df['stress'].values[i-seq_len:i], 
                                  df['workload'].values[i-seq_len:i]])
            all_inputs.append(seq)
            all_targets.append(df['stress'].values[i])
    return np.array(all_inputs), np.array(all_targets)

def run_partial_loso_comparison():
    print("="*70)
    print("LOSO SOTA COMPARISON - All 15 Folds")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    models_dir = os.path.join(base_dir, 'results', 'loso_models')
    
    all_files = get_all_subject_files(data_dir)
    n_folds = len(all_files)
    print(f"Testing on {n_folds} subjects")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    results = []
    
    for fold_idx in range(n_folds):
        test_file = all_files[fold_idx]
        train_files = [f for i, f in enumerate(all_files) if i != fold_idx]
        subject_name = os.path.basename(test_file)
        
        print(f"Fold {fold_idx+1}/15: {subject_name}")
        print("-" * 50)
        
        # 1. UDE (Load pre-trained model)
        model_path = os.path.join(models_dir, f'ude_fold_{fold_idx+1}.pth')
        if not os.path.exists(model_path):
            print(f"  ⚠️ Model not found, skipping fold")
            continue
            
        ude_model = UDE().to(device)
        ude_model.load_state_dict(torch.load(model_path, map_location=device))
        ude_model.eval()
        
        test_dataset = StressDataset(test_file, seq_len=60)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # UDE evaluation
        ude_mse = 0
        count = 0
        with torch.no_grad():
            for batch in test_loader:
                t_grid = batch['t'][0].to(device)
                y_true = batch['y'].to(device)
                features = batch['features'].to(device)
                y0 = y_true[:, 0, :]
                
                ude_model.set_current_batch(t_grid, features)
                y_pred = odeint(ude_model, y0, t_grid, method='dopri5', rtol=1e-3, atol=1e-4)
                y_pred = y_pred.permute(1, 0, 2)
                
                ude_mse += torch.mean((y_pred - y_true)**2).item() * len(y_true)
                count += len(y_true)
        
        ude_mse /= count
        print(f"  UDE: {ude_mse:.6f}")
        
        # 2. ML Baselines (RF, Ridge)
        X_train, y_train = prepare_ml_features(train_files)
        X_test, y_test = prepare_ml_features([test_file])
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_mse = mean_squared_error(y_test, rf.predict(X_test))
        print(f"  RF:  {rf_mse:.6f}")
        
        # Ridge
        ridge = Ridge()
        ridge.fit(X_train, y_train)
        ridge_mse = mean_squared_error(y_test, ridge.predict(X_test))
        print(f"  Ridge: {ridge_mse:.6f}")
        
        # 3. LSTM (Train quickly)
        X_lstm_train, y_lstm_train = prepare_lstm_data(train_files)
        X_lstm_test, y_lstm_test = prepare_lstm_data([test_file])
        
        lstm_model = LSTMBaseline(input_size=2, hidden_size=64, num_layers=2).to(device)
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Quick training
        X_train_t = torch.tensor(X_lstm_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_lstm_train, dtype=torch.float32).to(device)
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        
        lstm_model.train()
        for epoch in range(10):  # Quick training
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = lstm_model(X_batch)[:, -1, 0]  # Last timestep
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        lstm_model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_lstm_test, dtype=torch.float32).to(device)
            lstm_pred = lstm_model(X_test_t)[:, -1, 0].cpu().numpy()
        lstm_mse = mean_squared_error(y_lstm_test, lstm_pred)
        print(f"  LSTM: {lstm_mse:.6f}\n")
        
        results.append({
            'Fold': fold_idx + 1,
            'Subject': subject_name,
            'UDE': ude_mse,
            'RF': rf_mse,
            'Ridge': ridge_mse,
            'LSTM': lstm_mse
        })

    # Summary
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("AVERAGE MSE:")
    print(f"{'='*70}")
    for col in ['UDE', 'RF', 'Ridge', 'LSTM']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"{col:10s}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Save
    out_dir = os.path.join(base_dir, 'results', 'sota_comparison')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'loso_sota_results.csv'), index=False)
    print(f"\n✅ Saved to: {os.path.join(out_dir, 'loso_sota_results.csv')}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['UDE', 'RF', 'Ridge', 'LSTM']
    means = [df[m].mean() for m in methods]
    stds = [df[m].std() for m in methods]
    
    bars = ax.bar(methods, means, yerr=stds, capsize=5, alpha=0.7)
    bars[0].set_color('green')  # UDE
    
    ax.set_ylabel('MSE (lower is better)', fontsize=12)
    ax.set_title('LOSO Cross-Val: UDE vs Baselines (15 Folds)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sota_comparison.png'), dpi=150)
    print(f"✅ Plot saved to: {os.path.join(out_dir, 'sota_comparison.png')}")

if __name__ == "__main__":
    run_partial_loso_comparison()
