"""
Fair SOTA Comparison: Trajectory Forecasting
=============================================
ALL models solve the SAME task: predict S(1)...S(T) from S(0) + physiology.
No model sees past stress values as input features.

This is the FAIR comparison that demonstrates UDE's genuine advantage:
- UDE: Natural ODE integration from S(0)
- LSTM/Ridge/RF: Must autoregressively unroll (error amplifies)

Usage:
    python -m src.evaluation.fair_comparison
"""
import torch
import torch.nn as nn
import torch.optim as optim
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
from src.utils import StressDataset, FEATURE_COLUMNS

# ============================================================================
# LSTM Baseline (for trajectory forecasting)
# ============================================================================
class LSTMTrajectory(nn.Module):
    """LSTM for 1-step prediction (will be unrolled autoregressively for fair comparison)"""
    def __init__(self, input_size=18, hidden_size=64, num_layers=2):
        super().__init__()
        # Input: physiological features only (no past stress!)
        self.lstm = nn.LSTM(input_size + 1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, features_seq, stress_seq):
        """
        features_seq: (batch, seq_len, n_features) — physiological features
        stress_seq: (batch, seq_len, 1) — stress history (for teacher forcing during training)
        Returns: (batch, seq_len, 1) — predicted stress at each step
        """
        x = torch.cat([stress_seq, features_seq], dim=-1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


# ============================================================================
# Helper: Prepare data for fair comparison
# ============================================================================
def prepare_trajectory_data(data_files, seq_len=60):
    """
    Prepare data for trajectory forecasting.
    Returns features and stress sequences WITHOUT using past stress as input.
    """
    all_features = []
    all_stress = []
    
    for file in data_files:
        df = pd.read_csv(file)
        
        # Only use normalized physiological features
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        if len(feature_cols) == 0:
            print(f"  WARNING: No feature columns found in {file}")
            continue
            
        features = df[feature_cols].values.astype(np.float32)
        stress = df['stress'].values.astype(np.float32)
        
        # Create non-overlapping sequences for fair evaluation
        for i in range(0, len(df) - seq_len, seq_len):
            all_features.append(features[i:i+seq_len])
            all_stress.append(stress[i:i+seq_len])
    
    return np.array(all_features), np.array(all_stress)


# ============================================================================
# Fair comparison: All models predict full trajectory
# ============================================================================
def run_fair_comparison():
    """
    Fair SOTA comparison where ALL models predict full S(0)→S(T) trajectories.
    No model gets past stress values as input features.
    """
    print("=" * 70)
    print("FAIR TRAJECTORY FORECASTING COMPARISON")
    print("=" * 70)
    print("Task: Predict S(1)...S(60) from S(0) + physiology ONLY")
    print("NO model sees past stress values as input features.\n")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    models_dir = os.path.join(base_dir, 'results', 'loso_models')
    out_dir = os.path.join(base_dir, 'results', 'fair_comparison')
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all subject files
    all_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.startswith('u_wesad_') and f.endswith('.csv')
    ])
    
    n_folds = len(all_files)
    print(f"Subjects: {n_folds}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    seq_len = 60
    results = []
    
    for fold_idx in range(n_folds):
        test_file = all_files[fold_idx]
        train_files = [f for i, f in enumerate(all_files) if i != fold_idx]
        subject_name = os.path.basename(test_file)
        
        print(f"\nFold {fold_idx+1}/{n_folds}: {subject_name}")
        print("-" * 50)
        
        # Prepare data
        X_train, y_train = prepare_trajectory_data(train_files, seq_len=seq_len)
        X_test, y_test = prepare_trajectory_data([test_file], seq_len=seq_len)
        
        if len(X_test) == 0:
            print("  No test sequences, skipping")
            continue
        
        n_features = X_train.shape[2]
        
        # ==================================================================
        # METHOD 1: UDE (Natural trajectory via ODE integration)
        # ==================================================================
        model_path = os.path.join(models_dir, f'ude_fold_{fold_idx+1}.pth')
        if os.path.exists(model_path):
            ude_model = UDE(num_features=n_features).to(device)
            ude_model.load_state_dict(torch.load(model_path, map_location=device))
            ude_model.eval()
            
            ude_mses = []
            with torch.no_grad():
                for seq_idx in range(len(X_test)):
                    features = torch.FloatTensor(X_test[seq_idx:seq_idx+1]).to(device)
                    stress_true = y_test[seq_idx]
                    t = torch.arange(seq_len, dtype=torch.float32).to(device)
                    y0 = torch.tensor([stress_true[0]], dtype=torch.float32).to(device)
                    
                    ude_model.set_current_batch(t, features)
                    y_pred = odeint(ude_model, y0, t, method='dopri5', rtol=1e-3, atol=1e-4)
                    y_pred = y_pred.squeeze().cpu().numpy()
                    
                    ude_mses.append(mean_squared_error(stress_true, y_pred))
            
            ude_mse = np.mean(ude_mses)
        else:
            print(f"  UDE model not found at {model_path}")
            ude_mse = float('nan')
        
        print(f"  UDE (ODE trajectory):       MSE = {ude_mse:.6f}")
        
        # ==================================================================
        # METHOD 2: Ridge (Autoregressive unrolling — NO past stress input)
        # ==================================================================
        # Train: predict S(t+1) from [S(t), features(t)] — only 1-step
        X_ridge_train = []
        y_ridge_train = []
        for seq_features, seq_stress in zip(X_train, y_train):
            for t in range(seq_len - 1):
                # Input: current stress (1) + current features (n_features)
                x = np.concatenate([[seq_stress[t]], seq_features[t]])
                X_ridge_train.append(x)
                y_ridge_train.append(seq_stress[t + 1])
        
        X_ridge_train = np.array(X_ridge_train)
        y_ridge_train = np.array(y_ridge_train)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_ridge_train, y_ridge_train)
        
        # Test: AUTOREGRESSIVE unrolling (errors compound!)
        ridge_mses = []
        for seq_idx in range(len(X_test)):
            seq_features = X_test[seq_idx]
            stress_true = y_test[seq_idx]
            
            predictions = [stress_true[0]]  # Start with true S(0)
            current_s = stress_true[0]
            
            for t in range(seq_len - 1):
                x = np.concatenate([[current_s], seq_features[t]]).reshape(1, -1)
                current_s = ridge.predict(x)[0]
                predictions.append(current_s)
            
            ridge_mses.append(mean_squared_error(stress_true, predictions))
        
        ridge_mse = np.mean(ridge_mses)
        print(f"  Ridge (autoregressive):     MSE = {ridge_mse:.6f}")
        
        # ==================================================================
        # METHOD 3: Random Forest (Autoregressive unrolling)
        # ==================================================================
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_ridge_train, y_ridge_train)  # Same training data as Ridge
        
        rf_mses = []
        for seq_idx in range(len(X_test)):
            seq_features = X_test[seq_idx]
            stress_true = y_test[seq_idx]
            
            predictions = [stress_true[0]]
            current_s = stress_true[0]
            
            for t in range(seq_len - 1):
                x = np.concatenate([[current_s], seq_features[t]]).reshape(1, -1)
                current_s = rf.predict(x)[0]
                predictions.append(current_s)
            
            rf_mses.append(mean_squared_error(stress_true, predictions))
        
        rf_mse = np.mean(rf_mses)
        print(f"  RF   (autoregressive):      MSE = {rf_mse:.6f}")
        
        # ==================================================================
        # METHOD 4: LSTM (Autoregressive unrolling)
        # ==================================================================
        lstm_model = LSTMTrajectory(input_size=n_features, hidden_size=64).to(device)
        lstm_opt = optim.Adam(lstm_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train LSTM with teacher forcing
        lstm_model.train()
        train_features_t = torch.FloatTensor(X_train).to(device)
        train_stress_t = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
        
        for epoch in range(20):
            for i in range(0, len(train_features_t), 32):
                batch_f = train_features_t[i:i+32]
                batch_s = train_stress_t[i:i+32]
                
                # Input stress is shifted (teacher forcing)
                input_stress = batch_s[:, :-1, :]
                target_stress = batch_s[:, 1:, :]
                input_features = batch_f[:, :-1, :]
                
                lstm_opt.zero_grad()
                pred = lstm_model(input_features, input_stress)
                loss = criterion(pred, target_stress)
                loss.backward()
                lstm_opt.step()
        
        # Test LSTM: autoregressive
        lstm_model.eval()
        lstm_mses = []
        with torch.no_grad():
            for seq_idx in range(len(X_test)):
                seq_features = torch.FloatTensor(X_test[seq_idx]).to(device)
                stress_true = y_test[seq_idx]
                
                predictions = [stress_true[0]]
                current_s = stress_true[0]
                
                for t in range(seq_len - 1):
                    f_t = seq_features[t:t+1].unsqueeze(0)  # (1, 1, n_features)
                    s_t = torch.tensor([[[current_s]]]).to(device)  # (1, 1, 1)
                    pred = lstm_model(f_t, s_t)
                    current_s = pred.item()
                    predictions.append(current_s)
                
                lstm_mses.append(mean_squared_error(stress_true, predictions))
        
        lstm_mse = np.mean(lstm_mses)
        print(f"  LSTM (autoregressive):      MSE = {lstm_mse:.6f}")
        
        # ==================================================================
        # METHOD 5: Naive Baseline (predict S(t) = S(0) for all t)
        # ==================================================================
        naive_mses = []
        for seq_idx in range(len(X_test)):
            stress_true = y_test[seq_idx]
            naive_pred = np.full_like(stress_true, stress_true[0])
            naive_mses.append(mean_squared_error(stress_true, naive_pred))
        
        naive_mse = np.mean(naive_mses)
        print(f"  Naive (constant S(0)):      MSE = {naive_mse:.6f}")
        
        results.append({
            'Fold': fold_idx + 1,
            'Subject': subject_name,
            'UDE': ude_mse,
            'Ridge_AR': ridge_mse,
            'RF_AR': rf_mse,
            'LSTM_AR': lstm_mse,
            'Naive': naive_mse,
        })
    
    # Summary
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("FAIR COMPARISON SUMMARY (Trajectory Forecasting)")
    print("=" * 70)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("AVERAGE MSE (Lower is Better)")
    print("=" * 70)
    for col in ['UDE', 'Ridge_AR', 'RF_AR', 'LSTM_AR', 'Naive']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"  {col:15s}: {mean_val:.6f} ± {std_val:.6f}")
    
    # Calculate improvement
    ude_mean = df['UDE'].mean()
    for baseline in ['Ridge_AR', 'RF_AR', 'LSTM_AR', 'Naive']:
        bl_mean = df[baseline].mean()
        if bl_mean > 0:
            improvement = (bl_mean - ude_mean) / bl_mean * 100
            print(f"\n  UDE vs {baseline}: {improvement:+.1f}% {'better' if improvement > 0 else 'worse'}")
    
    # Save
    df.to_csv(os.path.join(out_dir, 'fair_comparison_results.csv'), index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    means = df[['UDE', 'Ridge_AR', 'RF_AR', 'LSTM_AR', 'Naive']].mean()
    stds = df[['UDE', 'Ridge_AR', 'RF_AR', 'LSTM_AR', 'Naive']].std()
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#95a5a6']
    bars = ax.bar(means.index, means.values, yerr=stds.values, capsize=5, color=colors, alpha=0.85)
    
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Fair Trajectory Forecasting Comparison (60-step horizon)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fair_comparison_plot.png'), dpi=150)
    print(f"\n✅ Results saved to: {out_dir}")
    
    return df


if __name__ == "__main__":
    run_fair_comparison()
