"""
KAGGLE NOTEBOOK: Fair Trajectory Comparison
=============================================
Runs after training is complete (01_train.py).
Compares MC-UDE vs Ridge/RF/LSTM on SAME trajectory forecasting task.

Upload trained models from 01_train.py, then run this.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import os
import json
from glob import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    'DATA_DIR': '/kaggle/input/wesad-normalized/normalized',
    'MODELS_DIR': '/kaggle/input/mc-ude-models/mc_ude_results',  # Trained models from 01_train.py
    'OUTPUT_DIR': '/kaggle/working/fair_comparison',
    'SEQ_LEN': 60,
}

FEATURE_COLUMNS = [
    'workload_norm', 'hrv_rmssd_norm', 'hrv_sdnn_norm', 'hrv_pnn50_norm',
    'hrv_lf_hf_norm', 'heart_rate_norm', 'eda_mean_norm', 'eda_std_norm',
    'eda_peaks_norm', 'resp_mean_norm', 'resp_std_norm', 'resp_rate_norm',
    'temp_mean_norm', 'temp_std_norm', 'activity_level_norm',
    'activity_std_norm', 'emg_mean_norm', 'emg_std_norm'
]


# ============================================================================
# MC-UDE Model (must match training model exactly)
# ============================================================================
class MCUDE(nn.Module):
    def __init__(self, hidden_dim=64, num_features=18):
        super().__init__()
        input_dim = 1 + num_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))
        self._alphas_raw = nn.Parameter(torch.ones(num_features) * (-2.2))
        self.current_features = None
        self.current_t = None
        self.num_features = num_features

    @property
    def beta(self): return F.softplus(self._beta_raw)
    @property
    def alphas(self): return F.softplus(self._alphas_raw)

    def set_current_batch(self, t, features):
        self.current_t = t; self.current_features = features

    def forward(self, t, y):
        S = y
        t_idx = torch.argmin(torch.abs(self.current_t - t))
        features = self.current_features[:, t_idx, :]
        recovery = -self.beta * S
        feature_contribution = torch.sum(self.alphas * features, dim=-1)
        f_known = recovery + feature_contribution
        S_expanded = S.unsqueeze(-1) if S.dim() == 1 else S
        nn_in = torch.cat([S_expanded, features], dim=-1)
        f_nn = self.net(nn_in)
        if S.dim() == 1: f_nn = f_nn.squeeze(-1)
        return f_known + f_nn


# ============================================================================
# LSTM Baseline
# ============================================================================
class LSTMBaseline(nn.Module):
    def __init__(self, input_size=19, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


# ============================================================================
# Data preparation
# ============================================================================
def prepare_data(data_files, seq_len=60):
    """Prepare trajectory data — no past stress as input.
    Also extracts actual time values to match training time scale."""
    all_features, all_stress, all_time = [], [], []
    for f in data_files:
        df = pd.read_csv(f)
        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        features = df[feat_cols].values.astype(np.float32)
        stress = df['stress'].values.astype(np.float32)
        time_vals = df['time'].values.astype(np.float32)
        for i in range(0, len(df) - seq_len, seq_len):
            all_features.append(features[i:i+seq_len])
            all_stress.append(stress[i:i+seq_len])
            t_seq = time_vals[i:i+seq_len] - time_vals[i]  # Zero-base like training
            all_time.append(t_seq)
    return np.array(all_features), np.array(all_stress), np.array(all_time)


# ============================================================================
# Main comparison
# ============================================================================
def run_fair_comparison():
    config = CONFIG
    print("=" * 70)
    print("FAIR TRAJECTORY FORECASTING COMPARISON")
    print("=" * 70)
    print("ALL models predict S(0)→S(60) trajectories. No past stress as input.\n")

    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    csv_files = sorted(glob(os.path.join(config['DATA_DIR'], '*.csv')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Subjects: {len(csv_files)}, Device: {device}\n")

    seq_len = config['SEQ_LEN']
    results = []

    for fold_idx in range(len(csv_files)):
        test_file = csv_files[fold_idx]
        train_files = [f for i, f in enumerate(csv_files) if i != fold_idx]
        subject = os.path.basename(test_file)
        print(f"\nFold {fold_idx+1}/{len(csv_files)}: {subject}")

        X_train, y_train, t_train = prepare_data(train_files, seq_len)
        X_test, y_test, t_test = prepare_data([test_file], seq_len)
        if len(X_test) == 0: continue
        n_features = X_train.shape[2]

        # --- MC-UDE ---
        model_path = os.path.join(config['MODELS_DIR'], f'mcude_fold_{fold_idx+1}.pth')
        if os.path.exists(model_path):
            model = MCUDE(num_features=n_features).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            ude_mses = []
            with torch.no_grad():
                for si in range(len(X_test)):
                    features = torch.FloatTensor(X_test[si:si+1]).to(device)
                    # CRITICAL: Use actual time values from CSV, NOT integer indices.
                    # Training uses CSV time (spacing ~0.017), so evaluation must match.
                    # Using [0,1,2,...,59] would be ~60x larger time scale → ODE blows up.
                    t = torch.FloatTensor(t_test[si]).to(device)
                    y0 = torch.tensor([y_test[si][0]]).to(device)
                    model.set_current_batch(t, features)
                    yp = odeint(model, y0, t, method='euler')
                    ude_mses.append(mean_squared_error(y_test[si], yp.squeeze().cpu().numpy()))
            ude_mse = np.mean(ude_mses)
        else:
            ude_mse = float('nan')
        print(f"  UDE:   {ude_mse:.6f}")

        # --- Ridge (autoregressive) ---
        X_r, y_r = [], []
        for sf, ss in zip(X_train, y_train):
            for t in range(seq_len - 1):
                X_r.append(np.concatenate([[ss[t]], sf[t]]))
                y_r.append(ss[t+1])
        ridge = Ridge(alpha=1.0).fit(np.array(X_r), np.array(y_r))

        ridge_mses = []
        for si in range(len(X_test)):
            preds = [y_test[si][0]]
            cs = y_test[si][0]
            for t in range(seq_len-1):
                cs = ridge.predict(np.concatenate([[cs], X_test[si][t]]).reshape(1,-1))[0]
                preds.append(cs)
            ridge_mses.append(mean_squared_error(y_test[si], preds))
        ridge_mse = np.mean(ridge_mses)
        print(f"  Ridge: {ridge_mse:.6f}")

        # --- RF (autoregressive) ---
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(np.array(X_r), np.array(y_r))

        rf_mses = []
        for si in range(len(X_test)):
            preds = [y_test[si][0]]
            cs = y_test[si][0]
            for t in range(seq_len-1):
                cs = rf.predict(np.concatenate([[cs], X_test[si][t]]).reshape(1,-1))[0]
                preds.append(cs)
            rf_mses.append(mean_squared_error(y_test[si], preds))
        rf_mse = np.mean(rf_mses)
        print(f"  RF:    {rf_mse:.6f}")

        # --- Naive baseline ---
        naive_mse = np.mean([mean_squared_error(y_test[si], np.full(seq_len, y_test[si][0])) for si in range(len(y_test))])
        print(f"  Naive: {naive_mse:.6f}")

        results.append({'Fold': fold_idx+1, 'Subject': subject,
                       'UDE': ude_mse, 'Ridge_AR': ridge_mse, 'RF_AR': rf_mse, 'Naive': naive_mse})

    df = pd.DataFrame(results)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for col in ['UDE', 'Ridge_AR', 'RF_AR', 'Naive']:
        print(f"  {col:12s}: {df[col].mean():.6f} ± {df[col].std():.6f}")

    df.to_csv(os.path.join(config['OUTPUT_DIR'], 'fair_comparison.csv'), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    means = df[['UDE', 'Ridge_AR', 'RF_AR', 'Naive']].mean()
    means.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#3498db', '#95a5a6'], alpha=0.85)
    ax.set_ylabel('MSE'); ax.set_title('Fair Trajectory Forecasting (60-step)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['OUTPUT_DIR'], 'comparison_plot.png'), dpi=150)
    print(f"\n✅ Saved to {config['OUTPUT_DIR']}")
    return df

if __name__ == "__main__":
    run_fair_comparison()
