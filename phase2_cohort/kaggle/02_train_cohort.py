"""
KAGGLE NOTEBOOK: Phase 2 — Cohort-Level MC-UDE Training
========================================================
Trains one MC-UDE model per cohort using POOLED data from
all subjects assigned to that cohort.

BEFORE RUNNING: Upload cohort_metadata.json from Phase 2 Step 1
and the normalized WESAD data.

Expected runtime: ~2-4 hours on P100 GPU
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
    'DATA_DIR': '/kaggle/input/wesad-normalized/normalized',
    'COHORT_META': '/kaggle/input/cohort-metadata/cohort_metadata.json',
    'OUTPUT_DIR': '/kaggle/working/cohort_ude_results',
    'SEQ_LEN': 60,
    'EPOCHS': 50,
    'BATCH_SIZE': 32,
    'LR': 0.001,
    'LAMBDA_L1': 0.001,
    'LAMBDA_PHYSICS': 0.01,
    'HIDDEN_DIM': 64,
    'NUM_FEATURES': 18,
}

FEATURE_COLUMNS = [
    'workload_norm', 'hrv_rmssd_norm', 'hrv_sdnn_norm', 'hrv_pnn50_norm',
    'hrv_lf_hf_norm', 'heart_rate_norm', 'eda_mean_norm', 'eda_std_norm',
    'eda_peaks_norm', 'resp_mean_norm', 'resp_std_norm', 'resp_rate_norm',
    'temp_mean_norm', 'temp_std_norm', 'activity_level_norm',
    'activity_std_norm', 'emg_mean_norm', 'emg_std_norm'
]

FEATURE_DISPLAY_NAMES = [
    'Workload', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF/HF',
    'Heart Rate', 'EDA_Mean', 'EDA_Std', 'EDA_Peaks',
    'Resp_Mean', 'Resp_Std', 'Resp_Rate',
    'Temp_Mean', 'Temp_Std', 'Activity_Mean', 'Activity_Std',
    'EMG_Mean', 'EMG_Std'
]

SUBJECT_IDS = [
    'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008',
    'S009', 'S010', 'S011', 'S013', 'S014', 'S015', 'S016', 'S017'
]

# Map subject ID to CSV filename
def subject_to_csv(sid):
    num = sid.replace('S', '')
    return f'u_wesad_{num}.csv'


# ============================================================================
# DATASET (same as Phase 1)
# ============================================================================
class StressDataset(Dataset):
    def __init__(self, csv_path, seq_len=60):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.feature_columns = [c for c in FEATURE_COLUMNS if c in self.df.columns]
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        self.stress = self.df['stress'].values.astype(np.float32)
        self.time = self.df['time'].values.astype(np.float32)
        self.num_features = len(self.feature_columns)
        self.n_sequences = max(0, (len(self.df) - seq_len) // (seq_len // 2))

    def __len__(self): return self.n_sequences

    def __getitem__(self, idx):
        start = idx * (self.seq_len // 2)
        end = start + self.seq_len
        return {
            't': torch.tensor(self.time[start:end] - self.time[start], dtype=torch.float32),
            'y': torch.tensor(self.stress[start:end], dtype=torch.float32).unsqueeze(-1),
            'features': torch.tensor(self.features[start:end], dtype=torch.float32),
        }


# ============================================================================
# MC-UDE MODEL (identical to Phase 1)
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
        self.current_t = t
        self.current_features = features

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

    def l1_loss(self, lambda_l1=0.001):
        return lambda_l1 * torch.sum(torch.abs(self.alphas))

    def physics_loss(self, lambda_physics=0.01):
        beta = self.beta
        low = F.relu(0.01 - beta)
        high = F.relu(beta - 5.0)
        return lambda_physics * (low + high).squeeze()

    def get_equation_string(self, threshold=0.01):
        beta = self.beta.item()
        alphas = self.alphas.detach().cpu().numpy()
        terms = [f"-{beta:.4f}·S"]
        for a, name in zip(alphas, FEATURE_DISPLAY_NAMES):
            if a > threshold: terms.append(f"+{a:.4f}·{name}")
        terms.append("+ NN(S, F)")
        return "dS/dt = " + " ".join(terms)

    def get_sparse_profile(self, threshold=0.01):
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
# TRAINING
# ============================================================================
def train_model(model, train_loader, val_loader, epochs, lr, device, config):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            t = batch['t'][0].to(device)
            y = batch['y'].to(device).squeeze(-1)
            features = batch['features'].to(device)
            y0 = y[:, 0]
            optimizer.zero_grad()
            model.set_current_batch(t, features)
            y_pred = odeint(model, y0, t, method='euler').permute(1, 0)
            mse = torch.mean((y_pred - y) ** 2)
            total = mse + model.l1_loss(config['LAMBDA_L1']) + model.physics_loss(config['LAMBDA_PHYSICS'])
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(mse.item())

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    t = batch['t'][0].to(device)
                    y = batch['y'].to(device).squeeze(-1)
                    features = batch['features'].to(device)
                    y0 = y[:, 0]
                    model.set_current_batch(t, features)
                    yp = odeint(model, y0, t, method='euler').permute(1, 0)
                    val_losses.append(torch.mean((yp - y) ** 2).item())
            n_act = sum(1 for a in model.alphas.detach().cpu().numpy() if a > 0.01)
            print(f"  Epoch {epoch+1}/{epochs}: Train={np.mean(losses):.6f} "
                  f"Val={np.mean(val_losses):.6f} β={model.beta.item():.4f} Active={n_act}/18")


# ============================================================================
# MAIN: Cohort Training
# ============================================================================
def run_cohort_training():
    config = CONFIG
    print("=" * 70)
    print("PHASE 2: COHORT-LEVEL MC-UDE TRAINING")
    print("=" * 70)

    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load cohort metadata
    with open(config['COHORT_META']) as f:
        cohort_meta = json.load(f)

    csv_files = sorted(glob(os.path.join(config['DATA_DIR'], '*.csv')))
    print(f"Found {len(csv_files)} subject CSV files\n")

    # Build subject -> CSV path map
    csv_map = {}
    for csv_path in csv_files:
        basename = os.path.basename(csv_path)
        # Extract subject number from u_wesad_XXX.csv
        num = basename.replace('u_wesad_', '').replace('.csv', '')
        sid = f"S{num}"
        csv_map[sid] = csv_path

    all_results = []

    for cohort_label, cohort_info in cohort_meta['cohorts'].items():
        cohort_name = cohort_info['name']
        members = cohort_info['members']
        print(f"\n{'=' * 70}")
        print(f"COHORT: {cohort_name} ({len(members)} subjects)")
        print(f"Members: {', '.join(members)}")
        print(f"{'=' * 70}")

        if len(members) < 2:
            print("  Skipping — need at least 2 subjects for leave-one-out")
            continue

        # Leave-one-out within cohort
        for held_out in members:
            fold_name = f"cohort{cohort_label}_{held_out}"
            model_path = os.path.join(config['OUTPUT_DIR'], f'{fold_name}.pth')

            if os.path.exists(model_path):
                print(f"\n  {held_out}: Already done, skipping")
                continue

            print(f"\n  Training (hold out {held_out})...")

            # Pool all members EXCEPT held_out
            train_members = [m for m in members if m != held_out]
            train_files = [csv_map[m] for m in train_members if m in csv_map]
            test_file = csv_map.get(held_out)

            if not test_file or not train_files:
                print(f"  WARNING: Missing CSV for {held_out}, skipping")
                continue

            # Create datasets
            train_datasets = [StressDataset(f, seq_len=config['SEQ_LEN']) for f in train_files]
            test_dataset = StressDataset(test_file, seq_len=config['SEQ_LEN'])

            train_data = []
            for ds in train_datasets:
                for i in range(len(ds)):
                    train_data.append(ds[i])

            train_loader = DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

            print(f"  Train: {len(train_data)} sequences from {len(train_members)} subjects")
            print(f"  Test: {len(test_dataset)} sequences from {held_out}")

            # Train
            model = MCUDE(hidden_dim=config['HIDDEN_DIM'],
                         num_features=config['NUM_FEATURES']).to(device)
            train_model(model, train_loader, test_loader,
                       config['EPOCHS'], config['LR'], device, config)

            # Evaluate
            model.eval()
            test_losses = []
            with torch.no_grad():
                for batch in test_loader:
                    t = batch['t'][0].to(device)
                    y = batch['y'].to(device).squeeze(-1)
                    features = batch['features'].to(device)
                    y0 = y[:, 0]
                    model.set_current_batch(t, features)
                    yp = odeint(model, y0, t, method='euler').permute(1, 0)
                    test_losses.append(torch.mean((yp - y) ** 2).item())
            test_mse = np.mean(test_losses)

            profile = model.get_sparse_profile()
            equation = model.get_equation_string()

            print(f"  ✅ Test MSE: {test_mse:.6f}")
            print(f"  Equation: {equation}")

            # Save
            torch.save(model.state_dict(), model_path)
            profile_data = {**profile, 'equation': equation, 'test_mse': test_mse}
            with open(os.path.join(config['OUTPUT_DIR'], f'{fold_name}_profile.json'), 'w') as f:
                json.dump(profile_data, f, indent=2)

            all_results.append({
                'Cohort': cohort_name,
                'Cohort_Label': cohort_label,
                'Held_Out': held_out,
                'Test_MSE': test_mse,
                'Beta': profile['beta'],
                'N_Active': profile['n_active'],
            })

            # Checkpoint
            pd.DataFrame(all_results).to_csv(
                os.path.join(config['OUTPUT_DIR'], 'cohort_training_results.csv'), index=False)

    # Also train a FULL cohort model (no held-out) for production use
    print(f"\n{'=' * 70}")
    print("TRAINING FULL COHORT MODELS (no held-out)")
    print(f"{'=' * 70}")

    for cohort_label, cohort_info in cohort_meta['cohorts'].items():
        cohort_name = cohort_info['name']
        members = cohort_info['members']
        model_path = os.path.join(config['OUTPUT_DIR'], f'cohort{cohort_label}_full.pth')

        if os.path.exists(model_path):
            print(f"\n  {cohort_name}: Already done")
            continue

        print(f"\n  Training full {cohort_name} model ({len(members)} subjects)...")
        all_files = [csv_map[m] for m in members if m in csv_map]

        all_datasets = [StressDataset(f, seq_len=config['SEQ_LEN']) for f in all_files]
        all_data = []
        for ds in all_datasets:
            for i in range(len(ds)):
                all_data.append(ds[i])

        # Use 90/10 split for train/val
        split = int(0.9 * len(all_data))
        train_loader = DataLoader(all_data[:split], batch_size=config['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(all_data[split:], batch_size=config['BATCH_SIZE'], shuffle=False)

        model = MCUDE(hidden_dim=config['HIDDEN_DIM'],
                     num_features=config['NUM_FEATURES']).to(device)
        train_model(model, train_loader, val_loader,
                   config['EPOCHS'], config['LR'], device, config)

        equation = model.get_equation_string()
        profile = model.get_sparse_profile()
        print(f"  ✅ Cohort Equation: {equation}")

        torch.save(model.state_dict(), model_path)
        profile_data = {**profile, 'equation': equation, 'cohort_name': cohort_name}
        with open(os.path.join(config['OUTPUT_DIR'], f'cohort{cohort_label}_full_profile.json'), 'w') as f:
            json.dump(profile_data, f, indent=2)

    # Summary
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{'=' * 70}")
        print("COHORT TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(df.to_string(index=False))
        for cohort in df['Cohort'].unique():
            sub = df[df['Cohort'] == cohort]
            print(f"\n  {cohort}: MSE = {sub['Test_MSE'].mean():.6f} ± {sub['Test_MSE'].std():.6f}")

    print(f"\n✅ All results saved to {config['OUTPUT_DIR']}")


if __name__ == "__main__":
    run_cohort_training()
