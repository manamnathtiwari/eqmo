"""
KAGGLE NOTEBOOK: Phase 2 — Evaluate Cohort vs Individual
=========================================================
Compares cohort-level UDE performance against individual UDE
and baseline models to quantify the cost of generalization.

Also trains a cold-start cohort assignment classifier.

Run AFTER: 02_train_cohort.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import pandas as pd
import numpy as np
import os
import json
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    'DATA_DIR': '/kaggle/input/wesad-normalized/normalized',
    'INDIVIDUAL_MODELS': '/kaggle/input/mc-ude-models/mc_ude_results',
    'COHORT_MODELS': '/kaggle/input/cohort-models/cohort_ude_results',
    'COHORT_META': '/kaggle/input/cohort-metadata/cohort_metadata.json',
    'OUTPUT_DIR': '/kaggle/working/cohort_evaluation',
    'SEQ_LEN': 60,
}

FEATURE_COLUMNS = [
    'workload_norm', 'hrv_rmssd_norm', 'hrv_sdnn_norm', 'hrv_pnn50_norm',
    'hrv_lf_hf_norm', 'heart_rate_norm', 'eda_mean_norm', 'eda_std_norm',
    'eda_peaks_norm', 'resp_mean_norm', 'resp_std_norm', 'resp_rate_norm',
    'temp_mean_norm', 'temp_std_norm', 'activity_level_norm',
    'activity_std_norm', 'emg_mean_norm', 'emg_std_norm'
]

SUBJECT_IDS = [
    'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008',
    'S009', 'S010', 'S011', 'S013', 'S014', 'S015', 'S016', 'S017'
]


# ============================================================================
# MC-UDE MODEL (same architecture)
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
# DATA PREPARATION
# ============================================================================
def prepare_data(data_files, seq_len=60):
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
            all_time.append(time_vals[i:i+seq_len] - time_vals[i])
    return np.array(all_features), np.array(all_stress), np.array(all_time)


def extract_baselines(csv_path, n_minutes=10):
    """Extract baseline physiological features from first N minutes for cold-start."""
    df = pd.read_csv(csv_path)
    # Estimate rows for N minutes (time step ~0.017 min)
    n_rows = min(int(n_minutes / 0.017), len(df) // 2)
    subset = df.iloc[:n_rows]
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    baselines = {}
    for col in feat_cols:
        baselines[f'{col}_mean'] = subset[col].mean()
        baselines[f'{col}_std'] = subset[col].std()
    return baselines


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def run_evaluation():
    config = CONFIG
    print("=" * 70)
    print("PHASE 2: COHORT vs INDIVIDUAL EVALUATION")
    print("=" * 70)

    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config['COHORT_META']) as f:
        cohort_meta = json.load(f)

    csv_files = sorted(glob(os.path.join(config['DATA_DIR'], '*.csv')))
    csv_map = {}
    for csv_path in csv_files:
        num = os.path.basename(csv_path).replace('u_wesad_', '').replace('.csv', '')
        csv_map[f"S{num}"] = csv_path

    seq_len = config['SEQ_LEN']
    results = []

    # Build reverse lookup: subject -> cohort
    subject_cohort = {}
    for label, info in cohort_meta['cohorts'].items():
        for member in info['members']:
            subject_cohort[member] = (label, info['name'])

    # Evaluate each subject
    for fold_idx, subject in enumerate(SUBJECT_IDS):
        if subject not in csv_map:
            continue

        cohort_label, cohort_name = subject_cohort.get(subject, ('?', 'Unknown'))
        print(f"\n--- {subject} (Cohort: {cohort_name}) ---")

        _, y_test, t_test = prepare_data([csv_map[subject]], seq_len)
        if len(y_test) == 0:
            continue

        # 1. Individual UDE MSE
        indiv_path = os.path.join(config['INDIVIDUAL_MODELS'], f'mcude_fold_{fold_idx+1}.pth')
        if os.path.exists(indiv_path):
            model = MCUDE().to(device)
            model.load_state_dict(torch.load(indiv_path, map_location=device))
            model.eval()
            mses = []
            with torch.no_grad():
                X_test, _, _ = prepare_data([csv_map[subject]], seq_len)
                for si in range(len(X_test)):
                    feat = torch.FloatTensor(X_test[si:si+1]).to(device)
                    t = torch.FloatTensor(t_test[si]).to(device)
                    y0 = torch.tensor([y_test[si][0]]).to(device)
                    model.set_current_batch(t, feat)
                    yp = odeint(model, y0, t, method='euler').squeeze().cpu().numpy()
                    mses.append(mean_squared_error(y_test[si], yp))
            indiv_mse = np.mean(mses)
        else:
            indiv_mse = float('nan')

        # 2. Cohort UDE MSE (using full cohort model)
        cohort_path = os.path.join(config['COHORT_MODELS'], f'cohort{cohort_label}_full.pth')
        if os.path.exists(cohort_path):
            model = MCUDE().to(device)
            model.load_state_dict(torch.load(cohort_path, map_location=device))
            model.eval()
            mses = []
            with torch.no_grad():
                X_test, _, _ = prepare_data([csv_map[subject]], seq_len)
                for si in range(len(X_test)):
                    feat = torch.FloatTensor(X_test[si:si+1]).to(device)
                    t = torch.FloatTensor(t_test[si]).to(device)
                    y0 = torch.tensor([y_test[si][0]]).to(device)
                    model.set_current_batch(t, feat)
                    yp = odeint(model, y0, t, method='euler').squeeze().cpu().numpy()
                    mses.append(mean_squared_error(y_test[si], yp))
            cohort_mse = np.mean(mses)
        else:
            cohort_mse = float('nan')

        # 3. Naive baseline
        naive_mse = np.mean([mean_squared_error(y_test[si], np.full(seq_len, y_test[si][0]))
                            for si in range(len(y_test))])

        # Degradation ratio
        ratio = cohort_mse / indiv_mse if indiv_mse > 0 else float('nan')

        print(f"  Individual MSE: {indiv_mse:.6f}")
        print(f"  Cohort MSE:     {cohort_mse:.6f}")
        print(f"  Naive MSE:      {naive_mse:.6f}")
        print(f"  Degradation:    {ratio:.2f}x")

        results.append({
            'Subject': subject, 'Cohort': cohort_name,
            'Individual_MSE': indiv_mse, 'Cohort_MSE': cohort_mse,
            'Naive_MSE': naive_mse, 'Degradation_Ratio': ratio
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config['OUTPUT_DIR'], 'cohort_vs_individual.csv'), index=False)

    # ======================================================================
    # Summary
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(df[['Subject', 'Cohort', 'Individual_MSE', 'Cohort_MSE', 'Degradation_Ratio']].to_string(index=False))
    print(f"\nMean Individual MSE: {df['Individual_MSE'].mean():.6f}")
    print(f"Mean Cohort MSE:     {df['Cohort_MSE'].mean():.6f}")
    print(f"Mean Degradation:    {df['Degradation_Ratio'].mean():.2f}x")
    print(f"Target: < 1.5x       {'✅ PASS' if df['Degradation_Ratio'].mean() < 1.5 else '⚠️ ABOVE TARGET'}")

    # ======================================================================
    # Cold-Start Classifier
    # ======================================================================
    print(f"\n{'=' * 70}")
    print("COLD-START COHORT ASSIGNMENT CLASSIFIER")
    print(f"{'=' * 70}")

    # Extract baselines for each subject
    baseline_features = []
    baseline_labels = []
    for subject in SUBJECT_IDS:
        if subject in csv_map and subject in subject_cohort:
            baselines = extract_baselines(csv_map[subject])
            baseline_features.append(baselines)
            baseline_labels.append(int(subject_cohort[subject][0]))

    if len(set(baseline_labels)) > 1:
        X_base = pd.DataFrame(baseline_features).values
        y_base = np.array(baseline_labels)

        # Leave-one-out cross-validation
        loo = LeaveOneOut()
        preds = []
        for train_idx, test_idx in loo.split(X_base):
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_base[train_idx], y_base[train_idx])
            preds.append(clf.predict(X_base[test_idx])[0])

        accuracy = accuracy_score(y_base, preds)
        print(f"LOO Accuracy: {accuracy:.1%}")
        print(f"Target: > 80%  {'✅ PASS' if accuracy > 0.8 else '⚠️ BELOW TARGET'}")

        # Feature importance
        clf_full = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_full.fit(X_base, y_base)
        feat_names = list(pd.DataFrame(baseline_features).columns)
        importances = sorted(zip(feat_names, clf_full.feature_importances_),
                           key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 Cold-Start Features:")
        for name, imp in importances[:5]:
            print(f"  {name}: {imp:.4f}")
    else:
        print("  Only 1 cohort — classifier not applicable")

    # ======================================================================
    # Comparison Plot
    # ======================================================================
    if not df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df))
        w = 0.3
        ax.bar(x - w, df['Individual_MSE'], w, label='Individual UDE', color='#2ecc71', alpha=0.85)
        ax.bar(x, df['Cohort_MSE'], w, label='Cohort UDE', color='#3498db', alpha=0.85)
        ax.bar(x + w, df['Naive_MSE'], w, label='Naive', color='#95a5a6', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df['Subject'], rotation=45)
        ax.set_ylabel('MSE')
        ax.set_title('Individual vs Cohort vs Naive MSE')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config['OUTPUT_DIR'], 'cohort_comparison.png'), dpi=200)
        print(f"\n✅ All results saved to {config['OUTPUT_DIR']}")


if __name__ == "__main__":
    run_evaluation()
