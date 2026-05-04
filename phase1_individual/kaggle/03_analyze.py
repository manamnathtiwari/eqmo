"""
KAGGLE NOTEBOOK: Explainability & Sparse Profile Analysis
==========================================================
Runs after training is complete (01_train.py).
Extracts per-subject sparse physiological profiles, generates
heatmaps, and creates publication-ready visualizations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# ============================================================================
# CONFIG
# ============================================================================
MODELS_DIR = '/kaggle/input/mc-ude-models/mc_ude_results'
OUTPUT_DIR = '/kaggle/working/analysis'
NUM_FEATURES = 18

FEATURE_DISPLAY_NAMES = [
    'Workload', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF/HF',
    'Heart Rate', 'EDA_Mean', 'EDA_Std', 'EDA_Peaks',
    'Resp_Mean', 'Resp_Std', 'Resp_Rate',
    'Temp_Mean', 'Temp_Std', 'Activity_Mean', 'Activity_Std',
    'EMG_Mean', 'EMG_Std'
]

# ============================================================================
# MC-UDE Model (same as training)
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
        self.num_features = num_features

    @property
    def beta(self): return F.softplus(self._beta_raw)
    @property
    def alphas(self): return F.softplus(self._alphas_raw)

    def get_equation_string(self, threshold=0.01):
        beta = self.beta.item()
        alphas = self.alphas.detach().cpu().numpy()
        terms = [f"-{beta:.4f}·S"]
        for i, (a, n) in enumerate(zip(alphas, FEATURE_DISPLAY_NAMES)):
            if a > threshold: terms.append(f"+{a:.4f}·{n}")
        terms.append("+ NN(S,F)")
        return "dS/dt = " + " ".join(terms)


# ============================================================================
# Analysis
# ============================================================================
def run_analysis():
    print("=" * 70)
    print("MC-UDE EXPLAINABILITY & SPARSE PROFILE ANALYSIS")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all trained models
    model_files = sorted(glob(os.path.join(MODELS_DIR, 'mcude_fold_*.pth')))
    print(f"Found {len(model_files)} trained models\n")

    if len(model_files) == 0:
        print("ERROR: No trained models found. Run 01_train.py first.")
        return

    subjects = []
    alpha_matrix = []
    betas = []
    equations = []
    profiles = []

    for f in model_files:
        fold = int(f.split('fold_')[1].split('.')[0])
        model = MCUDE(num_features=NUM_FEATURES)
        model.load_state_dict(torch.load(f, map_location='cpu'))
        model.eval()

        alphas = model.alphas.detach().cpu().numpy()
        beta = model.beta.item()
        eq = model.get_equation_string()
        n_active = sum(1 for a in alphas if a > 0.01)

        subjects.append(f"S{fold}")
        alpha_matrix.append(alphas)
        betas.append(beta)
        equations.append(eq)

        # Classify stress profile
        top_features = sorted(zip(FEATURE_DISPLAY_NAMES, alphas), key=lambda x: x[1], reverse=True)
        if top_features[0][0].startswith('HRV') or top_features[0][0] == 'Heart Rate':
            profile_type = "Cardiac Responder"
        elif top_features[0][0].startswith('EDA'):
            profile_type = "Anxiety Responder"
        elif top_features[0][0].startswith('Resp'):
            profile_type = "Respiratory Responder"
        elif top_features[0][0] == 'Workload':
            profile_type = "Cognitive Load Responder"
        else:
            profile_type = "Mixed Responder"

        profiles.append({
            'Subject': f"S{fold}",
            'Profile': profile_type,
            'Beta': beta,
            'N_Active': n_active,
            'Top_Feature': top_features[0][0],
            'Top_Alpha': float(top_features[0][1]),
            'Equation': eq,
        })

        print(f"  S{fold}: β={beta:.4f}, Active={n_active}/18, Type={profile_type}")
        print(f"    Top: {top_features[0][0]}={top_features[0][1]:.4f}, "
              f"{top_features[1][0]}={top_features[1][1]:.4f}")

    alpha_matrix = np.array(alpha_matrix)

    # ======================================================================
    # Figure 1: Alpha Heatmap (Publication Ready)
    # ======================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        alpha_matrix,
        xticklabels=FEATURE_DISPLAY_NAMES,
        yticklabels=subjects,
        cmap='YlOrRd',
        annot=True, fmt='.3f',
        linewidths=0.5,
        cbar_kws={'label': 'Feature Sensitivity (α)'},
        ax=ax
    )
    ax.set_title('Per-Subject Feature Sensitivities (L1-Sparse MC-UDE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Physiological Features')
    ax.set_ylabel('Subjects')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_alpha_heatmap.png'), dpi=300)
    print(f"\n✅ Figure 1 saved: fig1_alpha_heatmap.png")

    # ======================================================================
    # Figure 2: Recovery Rate Comparison
    # ======================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Beta bar chart
    colors = ['#e74c3c' if b < np.median(betas) else '#2ecc71' for b in betas]
    axes[0].barh(subjects, betas, color=colors, alpha=0.85)
    axes[0].set_xlabel('Recovery Rate (β)')
    axes[0].set_title('Per-Subject Recovery Rates', fontweight='bold')
    axes[0].axvline(np.median(betas), color='black', linestyle='--', alpha=0.5, label='Median')
    axes[0].legend()

    # Feature importance (mean across subjects)
    mean_alphas = alpha_matrix.mean(axis=0)
    sorted_idx = np.argsort(mean_alphas)[::-1]
    axes[1].barh(
        [FEATURE_DISPLAY_NAMES[i] for i in sorted_idx[:10]],
        mean_alphas[sorted_idx[:10]],
        color='steelblue', alpha=0.85
    )
    axes[1].set_xlabel('Mean Sensitivity (α)')
    axes[1].set_title('Top 10 Stress Predictors (Population Level)', fontweight='bold')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_recovery_and_importance.png'), dpi=300)
    print("✅ Figure 2 saved: fig2_recovery_and_importance.png")

    # ======================================================================
    # Figure 3: Sparsity Pattern
    # ======================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    sparsity = (alpha_matrix < 0.01).astype(float)  # 1 = zero, 0 = active
    sns.heatmap(
        sparsity, xticklabels=FEATURE_DISPLAY_NAMES, yticklabels=subjects,
        cmap='RdYlGn_r', cbar_kws={'label': 'Feature Active (0) / Pruned (1)'},
        ax=ax, linewidths=0.5
    )
    ax.set_title('L1-Sparse Feature Selection Map', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_sparsity_map.png'), dpi=300)
    print("✅ Figure 3 saved: fig3_sparsity_map.png")

    # ======================================================================
    # Save all data
    # ======================================================================
    pd.DataFrame(profiles).to_csv(os.path.join(OUTPUT_DIR, 'subject_profiles.csv'), index=False)

    alpha_df = pd.DataFrame(alpha_matrix, columns=FEATURE_DISPLAY_NAMES, index=subjects)
    alpha_df['Beta'] = betas
    alpha_df.to_csv(os.path.join(OUTPUT_DIR, 'alpha_matrix.csv'))

    # Save equations
    with open(os.path.join(OUTPUT_DIR, 'learned_equations.txt'), 'w') as f:
        for subj, eq, prof in zip(subjects, equations, profiles):
            f.write(f"{subj} [{prof['Profile']}]:\n  {eq}\n\n")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")

    # Summary statistics
    df = pd.DataFrame(profiles)
    print(f"\nProfile Distribution:")
    print(df['Profile'].value_counts().to_string())
    print(f"\nMean β: {np.mean(betas):.4f} ± {np.std(betas):.4f}")
    print(f"Mean Active Features: {df['N_Active'].mean():.1f}/18")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

    return df

if __name__ == "__main__":
    run_analysis()
