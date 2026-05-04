"""
Phase 2 Step 4: Cohort Analysis & What-If Simulation
=====================================================
Local script. Runs after downloading cohort training/eval results from Kaggle.
Generates publication figures and cohort-level what-if analysis.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTER_RESULTS = os.path.join(BASE_DIR, "cluster_results")
COHORT_RESULTS = os.path.join(BASE_DIR, "cohort_results", "cohort_ude_results")  # Downloaded from Kaggle
PHASE1_RESULTS = os.path.join(os.path.dirname(BASE_DIR), "phase1_individual", "mc_ude_results")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis")

FEATURE_DISPLAY_NAMES = [
    'Workload', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_pNN50', 'HRV_LF/HF',
    'Heart Rate', 'EDA_Mean', 'EDA_Std', 'EDA_Peaks',
    'Resp_Mean', 'Resp_Std', 'Resp_Rate',
    'Temp_Mean', 'Temp_Std', 'Activity_Mean', 'Activity_Std',
    'EMG_Mean', 'EMG_Std'
]


def run_analysis():
    print("=" * 70)
    print("PHASE 2 STEP 4: COHORT ANALYSIS & VISUALIZATION")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load cohort metadata
    meta_path = os.path.join(CLUSTER_RESULTS, "cohort_metadata.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found. Run 01_cluster_profiles.py first.")
        return

    with open(meta_path) as f:
        cohort_meta = json.load(f)

    print(f"\nFound {len(cohort_meta['cohorts'])} cohorts\n")

    # ======================================================================
    # 1. Compare Cohort vs Individual α profiles
    # ======================================================================
    print("--- Comparing Cohort vs Individual Profiles ---")

    # Load individual profiles
    indiv_profiles = {}
    for fold in range(1, 16):
        path = os.path.join(PHASE1_RESULTS, f"profile_fold_{fold}.json")
        if os.path.exists(path):
            with open(path) as f:
                indiv_profiles[fold] = json.load(f)

    # Load cohort profiles (if available from Kaggle results)
    cohort_profiles = {}
    for label in cohort_meta['cohorts']:
        path = os.path.join(COHORT_RESULTS, f"cohort{label}_full_profile.json")
        if os.path.exists(path):
            with open(path) as f:
                cohort_profiles[label] = json.load(f)
            print(f"  Loaded cohort {label} profile")
        else:
            # Use mean from clustering as placeholder
            cohort_profiles[label] = {
                'all_alphas': cohort_meta['cohorts'][label]['mean_alpha'],
                'beta': cohort_meta['cohorts'][label]['mean_beta'],
            }
            print(f"  Using clustered mean for cohort {label} (no trained model yet)")

    # ======================================================================
    # Fig 1: Cohort Equations Side by Side
    # ======================================================================
    fig, axes = plt.subplots(1, len(cohort_profiles), figsize=(7 * len(cohort_profiles), 6))
    if len(cohort_profiles) == 1:
        axes = [axes]

    for ax, (label, profile) in zip(axes, cohort_profiles.items()):
        alphas = np.array(profile['all_alphas'])
        sorted_idx = np.argsort(alphas)[::-1]

        colors = ['#e74c3c' if a > np.mean(alphas) else '#3498db' for a in alphas[sorted_idx]]
        ax.barh(
            [FEATURE_DISPLAY_NAMES[i] for i in sorted_idx],
            alphas[sorted_idx],
            color=colors, alpha=0.85
        )
        name = cohort_meta['cohorts'][label]['name']
        n = cohort_meta['cohorts'][label]['size']
        ax.set_title(f'{name}\n(n={n}, β={profile["beta"]:.4f})', fontweight='bold')
        ax.set_xlabel('Sensitivity (α)')
        ax.invert_yaxis()

    plt.suptitle('Cohort-Level α Profiles', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_cohort_equations.png'), dpi=200, bbox_inches='tight')
    print("\n✅ fig1_cohort_equations.png saved")

    # ======================================================================
    # Fig 2: Individual vs Cohort α deviation
    # ======================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    deviations = []
    for label, info in cohort_meta['cohorts'].items():
        cohort_alpha = np.array(cohort_profiles[label]['all_alphas'])
        for fold in info['folds']:
            if fold in indiv_profiles:
                indiv_alpha = np.array(indiv_profiles[fold]['all_alphas'])
                dev = np.sqrt(np.mean((indiv_alpha - cohort_alpha) ** 2))
                deviations.append({
                    'Subject': f"S{fold:03d}",
                    'Cohort': info['name'],
                    'RMSE_Deviation': dev,
                })

    if deviations:
        dev_df = pd.DataFrame(deviations)
        colors_map = {name: color for name, color in
                      zip(dev_df['Cohort'].unique(), plt.cm.Set2.colors)}
        bars = ax.bar(
            dev_df['Subject'], dev_df['RMSE_Deviation'],
            color=[colors_map[c] for c in dev_df['Cohort']], alpha=0.85
        )
        ax.set_ylabel('α RMSE (Individual vs Cohort)')
        ax.set_title('How Much Each Subject Deviates from Their Cohort', fontweight='bold')
        ax.axhline(dev_df['RMSE_Deviation'].mean(), color='red', linestyle='--',
                    alpha=0.5, label=f"Mean = {dev_df['RMSE_Deviation'].mean():.4f}")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_cohort_deviation.png'), dpi=200)
        print("✅ fig2_cohort_deviation.png saved")

    # ======================================================================
    # Fig 3: Cohort What-If Analysis (simulation using α profiles)
    # ======================================================================
    print("\n--- Cohort What-If Analysis ---")

    # Simulate what happens if we scale each feature by 0.5x and 2.0x
    # using the linear part of the equation only (α contribution)
    fig, axes = plt.subplots(1, len(cohort_profiles), figsize=(7 * len(cohort_profiles), 8))
    if len(cohort_profiles) == 1:
        axes = [axes]

    for ax, (label, profile) in zip(axes, cohort_profiles.items()):
        alphas = np.array(profile['all_alphas'])
        name = cohort_meta['cohorts'][label]['name']

        # Impact = alpha * (scale_factor - 1.0)
        # If we double a feature (scale=2.0), the extra contribution is alpha * 1.0
        # If we halve (scale=0.5), the reduction is alpha * (-0.5)
        impact_double = alphas * 1.0   # Extra stress from doubling
        impact_halve = alphas * (-0.5)  # Stress reduction from halving

        sorted_idx = np.argsort(np.abs(impact_double))[::-1]
        features = [FEATURE_DISPLAY_NAMES[i] for i in sorted_idx[:10]]
        impact_d = impact_double[sorted_idx[:10]]
        impact_h = impact_halve[sorted_idx[:10]]

        y = np.arange(len(features))
        ax.barh(y - 0.15, impact_d, 0.3, label='Double (2x)', color='#e74c3c', alpha=0.8)
        ax.barh(y + 0.15, impact_h, 0.3, label='Halve (0.5x)', color='#2ecc71', alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(features)
        ax.set_xlabel('Stress Impact (Δ contribution)')
        ax.set_title(f'{name}\nIntervention Impact', fontweight='bold')
        ax.legend()
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    plt.suptitle('Cohort-Level What-If: Feature Intervention Impact', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_cohort_whatif.png'), dpi=200, bbox_inches='tight')
    print("✅ fig3_cohort_whatif.png saved")

    # ======================================================================
    # Load evaluation results if available
    # ======================================================================
    # Check multiple possible locations for eval results
    eval_candidates = [
        os.path.join(COHORT_RESULTS, 'cohort_vs_individual.csv'),
        os.path.join(BASE_DIR, 'cohort_results', 'cohort_evaluation', 'cohort_vs_individual.csv'),
        os.path.join(BASE_DIR, 'cohort_results', 'cohort_vs_individual.csv'),
    ]
    eval_path = next((p for p in eval_candidates if os.path.exists(p)), None)
    if eval_path:
        eval_df = pd.read_csv(eval_path)
        print(f"\n--- Evaluation Results ---")
        print(eval_df.to_string(index=False))
        print(f"\nMean Degradation Ratio: {eval_df['Degradation_Ratio'].mean():.2f}x")
    else:
        print("\n⚠️ No evaluation results yet (run 03_evaluate_cohort.py on Kaggle first)")

    # ======================================================================
    # Summary Report
    # ======================================================================
    report = {
        'num_cohorts': len(cohort_meta['cohorts']),
        'cohorts': {},
    }
    for label, info in cohort_meta['cohorts'].items():
        alpha = np.array(cohort_profiles[label]['all_alphas'])
        top3 = sorted(zip(FEATURE_DISPLAY_NAMES, alpha), key=lambda x: x[1], reverse=True)[:3]
        report['cohorts'][info['name']] = {
            'size': info['size'],
            'beta': cohort_profiles[label]['beta'],
            'top_features': {f: round(float(a), 4) for f, a in top3},
        }

    with open(os.path.join(OUTPUT_DIR, 'cohort_summary.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ All analysis saved to {OUTPUT_DIR}")
    return report


if __name__ == "__main__":
    run_analysis()
