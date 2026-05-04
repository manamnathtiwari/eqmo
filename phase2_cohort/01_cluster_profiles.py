"""
Phase 2 Step 1: Cluster Phase 1 α Profiles
============================================
Loads trained per-subject profiles from Phase 1, extracts the 18-dimensional
α vectors, and clusters subjects into cohort groups.

Runs locally (no GPU needed). Uses Phase 1 results only.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PHASE1_RESULTS = os.path.join(PROJECT_ROOT, "phase1_individual", "mc_ude_results")
OUTPUT_DIR = os.path.join(BASE_DIR, "cluster_results")

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


def load_phase1_profiles():
    """Load all profile JSONs from Phase 1 results."""
    profiles = []
    for fold in range(1, 16):
        path = os.path.join(PHASE1_RESULTS, f"profile_fold_{fold}.json")
        if os.path.exists(path):
            with open(path) as f:
                profile = json.load(f)
            profile['fold'] = fold
            profile['subject'] = SUBJECT_IDS[fold - 1] if fold <= len(SUBJECT_IDS) else f"S{fold}"
            profiles.append(profile)
        else:
            print(f"  WARNING: {path} not found")
    return profiles


def extract_alpha_matrix(profiles):
    """Extract α vectors from profiles into a matrix."""
    alpha_matrix = []
    subjects = []
    betas = []
    for p in profiles:
        alpha_matrix.append(p['all_alphas'])
        subjects.append(p['subject'])
        betas.append(p['beta'])
    return np.array(alpha_matrix), subjects, np.array(betas)


def find_optimal_k(alpha_matrix, max_k=6):
    """Test K=2..max_k and find best K using silhouette score."""
    scores = {}
    for k in range(2, min(max_k + 1, len(alpha_matrix))):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(alpha_matrix)
        score = silhouette_score(alpha_matrix, labels)
        scores[k] = score
        print(f"  K={k}: silhouette={score:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"\n  Best K = {best_k} (silhouette = {scores[best_k]:.4f})")
    return best_k, scores


def assign_cohort_names(labels, alpha_matrix, feature_names):
    """Name each cohort based on its dominant features."""
    cohort_names = {}
    for label in sorted(set(labels)):
        mask = labels == label
        mean_alpha = alpha_matrix[mask].mean(axis=0)
        top_idx = np.argmax(mean_alpha)
        top_feature = feature_names[top_idx]

        if 'Heart' in top_feature or 'HRV' in top_feature:
            name = "Cardiac Responder"
        elif 'EDA' in top_feature:
            name = "Electrodermal Responder"
        elif 'Resp' in top_feature:
            name = "Respiratory Responder"
        elif 'Workload' in top_feature:
            name = "Cognitive Responder"
        elif 'Temp' in top_feature:
            name = "Temperature Responder"
        elif 'EMG' in top_feature:
            name = "Muscular Responder"
        elif 'Activity' in top_feature:
            name = "Activity Responder"
        else:
            name = f"Cluster {label}"

        # If two cohorts get same name, differentiate
        if name in cohort_names.values():
            name = f"{name} (Group {label + 1})"

        cohort_names[label] = name
    return cohort_names


def run_clustering():
    """Main clustering pipeline."""
    print("=" * 70)
    print("PHASE 2 STEP 1: CLUSTER PHASE 1 α PROFILES")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load profiles
    profiles = load_phase1_profiles()
    print(f"\nLoaded {len(profiles)} profiles from Phase 1\n")

    if len(profiles) < 3:
        print("ERROR: Need at least 3 profiles for clustering")
        return

    alpha_matrix, subjects, betas = extract_alpha_matrix(profiles)
    print(f"Alpha matrix shape: {alpha_matrix.shape}")
    print(f"Beta range: {betas.min():.4f} - {betas.max():.4f}\n")

    # ======================================================================
    # 1. Silhouette analysis
    # ======================================================================
    print("--- Silhouette Analysis ---")
    best_k, scores = find_optimal_k(alpha_matrix)

    # ======================================================================
    # 2. Run clustering with best K
    # ======================================================================
    print(f"\n--- Clustering with K={best_k} ---")
    clustering = AgglomerativeClustering(n_clusters=best_k)
    labels = clustering.fit_predict(alpha_matrix)
    cohort_names = assign_cohort_names(labels, alpha_matrix, FEATURE_DISPLAY_NAMES)

    # ======================================================================
    # 3. Print cluster assignments
    # ======================================================================
    print(f"\nCluster Assignments:")
    assignments = []
    for i, (subj, label) in enumerate(zip(subjects, labels)):
        cohort = cohort_names[label]
        top_feat = FEATURE_DISPLAY_NAMES[np.argmax(alpha_matrix[i])]
        print(f"  {subj}: {cohort} (top feature: {top_feat}, β={betas[i]:.4f})")
        assignments.append({
            'Subject': subj,
            'Fold': i + 1,
            'Cohort_Label': int(label),
            'Cohort_Name': cohort,
            'Top_Feature': top_feat,
            'Beta': float(betas[i]),
        })

    # Cohort summaries
    print(f"\nCohort Sizes:")
    for label, name in cohort_names.items():
        count = sum(1 for l in labels if l == label)
        mean_beta = betas[labels == label].mean()
        print(f"  {name}: {count} subjects (mean β={mean_beta:.4f})")

    # ======================================================================
    # 4. Figures
    # ======================================================================

    # Fig 1: Dendrogram
    fig, ax = plt.subplots(figsize=(12, 6))
    Z = linkage(alpha_matrix, method='ward')
    dendrogram(Z, labels=subjects, ax=ax, leaf_rotation=45)
    ax.set_title("Hierarchical Clustering of Stress Response Profiles", fontweight='bold')
    ax.set_ylabel("Ward Distance")
    ax.axhline(y=Z[-(best_k - 1), 2], color='red', linestyle='--', alpha=0.7, label=f'Cut for K={best_k}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_dendrogram.png'), dpi=200)
    print(f"\n✅ fig1_dendrogram.png saved")

    # Fig 2: Cluster heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    # Sort by cluster
    order = np.argsort(labels)
    sorted_matrix = alpha_matrix[order]
    sorted_subjects = [subjects[i] for i in order]
    sorted_labels = labels[order]

    # Add cluster separator lines
    sns.heatmap(
        sorted_matrix,
        xticklabels=FEATURE_DISPLAY_NAMES,
        yticklabels=[f"{s} ({cohort_names[l]})" for s, l in zip(sorted_subjects, sorted_labels)],
        cmap='YlOrRd', annot=True, fmt='.3f',
        linewidths=0.5, ax=ax,
        cbar_kws={'label': 'Feature Sensitivity (α)'}
    )
    ax.set_title('Per-Subject α Profiles Grouped by Cohort', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_cohort_heatmap.png'), dpi=200)
    print("✅ fig2_cohort_heatmap.png saved")

    # Fig 3: Silhouette plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = sorted(scores.keys())
    ax.plot(ks, [scores[k] for k in ks], 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Optimal K Selection', fontweight='bold')
    ax.set_xticks(ks)
    ax.axvline(best_k, color='red', linestyle='--', alpha=0.5, label=f'Best K={best_k}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_silhouette.png'), dpi=200)
    print("✅ fig3_silhouette.png saved")

    # Fig 4: Cohort mean profiles comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(FEATURE_DISPLAY_NAMES))
    width = 0.8 / best_k
    colors = plt.cm.Set2(np.linspace(0, 1, best_k))

    for label, name in cohort_names.items():
        mask = labels == label
        mean_alpha = alpha_matrix[mask].mean(axis=0)
        offset = (label - (best_k - 1) / 2) * width
        ax.bar(x + offset, mean_alpha, width, label=name, color=colors[label], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_DISPLAY_NAMES, rotation=45, ha='right')
    ax.set_ylabel('Mean α')
    ax.set_title('Cohort Mean Profiles Comparison', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_cohort_comparison.png'), dpi=200)
    print("✅ fig4_cohort_comparison.png saved")

    # ======================================================================
    # 5. Save results
    # ======================================================================
    assignments_df = pd.DataFrame(assignments)
    assignments_df.to_csv(os.path.join(OUTPUT_DIR, 'cohort_assignments.csv'), index=False)

    # Save cohort metadata for downstream scripts
    cohort_meta = {
        'best_k': best_k,
        'silhouette_scores': {str(k): v for k, v in scores.items()},
        'cohorts': {}
    }
    for label, name in cohort_names.items():
        mask = labels == label
        members = [subjects[i] for i in range(len(subjects)) if labels[i] == label]
        folds = [i + 1 for i in range(len(subjects)) if labels[i] == label]
        cohort_meta['cohorts'][str(label)] = {
            'name': name,
            'members': members,
            'folds': folds,
            'mean_alpha': alpha_matrix[mask].mean(axis=0).tolist(),
            'mean_beta': float(betas[mask].mean()),
            'size': int(mask.sum()),
        }
    with open(os.path.join(OUTPUT_DIR, 'cohort_metadata.json'), 'w') as f:
        json.dump(cohort_meta, f, indent=2)

    print(f"\n✅ All results saved to {OUTPUT_DIR}")
    print(f"\nCohort metadata saved — ready for 02_train_cohort.py")

    return assignments_df, cohort_meta


if __name__ == "__main__":
    run_clustering()
