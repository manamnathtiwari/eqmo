"""
Explainability Analysis Script - MULTI-COEFFICIENT VERSION
---------------------------------------------------------
Extracts and visualizes learned features sensitivities (α₁...α₁₈) and β
from trained multi-coefficient UDE models.

Each subject gets:
- 18 feature-specific sensitivities (which signals drive THEIR stress)
- 1 recovery rate (how fast they bounce back)

This enables personalized stress phenotyping.
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_model import UDE
from src.utils import FEATURE_COLUMNS

def analyze_explainability():
    print("="*70)
    print("MULTI-COEFFICIENT EXPLAINABILITY: 19-PARAMETER PROFILES")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, 'results', 'loso_models')
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    
    # Get all subject files
    all_files = sorted([f for f in os.listdir(data_dir) if f.startswith('u_wesad_') and f.endswith('.csv')])
    
    # Find completed models
    model_files = [f for f in os.listdir(models_dir) if f.startswith('ude_fold_') and f.endswith('.pth')]
    model_files.sort()
    
    results = []
    alpha_matrix = []  # For heatmap
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Found {len(model_files)} trained models.\n")
    
    for f in model_files:
        # Parse Fold ID
        fold_idx = int(f.split('_')[-1].split('.')[0]) - 1
        subject_file = all_files[fold_idx]
        subject_id = os.path.basename(subject_file).replace('.csv', '')
        
        # Load Model
        model = UDE().to(device)
        model.load_state_dict(torch.load(os.path.join(models_dir, f), map_location=device))
        
        # Extract ALL parameters
        params = model.get_interpretable_params()
        alphas = params['alphas']  # Array of 18
        beta = params['beta']
        
        # Store for heatmap
        alpha_matrix.append(alphas)
        
        # Store summary
        results.append({
            'Subject': subject_id,
            'Beta (Recovery)': beta,
            'Alpha_Mean': alphas.mean(),
            'Alpha_Std': alphas.std(),
            'Dominant_Feature': FEATURE_COLUMNS[alphas.argmax()],
            'Dominant_Alpha': alphas.max(),
            'Risk_Score': params['risk_score']
        })
    
    df_summary = pd.DataFrame(results)
    alpha_matrix = np.array(alpha_matrix)
    
    # Display Results
    print("\nSUMMARY: 19-Parameter Profiles")
    print("-" * 70)
    print(df_summary.to_string(index=False))
    print("-" * 70)
    
    # Interpretation Examples
    print("\nEXAMPLE INTERPRETATIONS:")
    for i in range(min(3, len(df_summary))):
        subj = results[i]
        print(f"\n{subj['Subject']}:")
        print(f"  - Recovery Rate: {subj['Beta (Recovery)']:.3f}")
        print(f"  - Dominant Driver: {subj['Dominant_Feature']} (α={subj['Dominant_Alpha']:.3f})")
        print(f"  - Risk Score: {subj['Risk_Score']:.2f}")
    
    # === VISUALIZATION 1: Heatmap ===
    print("\nGenerating heatmap...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(alpha_matrix, 
                xticklabels=FEATURE_COLUMNS,
                yticklabels=[r['Subject'] for r in results],
                cmap='YlOrRd',
                annot=False,
                cbar_kws={'label': 'Sensitivity (α)'},
                ax=ax)
    
    plt.title('Feature Sensitivity Profiles (19-Parameter Model)', fontsize=14, fontweight='bold')
    plt.xlabel('Physiological Features')
    plt.ylabel('Subjects')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    out_dir = os.path.join(base_dir, 'results', 'explainability')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'sensitivity_heatmap.png'), dpi=150)
    print(f"Saved: {os.path.join(out_dir, 'sensitivity_heatmap.png')}")
    
    # === VISUALIZATION 2: Top Features per Person ===
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(min(15, len(alpha_matrix))):
        ax = axes[i]
        sorted_idx = np.argsort(alpha_matrix[i])[::-1][:5]  # Top 5
        top_features = [FEATURE_COLUMNS[j] for j in sorted_idx]
        top_values = alpha_matrix[i][sorted_idx]
        
        ax.barh(top_features, top_values, color='steelblue')
        ax.set_title(results[i]['Subject'], fontsize=10)
        ax.set_xlabel('α')
        ax.invert_yaxis()
    
    plt.suptitle('Top 5 Stress Drivers per Subject', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'top_features.png'), dpi=150)
    print(f"Saved: {os.path.join(out_dir, 'top_features.png')}")
    
    # Save detailed parameters
    df_detailed = pd.DataFrame(alpha_matrix, columns=FEATURE_COLUMNS)
    df_detailed.insert(0, 'Subject', [r['Subject'] for r in results])
    df_detailed.insert(1, 'Beta', [r['Beta (Recovery)'] for r in results])
    df_detailed.to_csv(os.path.join(out_dir, 'parameters_detailed.csv'), index=False)
    print(f"Saved: {os.path.join(out_dir, 'parameters_detailed.csv')}")
    
    # Save summary
    df_summary.to_csv(os.path.join(out_dir, 'parameters_summary.csv'), index=False)
    print(f"Saved: {os.path.join(out_dir, 'parameters_summary.csv')}")
    
    print("\n✅ Multi-coefficient explainability analysis complete!")

if __name__ == "__main__":
    analyze_explainability()
