"""
Analyze 13 Trained Multi-Coefficient UDE Models
Extracts equations, visualizes alphas, generates paper figures
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Feature names
FEATURE_NAMES = [
    'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50', 'hrv_lf_hf',
    'heart_rate', 'workload',
    'eda_mean', 'eda_std', 'eda_peaks',
    'resp_mean', 'resp_std', 'resp_rate',
    'temp_mean', 'temp_std',
    'activity_level', 'activity_std',
    'emg_mean', 'emg_std'
]

def load_all_results(results_dir='results/multicoeff_models'):
    """Load all 13 models and their alphas"""
    results_dir = Path(results_dir)
    
    all_alphas = []
    all_betas = []
    
    for fold in range(1, 14):  # 13 folds
        # Load alphas
        alphas_path = results_dir / f'alphas_fold_{fold}.csv'
        if alphas_path.exists():
            alphas_df = pd.read_csv(alphas_path)
            alphas_df['Fold'] = fold
            all_alphas.append(alphas_df)
            
            # Load model for beta
            model_path = results_dir / f'multicoeff_ude_fold_{fold}.pth'
            if model_path.exists():
                from src.models.ude_multicoeff import UDEMultiCoeff
                model = UDEMultiCoeff(hidden_dim=64, num_features=18)
                model.load_state_dict(torch.load(model_path))
                params = model.get_learned_params()
                all_betas.append(params['beta'])
    
    alphas_df = pd.concat(all_alphas, ignore_index=True)
    
    return alphas_df, all_betas


def print_summary_statistics(alphas_df, betas):
    """Print summary statistics"""
    print("="*70)
    print("MULTI-COEFFICIENT UDE - 13 MODELS SUMMARY")
    print("="*70)
    
    print(f"\nRecovery Rate (Beta):")
    print(f"  Mean: {np.mean(betas):.6f}")
    print(f"  Std:  {np.std(betas):.6f}")
    print(f"  Range: [{np.min(betas):.6f}, {np.max(betas):.6f}]")
    
    print(f"\nFeature Sensitivities (Alphas):")
    print(f"  Total features: 18")
    print(f"  Models: 13")
    
    # Mean alpha per feature
    mean_alphas = alphas_df.groupby('Feature')['Alpha'].mean().sort_values(ascending=False)
    
    print(f"\nTop 5 Most Important Features:")
    for i, (feature, alpha) in enumerate(mean_alphas.head(5).items(), 1):
        print(f"  {i}. {feature}: {alpha:.6f}")
    
    print(f"\nBottom 5 Least Important Features:")
    for i, (feature, alpha) in enumerate(mean_alphas.tail(5).items(), 1):
        print(f"  {i}. {feature}: {alpha:.6f}")


def plot_alpha_heatmap(alphas_df, output_path='paper/figures/alphas_heatmap.png'):
    """Create heatmap of alphas across folds"""
    # Pivot for heatmap
    pivot = alphas_df.pivot(index='Fold', columns='Feature', values='Alpha')
    
    # Create figure
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Alpha Value'})
    plt.title('Learned Feature Sensitivities Across 13 Subjects', fontsize=14, fontweight='bold')
    plt.xlabel('Physiological Features', fontsize=12)
    plt.ylabel('Subject (Fold)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved heatmap: {output_path}")
    plt.close()


def plot_alpha_distributions(alphas_df, output_path='paper/figures/alphas_boxplot.png'):
    """Create boxplot of alpha distributions"""
    # Calculate mean alpha per feature
    mean_alphas = alphas_df.groupby('Feature')['Alpha'].mean().sort_values(ascending=False)
    
    # Reorder for plotting
    alphas_df['Feature'] = pd.Categorical(alphas_df['Feature'], 
                                          categories=mean_alphas.index, 
                                          ordered=True)
    
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=alphas_df, x='Feature', y='Alpha', color='skyblue')
    plt.title('Distribution of Feature Sensitivities (13 Subjects)', fontsize=14, fontweight='bold')
    plt.xlabel('Physiological Features', fontsize=12)
    plt.ylabel('Alpha Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved boxplot: {output_path}")
    plt.close()


def print_example_equations(alphas_df, betas, num_examples=3):
    """Print example discovered equations"""
    print(f"\n{'='*70}")
    print(f"EXAMPLE DISCOVERED EQUATIONS")
    print(f"{'='*70}")
    
    for fold in range(1, min(num_examples + 1, 14)):
        fold_alphas = alphas_df[alphas_df['Fold'] == fold]
        beta = betas[fold - 1]
        
        print(f"\nSubject {fold}:")
        print(f"dS/dt = -{beta:.4f}·S", end='')
        
        # Show top 5 terms
        top_terms = fold_alphas.nlargest(5, 'Alpha')
        for _, row in top_terms.iterrows():
            print(f" + {row['Alpha']:.4f}·{row['Feature']}", end='')
        
        print(" + ... + NN(S,F)")


def create_results_table(results_path='results/multicoeff_models/loso_results.csv'):
    """Create formatted results table for paper"""
    if not Path(results_path).exists():
        print(f"⚠️  Results file not found: {results_path}")
        return
    
    results = pd.read_csv(results_path)
    
    print(f"\n{'='*70}")
    print("LOSO CROSS-VALIDATION RESULTS (13 FOLDS)")
    print(f"{'='*70}")
    print(results.to_string(index=False))
    
    print(f"\nSummary Statistics:")
    print(f"  Mean Test MSE: {results['Test_MSE'].mean():.6f} ± {results['Test_MSE'].std():.6f}")
    print(f"  Min Test MSE:  {results['Test_MSE'].min():.6f}")
    print(f"  Max Test MSE:  {results['Test_MSE'].max():.6f}")
    
    # Save formatted table
    output_path = 'paper/tables/loso_results.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\n✅ Saved results table: {output_path}")


def main():
    """Run complete analysis"""
    print("Loading 13 trained models...")
    
    # Load data
    alphas_df, betas = load_all_results()
    
    print(f"✅ Loaded {len(betas)} models with {len(alphas_df)} alpha values\n")
    
    # Print statistics
    print_summary_statistics(alphas_df, betas)
    
    # Print example equations
    print_example_equations(alphas_df, betas, num_examples=3)
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print(f"{'='*70}")
    
    plot_alpha_heatmap(alphas_df)
    plot_alpha_distributions(alphas_df)
    
    # Create results table
    create_results_table()
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  - paper/figures/alphas_heatmap.png")
    print("  - paper/figures/alphas_boxplot.png")
    print("  - paper/tables/loso_results.csv")
    print("\nYou can now:")
    print("  1. Use these figures in your paper")
    print("  2. Report Mean MSE in results section")
    print("  3. Show example equations in discussion")
    print("  4. Submit your paper!")


if __name__ == "__main__":
    main()
