"""
PATH C: Comprehensive Comparison
---------------------------------
Compares three approaches:
1. Sparse Linear (Simple baseline)
2. Multi-Coefficient UDE (Complex model)
3. Random Forest (Black-box baseline)

Goal: Show when complexity (UDE) is justified vs when simplicity (Linear) suffices.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compare_all_methods():
    print("="*70)
    print("PATH C: COMPREHENSIVE METHOD COMPARISON")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Load results
    results = {}
    
    # Sparse Linear
    linear_path = os.path.join(base_dir, 'results', 'sparse_linear', 'loso_results.csv')
    if os.path.exists(linear_path):
        df_linear = pd.read_csv(linear_path)
        results['Sparse_Linear'] = df_linear['Test_MSE'].values
        print(f"✅ Loaded Sparse Linear: {len(results['Sparse_Linear'])} folds")
    else:
        print(f"❌ Sparse Linear results not found. Run: python src/models/sparse_linear.py")
    
    # UDE
    ude_path = os.path.join(base_dir, 'results', 'loso_models', 'loso_results.csv')
    if os.path.exists(ude_path):
        df_ude = pd.read_csv(ude_path)
        results['UDE'] = df_ude['test_mse'].values
        print(f"✅ Loaded UDE: {len(results['UDE'])} folds")
    else:
        print(f"⚠️  UDE results not found yet. Will train on Kaggle.")
    
    # RF (from SOTA comparison)
    rf_path = os.path.join(base_dir, 'results', 'sota_comparison', 'forecasting_results.csv')
    if os.path.exists(rf_path):
        df_rf = pd.read_csv(rf_path)
        results['RF_Autoreg'] = df_rf['RF_Autoreg'].values
        print(f"✅ Loaded RF: {len(results['RF_Autoreg'])} folds")
    else:
        print(f"⚠️  RF results not found.")
    
    if len(results) < 2:
        print("\n⚠️  Need at least 2 methods to compare. Run training first.")
        return
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(results)
    comparison['Fold'] = range(1, len(comparison)+1)
    
    print("\n" + "="*70)
    print("RESULTS BY FOLD")
    print("="*70)
    print(comparison.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    for method in results.keys():
        mean = comparison[method].mean()
        std = comparison[method].std()
        print(f"{method:20s}: {mean:.6f} ± {std:.6f}")
    
    # Statistical tests
    if 'Sparse_Linear' in results and 'UDE' in results:
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(results['Sparse_Linear'], results['UDE'])
        print(f"\nPaired t-test (Sparse Linear vs UDE):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f}")
        if p_val < 0.05:
            winner = "Sparse Linear" if t_stat > 0 else "UDE"
            print(f"  Result: {winner} is significantly better (p < 0.05)")
        else:
            print(f"  Result: No significant difference (p >= 0.05)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    comparison_melted = comparison.melt(id_vars=['Fold'], var_name='Method', value_name='MSE')
    sns.boxplot(data=comparison_melted, x='Method', y='MSE', ax=axes[0])
    axes[0].set_title('Performance Distribution', fontweight='bold')
    axes[0].set_ylabel('Mean Squared Error (lower is better)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Fold-by-fold comparison
    for method in results.keys():
        axes[1].plot(comparison['Fold'], comparison[method], marker='o', label=method, alpha=0.7)
    
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Fold-by-Fold Comparison', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    out_dir = os.path.join(base_dir, 'results', 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'method_comparison.png'), dpi=150)
    comparison.to_csv(os.path.join(out_dir, 'comparison_results.csv'), index=False)
    
    print(f"\n✅ Comparison saved to: {out_dir}")
    
    # Interpretability comparison
    print("\n" + "="*70)
    print("INTERPRETABILITY COMPARISON")
    print("="*70)
    
    # Sparse Linear
    if 'Sparse_Linear' in results:
        linear_coef_path = os.path.join(base_dir, 'results', 'sparse_linear', 'coefficients.csv')
        if os.path.exists(linear_coef_path):
            df_linear_coef = pd.read_csv(linear_coef_path)
            n_features = df_linear_coef.shape[1] - 2  # Exclude Subject, Beta
            n_zero = (np.abs(df_linear_coef.iloc[:, 2:]) < 1e-4).sum().sum()
            n_total = n_features * len(df_linear_coef)
            sparsity = 100 * n_zero / n_total
            print(f"Sparse Linear:")
            print(f"  - Parameters per person: {n_features + 1} (alphas + beta)")
            print(f"  - Average sparsity: {sparsity:.1f}%")
            print(f"  - Interpretation: Direct coefficients, zero = irrelevant")
    
    # UDE
    if 'UDE' in results:
        ude_params_path = os.path.join(base_dir, 'results', 'explainability', 'parameters_detailed.csv')
        if os.path.exists(ude_params_path):
            df_ude_params = pd.read_csv(ude_params_path)
            n_features_ude = df_ude_params.shape[1] - 2
            print(f"\nUDE (Multi-Coefficient):")
            print(f"  - Parameters per person: {n_features_ude + 1} (alphas + beta)")
            print(f"  - Sparsity: N/A (all features used)")
            print(f"  - Interpretation: Differential equation coefficients + NN correction")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if 'Sparse_Linear' in results and 'UDE' in results:
        linear_mean = comparison['Sparse_Linear'].mean()
        ude_mean = comparison['UDE'].mean()
        diff_pct = 100 * abs(linear_mean - ude_mean) / linear_mean
        
        if diff_pct < 5:
            print("📌 Sparse Linear and UDE have similar accuracy (<5% difference)")
            print("   RECOMMENDATION: Use Sparse Linear (simpler, faster, equally good)")
        elif linear_mean < ude_mean:
            print("📌 Sparse Linear outperforms UDE")
            print("   RECOMMENDATION: Linear model is sufficient for this problem")
        else:
            print("📌 UDE outperforms Sparse Linear")
            print(f"   Improvement: {diff_pct:.1f}%")
            print("   RECOMMENDATION: Complexity (UDE) is justified")

if __name__ == "__main__":
    compare_all_methods()
