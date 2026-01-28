"""
Sparse Linear Dynamical System (BASELINE)
-----------------------------------------
Simple, interpretable baseline for stress dynamics modeling.

Model: S(t) = β*S(t-1) + Σᵢ αᵢ*Fᵢ(t) + ε

Where:
- β: Memory/momentum term (single parameter)
- αᵢ: Feature sensitivities (18 parameters, SPARSE via L1)
- SPARSE: L1 regularization forces most αᵢ → 0
- Only important features have non-zero αᵢ

This is Path C's BASELINE - we start here, then show when UDE is needed.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_sparse_linear_loso():
    """
    Train Sparse Linear model using LOSO cross-validation.
    Much faster than UDE (minutes vs hours).
    """
    print("="*70)
    print("SPARSE LINEAR BASELINE - LOSO CROSS-VALIDATION")
    print("="*70)
    print("Model: S(t) = β*S(t-1) + Σᵢ αᵢ*Fᵢ(t)")
    print("Method: Elastic Net (L1 sparsity + L2 stability)")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                       if f.startswith('u_wesad_') and f.endswith('.csv')])
    
    print(f"\nTotal subjects: {len(all_files)}\n")
    
    results = []
    all_coefficients = []
    
    feature_names = None
    
    for fold_idx in range(len(all_files)):
        print(f"Fold {fold_idx+1}/{len(all_files)}")
        
        test_file = all_files[fold_idx]
        train_files = [f for i, f in enumerate(all_files) if i != fold_idx]
        
        # Prepare training data
        X_train_list = []
        y_train_list = []
        
        for train_file in train_files:
            df = pd.read_csv(train_file)
            
            # Get feature columns
            if feature_names is None:
                feature_names = [c for c in df.columns if c not in ['time', 'stress', 'label']]
            
            # Create lagged stress (S(t-1))
            stress_lagged = df['stress'].shift(1).fillna(0).values
            features = df[feature_names].values
            stress_current = df['stress'].values
            
            # Input: [S(t-1), F1(t), F2(t), ..., F18(t)]
            X = np.column_stack([stress_lagged, features])
            y = stress_current
            
            X_train_list.append(X)
            y_train_list.append(y)
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Train Elastic Net (L1 + L2 regularization)
        # L1 ratio = 0.9 means 90% L1 (sparsity), 10% L2 (stability)
        model = ElasticNetCV(l1_ratio=0.9, cv=5, max_iter=10000, random_state=42)
        model.fit(X_train, y_train)
        
        # Test
        df_test = pd.read_csv(test_file)
        stress_lagged_test = df_test['stress'].shift(1).fillna(0).values
        features_test = df_test[feature_names].values
        X_test = np.column_stack([stress_lagged_test, features_test])
        y_test = df_test['stress'].values
        
        y_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        
        # Extract coefficients
        beta = model.coef_[0]  # Memory term
        alphas = model.coef_[1:]  # Feature sensitivities
        
        # Count non-zero (selected features)
        n_selected = np.sum(np.abs(alphas) > 1e-4)
        
        print(f"  Test MSE: {test_mse:.6f}")
        print(f"  Memory (β): {beta:.4f}")
        print(f"  Active features: {n_selected}/{len(alphas)}")
        print(f"  Sparsity: {100*(1-n_selected/len(alphas)):.1f}%\n")
        
        results.append({
            'Fold': fold_idx + 1,
            'Subject': os.path.basename(test_file).replace('.csv', ''),
            'Test_MSE': test_mse,
            'Beta': beta,
            'N_Active_Features': n_selected,
            'Sparsity_%': 100*(1-n_selected/len(alphas))
        })
        
        all_coefficients.append(alphas)
    
    # Summary
    df_results = pd.DataFrame(results)
    coef_matrix = np.array(all_coefficients)
    
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)
    print(f"\nMean Test MSE: {df_results['Test_MSE'].mean():.6f} ± {df_results['Test_MSE'].std():.6f}")
    print(f"Mean Sparsity: {df_results['Sparsity_%'].mean():.1f}%")
    print(f"Mean Active Features: {df_results['N_Active_Features'].mean():.1f}/{len(feature_names)}")
    
    # Save results
    out_dir = os.path.join(base_dir, 'results', 'sparse_linear')
    os.makedirs(out_dir, exist_ok=True)
    
    df_results.to_csv(os.path.join(out_dir, 'loso_results.csv'), index=False)
    
    # Save coefficients
    df_coef = pd.DataFrame(coef_matrix, columns=feature_names)
    df_coef.insert(0, 'Subject', [r['Subject'] for r in results])
    df_coef.insert(1, 'Beta', [r['Beta'] for r in results])
    df_coef.to_csv(os.path.join(out_dir, 'coefficients.csv'), index=False)
    
    # Visualize sparsity
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    sns.heatmap(coef_matrix, 
                xticklabels=feature_names,
                yticklabels=[r['Subject'] for r in results],
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Coefficient (α)'},
                ax=axes[0])
    axes[0].set_title('Sparse Linear Coefficients (White = Zero)', fontweight='bold')
    axes[0].set_xlabel('Features')
    axes[0].set_ylabel('Subjects')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Feature importance (mean absolute coefficient)
    feature_importance = np.abs(coef_matrix).mean(axis=0)
    sorted_idx = np.argsort(feature_importance)[::-1][:10]
    
    axes[1].barh(range(10), feature_importance[sorted_idx], color='steelblue')
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([feature_names[i] for i in sorted_idx])
    axes[1].set_xlabel('Mean |Coefficient|')
    axes[1].set_title('Top 10 Most Important Features', fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'visualization.png'), dpi=150)
    print(f"\n✅ Results saved to: {out_dir}")
    
    return df_results, coef_matrix

if __name__ == "__main__":
    train_sparse_linear_loso()
