"""
Symbolic Regression for ALL LOSO Models
Discovers equations from each trained fold
"""
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.ude_model import UDE

def discover_equation_for_model(model_path, fold_num):
    """Discover equation for a single model"""
    
    # Load Model
    model = UDE()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    alpha = model.alpha.item()
    beta = model.beta.item()
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num}")
    print(f"{'='*70}")
    print(f"Learned Parameters: α={alpha:.4f}, β={beta:.4f}, Risk={alpha/(beta+1e-6):.2f}")
    
    # Generate data points for regression
    # We want to approximate the effective dynamics f(S, W) learned by the high-dim model.
    # We perform a "Partial Dependence" analysis:
    # - Vary S and W (Workload is Feature 0)
    # - Fix all other 17 features to 0 (mean value in normalized space)
    
    S_vals = np.linspace(0, 1, 50)
    W_vals = np.linspace(0, 1, 50)
    
    S_grid, W_grid = np.meshgrid(S_vals, W_vals)
    S_flat = S_grid.flatten()
    W_flat = W_grid.flatten()
    n_samples = len(S_flat)
    
    # Construct input vector: [S, W, feat2, feat3, ..., feat18]
    # S is standalone input (col 0 in cat)
    # W is feat 0 (col 1 in cat)
    
    # 1. Stress Input (N, 1)
    s_tensor = torch.tensor(S_flat, dtype=torch.float32).unsqueeze(1)
    
    # 2. Feature Input (N, 18)
    feat_tensor = torch.zeros((n_samples, 18), dtype=torch.float32)
    feat_tensor[:, 0] = torch.tensor(W_flat, dtype=torch.float32) # Set Workload
    
    with torch.no_grad():
        # UDE.forward expects inputs, but we want to isolate the Neural Term component
        # accessible via model.net().
        # UDE.net input is cat([S, features]) -> Shape (N, 19)
        
        nn_in = torch.cat([s_tensor, feat_tensor], dim=-1)
        nn_out = model.net(nn_in).numpy().flatten()
        
    # For symbolic regression, we only care about how it depends on S and W
    inputs = np.stack([S_flat, W_flat], axis=1) # (N, 2) passed to poly features
    
    # Symbolic Regression
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_features = poly.fit_transform(inputs)
    feature_names = poly.get_feature_names_out(['S', 'W'])
    
    # Sparse regression
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_features, nn_out)
    
    # Extract significant terms
    equation_terms = []
    for coef, name in zip(lasso.coef_, feature_names):
        if abs(coef) > 1e-3:
            equation_terms.append((name, coef))
    
    # Print discovered equation
    print("\nDiscovered Neural Network Correction:")
    if not equation_terms:
        print("  0 (negligible - pure physics model)")
        correction = "0"
    else:
        terms_str = []
        for name, coef in equation_terms:
            terms_str.append(f"{coef:+.4f}*{name}")
        correction = " ".join(terms_str)
        print(f"  {correction}")
    
    # Full equation
    print("\n📐 Full Discovered Dynamics:")
    print(f"  dS/dt = -{beta:.4f}·S + {alpha:.4f}·W + ({correction})")
    print()
    
    return {
        'fold': fold_num,
        'alpha': alpha,
        'beta': beta,
        'risk': alpha/(beta+1e-6),
        'correction': correction,
        'num_terms': len(equation_terms)
    }

def main():
    print("="*70)
    print("SYMBOLIC REGRESSION: DISCOVER EQUATIONS FROM ALL LOSO MODELS")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, 'results', 'loso_models')
    
    # Find all fold models
    model_files = sorted([f for f in os.listdir(models_dir) 
                         if f.startswith('ude_fold_') and f.endswith('.pth')])
    
    print(f"\nFound {len(model_files)} trained models")
    
    results = []
    
    for model_file in model_files:
        fold_num = int(model_file.split('_')[-1].split('.')[0])
        model_path = os.path.join(models_dir, model_file)
        
        result = discover_equation_for_model(model_path, fold_num)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: DISCOVERED EQUATIONS")
    print("="*70)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save
    output_dir = os.path.join(base_dir, 'results', 'symbolic')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'discovered_equations.csv')
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Average parameters
    print(f"\n📊 AVERAGE ACROSS ALL FOLDS:")
    print(f"  α (Sensitivity): {df['alpha'].mean():.4f} ± {df['alpha'].std():.4f}")
    print(f"  β (Recovery):    {df['beta'].mean():.4f} ± {df['beta'].std():.4f}")
    print(f"  Risk Score:      {df['risk'].mean():.2f} ± {df['risk'].std():.2f}")

if __name__ == "__main__":
    main()
