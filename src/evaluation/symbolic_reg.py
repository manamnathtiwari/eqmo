import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_model import UDE

def discover_equation():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, 'results', 'ude_model.pth')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # 1. Load Model
    model = UDE()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print(f"Learned Parameters: Beta={model.beta.item():.4f}, Alpha={model.alpha.item():.4f}")
    
    # 2. Generate data points for regression
    # Sample S and W in reasonable ranges
    S_vals = np.linspace(0, 1, 50)
    W_vals = np.linspace(0, 1, 50)
    
    S_grid, W_grid = np.meshgrid(S_vals, W_vals)
    S_flat = S_grid.flatten()
    W_flat = W_grid.flatten()
    
    inputs = torch.tensor(np.stack([S_flat, W_flat], axis=1), dtype=torch.float32) # (N, 2)
    
    with torch.no_grad():
        # Get the neural term output
        # f_nn = model.net(inputs)
        nn_out = model.net(inputs).numpy().flatten()
        
    # 3. Symbolic Regression (SINDy style with Lasso)
    # We want to approximate nn_out = f(S, W)
    # Candidate library: 1, S, W, S^2, W^2, SW, S^3...
    
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_features = poly.fit_transform(inputs.numpy())
    feature_names = poly.get_feature_names_out(['S', 'W'])
    
    # Sparse regression
    lasso = Lasso(alpha=0.001) # alpha controls sparsity
    lasso.fit(X_features, nn_out)
    
    # 4. Print Result
    print("\nDiscovered Correction Term (Neural Net approximation):")
    equation = []
    for coef, name in zip(lasso.coef_, feature_names):
        if abs(coef) > 1e-3:
            equation.append(f"{coef:+.4f}*{name}")
            
    if not equation:
        print("0 (Neural net learned negligible correction)")
    else:
        print(" ".join(equation))
        
    print("\nFull Discovered Dynamics:")
    print(f"dS/dt = -{model.beta.item():.4f}*S + {model.alpha.item():.4f}*W + ({' '.join(equation)})")

if __name__ == "__main__":
    discover_equation()
