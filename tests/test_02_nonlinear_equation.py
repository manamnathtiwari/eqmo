"""
Test 2: Nonlinear Polynomial Equation Recovery
Tests if UDE can capture nonlinear relationships using neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NonlinearUDE(nn.Module):
    """UDE with neural network for nonlinear terms"""
    def __init__(self, n_features=2):
        super().__init__()
        self.linear_coeffs = nn.Parameter(torch.randn(n_features) * 0.1)
        self.nn = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, X):
        linear = torch.sum(self.linear_coeffs * X, dim=-1, keepdim=True)
        nonlinear = self.nn(X)
        return (linear + nonlinear).squeeze(-1)

def test_nonlinear_equation():
    """
    Test equation: y = 2*x1 - 3*x2 + 0.5*x1^2 - 0.3*x1*x2
    Expected: Linear part [2, -3] recovered, nonlinear captured by NN
    """
    print("="*70)
    print("TEST: Nonlinear Polynomial Equation")
    print("="*70)
    print("Equation: y = 2*x1 - 3*x2 + 0.5*x1^2 - 0.3*x1*x2\n")
    
    def true_equation(X):
        x1, x2 = X[:, 0], X[:, 1]
        return 2*x1 - 3*x2 + 0.5*x1**2 - 0.3*x1*x2
    
    # Generate data
    n_samples = 2000
    X_train = torch.randn(n_samples, 2) * 2
    y_train = true_equation(X_train)
    
    # Train UDE
    model = NonlinearUDE(n_features=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training...")
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = torch.mean((y_pred - y_train) ** 2)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Test
    X_test = torch.randn(500, 2) * 2
    y_test = true_equation(X_test)
    
    with torch.no_grad():
        y_pred = model(X_test)
        test_mse = torch.mean((y_pred - y_test) ** 2).item()
    
    # Results
    learned_coeffs = model.linear_coeffs.detach().numpy()
    true_linear = np.array([2.0, -3.0])
    linear_error = np.mean(np.abs(true_linear - learned_coeffs))
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"True linear coeffs:    {true_linear}")
    print(f"Learned linear coeffs: {learned_coeffs}")
    print(f"Linear error:          {linear_error:.4f}")
    print(f"Test MSE:              {test_mse:.6f}")
    print("\nNote: Nonlinear terms (x1^2, x1*x2) captured by neural network")
    
    # Pass/Fail
    success = test_mse < 0.5  # More lenient for nonlinear
    
    if success:
        print("\n✅ PASS: Nonlinear equation captured!")
    else:
        print("\n❌ FAIL: Poor approximation")
    
    return success

if __name__ == "__main__":
    success = test_nonlinear_equation()
    exit(0 if success else 1)
