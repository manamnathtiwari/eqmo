"""
Test 1: Linear Equation Recovery
Tests if UDE can recover simple linear equations from data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearUDE(nn.Module):
    """Simple UDE for linear equation recovery"""
    def __init__(self, n_features=3):
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(n_features) * 0.1)
    
    def forward(self, X):
        return torch.sum(self.coefficients * X, dim=-1)

def test_linear_equation():
    """
    Test equation: y = 2*x1 - 3*x2 + 1*x3
    Expected: UDE should recover coefficients [2, -3, 1]
    """
    print("="*70)
    print("TEST: Linear Equation Recovery")
    print("="*70)
    print("Equation: y = 2*x1 - 3*x2 + 1*x3\n")
    
    # True coefficients
    true_coeffs = torch.tensor([2.0, -3.0, 1.0])
    
    # Generate training data
    n_samples = 1000
    X_train = torch.randn(n_samples, 3)
    y_train = torch.sum(true_coeffs * X_train, dim=-1)
    
    # Train UDE
    model = LinearUDE(n_features=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training...")
    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = torch.mean((y_pred - y_train) ** 2)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Test on new data
    X_test = torch.randn(200, 3)
    y_test = torch.sum(true_coeffs * X_test, dim=-1)
    
    with torch.no_grad():
        y_pred = model(X_test)
        test_mse = torch.mean((y_pred - y_test) ** 2).item()
    
    # Results
    learned_coeffs = model.coefficients.detach().numpy()
    errors = np.abs(true_coeffs.numpy() - learned_coeffs)
    avg_error = np.mean(errors)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"True coefficients:    {true_coeffs.numpy()}")
    print(f"Learned coefficients: {learned_coeffs}")
    print(f"Errors:               {errors}")
    print(f"Average error:        {avg_error:.4f}")
    print(f"Test MSE:             {test_mse:.6f}")
    
    # Pass/Fail
    success = avg_error < 0.1 and test_mse < 0.01
    
    if success:
        print("\n✅ PASS: Linear equation recovered successfully!")
    else:
        print("\n❌ FAIL: Equation recovery failed")
    
    return success

if __name__ == "__main__":
    success = test_linear_equation()
    exit(0 if success else 1)
