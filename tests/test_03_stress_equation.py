"""
Test 3: Stress Equation Recovery
Tests UDE on stress-like equation: dS/dt = -0.2*S + 0.8*HRV + 0.5*EDA
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class StressUDE(nn.Module):
    """UDE for stress equation"""
    def __init__(self, n_features=3):
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(n_features) * 0.1)
        self.nn = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, X):
        linear = torch.sum(self.coefficients * X, dim=-1, keepdim=True)
        nonlinear = self.nn(X)
        return (linear + nonlinear).squeeze(-1)

def test_stress_equation():
    """
    Test equation: dS/dt = -0.2*S + 0.8*HRV + 0.5*EDA
    Expected: Recover coefficients [-0.2, 0.8, 0.5]
    """
    print("="*70)
    print("TEST: Stress Equation Recovery")
    print("="*70)
    print("Equation: dS/dt = -0.2*S + 0.8*HRV + 0.5*EDA\n")
    
    # True coefficients [S, HRV, EDA]
    true_coeffs = torch.tensor([-0.2, 0.8, 0.5])
    
    # Generate realistic stress data
    n_samples = 1500
    
    # Realistic ranges
    S = torch.rand(n_samples, 1) * 0.8  # Stress: 0-0.8
    HRV = torch.rand(n_samples, 1) * 0.5 + 0.3  # HRV: 0.3-0.8
    EDA = torch.rand(n_samples, 1) * 0.3 + 0.1  # EDA: 0.1-0.4
    
    X_train = torch.cat([S, HRV, EDA], dim=1)
    y_train = torch.sum(true_coeffs * X_train, dim=-1)
    
    # Train UDE
    model = StressUDE(n_features=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training...")
    for epoch in range(800):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = torch.mean((y_pred - y_train) ** 2)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Test
    S_test = torch.rand(300, 1) * 0.8
    HRV_test = torch.rand(300, 1) * 0.5 + 0.3
    EDA_test = torch.rand(300, 1) * 0.3 + 0.1
    X_test = torch.cat([S_test, HRV_test, EDA_test], dim=1)
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
    print(f"{'Variable':<10} {'True':<10} {'Learned':<10} {'Error':<10}")
    print("-"*40)
    labels = ['Stress', 'HRV', 'EDA']
    for label, true, learned, error in zip(labels, true_coeffs.numpy(), learned_coeffs, errors):
        print(f"{label:<10} {true:<10.3f} {learned:<10.3f} {error:<10.3f}")
    
    print(f"\nAverage error: {avg_error:.4f}")
    print(f"Test MSE:      {test_mse:.6f}")
    
    # Interpretation
    print("\nInterpretation:")
    print(f"  Stress decay:  {learned_coeffs[0]:.3f} (negative = stress decreases over time)")
    print(f"  HRV effect:    {learned_coeffs[1]:.3f} (positive = higher HRV reduces stress)")
    print(f"  EDA effect:    {learned_coeffs[2]:.3f} (positive = higher EDA increases stress)")
    
    # Pass/Fail
    success = avg_error < 0.2 and test_mse < 0.01
    
    if success:
        print("\n✅ PASS: Stress equation recovered!")
    else:
        print("\n❌ FAIL: Poor recovery")
    
    return success

if __name__ == "__main__":
    success = test_stress_equation()
    exit(0 if success else 1)
