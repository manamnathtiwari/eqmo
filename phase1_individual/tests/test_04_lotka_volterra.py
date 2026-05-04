"""
Test 4: Lotka-Volterra (Predator-Prey) System
Tests UDE on classic coupled ODE system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.integrate import odeint

class LotkaVolterraUDE(nn.Module):
    """UDE for Lotka-Volterra equations"""
    def __init__(self):
        super().__init__()
        self.linear_coeffs = nn.Parameter(torch.randn(2) * 0.1)
        self.nn = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, X):
        linear = torch.sum(self.linear_coeffs * X, dim=-1, keepdim=True)
        nonlinear = self.nn(X)
        return (linear + nonlinear).squeeze(-1)

def test_lotka_volterra():
    """
    Lotka-Volterra equations:
    dx/dt = 1.5*x - 1.0*x*y  (prey)
    dy/dt = 1.0*x*y - 1.0*y  (predator)
    """
    print("="*70)
    print("TEST: Lotka-Volterra (Predator-Prey)")
    print("="*70)
    print("Equations:")
    print("  dx/dt = 1.5*x - 1.0*x*y  (prey)")
    print("  dy/dt = 1.0*x*y - 1.0*y  (predator)\n")
    
    # Parameters
    alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 1.0
    
    def lotka_volterra(state, t):
        x, y = state
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return [dxdt, dydt]
    
    # Generate trajectories
    print("Generating data...")
    t_span = np.linspace(0, 15, 100)
    
    all_prey_data = []
    all_pred_data = []
    all_dxdt = []
    all_dydt = []
    
    for _ in range(50):
        x0 = np.random.uniform(0.5, 3.0)
        y0 = np.random.uniform(0.5, 3.0)
        
        solution = odeint(lotka_volterra, [x0, y0], t_span)
        
        for i in range(len(solution)):
            x, y = solution[i]
            all_prey_data.append([x, y])
            all_pred_data.append([x, y])
            
            dxdt = alpha * x - beta * x * y
            dydt = delta * x * y - gamma * y
            all_dxdt.append(dxdt)
            all_dydt.append(dydt)
    
    X_prey = torch.FloatTensor(all_prey_data)
    y_prey = torch.FloatTensor(all_dxdt)
    
    X_pred = torch.FloatTensor(all_pred_data)
    y_pred_true = torch.FloatTensor(all_dydt)
    
    print(f"Generated {len(X_prey)} data points\n")
    
    # Train prey equation
    print("Training prey equation (dx/dt)...")
    model_prey = LotkaVolterraUDE()
    optimizer = optim.Adam(model_prey.parameters(), lr=0.005)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        pred = model_prey(X_prey)
        loss = torch.mean((pred - y_prey) ** 2)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Train predator equation
    print("\nTraining predator equation (dy/dt)...")
    model_predator = LotkaVolterraUDE()
    optimizer = optim.Adam(model_predator.parameters(), lr=0.005)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        pred = model_predator(X_pred)
        loss = torch.mean((pred - y_pred_true) ** 2)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Test
    X_test = torch.FloatTensor([[2.0, 1.0], [1.0, 2.0], [1.5, 1.5]])
    
    with torch.no_grad():
        prey_pred = model_prey(X_test)
        predator_pred = model_predator(X_test)
    
    # True values at test points
    true_prey = [alpha*2.0 - beta*2.0*1.0,
                 alpha*1.0 - beta*1.0*2.0,
                 alpha*1.5 - beta*1.5*1.5]
    
    true_predator = [delta*2.0*1.0 - gamma*1.0,
                     delta*1.0*2.0 - gamma*2.0,
                     delta*1.5*1.5 - gamma*1.5]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("\nTest predictions:")
    print(f"{'Point':<15} {'dx/dt Pred':<12} {'dx/dt True':<12} {'Error':<10}")
    print("-"*50)
    for i, (pred, true) in enumerate(zip(prey_pred, true_prey)):
        error = abs(pred.item() - true)
        print(f"[{X_test[i, 0]:.1f}, {X_test[i, 1]:.1f}]      {pred.item():<12.3f} {true:<12.3f} {error:<10.3f}")
    
    prey_error = np.mean([abs(p.item() - t) for p, t in zip(prey_pred, true_prey)])
    pred_error = np.mean([abs(p.item() - t) for p, t in zip(predator_pred, true_predator)])
    
    print(f"\nAverage prey error:     {prey_error:.4f}")
    print(f"Average predator error: {pred_error:.4f}")
    
    # Pass/Fail
    success = prey_error < 0.5 and pred_error < 0.5
    
    if success:
        print("\n✅ PASS: Lotka-Volterra system captured!")
    else:
        print("\n❌ FAIL: Poor approximation")
    
    return success

if __name__ == "__main__":
    success = test_lotka_volterra()
    exit(0 if success else 1)
