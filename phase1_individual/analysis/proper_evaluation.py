"""
Proper Evaluation Script - Matching Exact Training Conditions

This evaluates the 13 Kaggle models using the EXACT same setup as training:
- seq_len=100 (same as training)
- batch_size=16 (same as training)
- Overlapping sequences (same as training)
- Same evaluation method

This should give us the accurate test MSE to compare with LSTM baseline.
"""

import torch
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.ude_multicoeff import UDEMultiCoeff
from src.utils import StressDataset, FEATURE_COLUMNS
from torch.utils.data import DataLoader
from torchdiffeq import odeint


def evaluate_model_exact(model, test_loader, device='cpu'):
    """
    Evaluate model using EXACT same method as training
    
    This matches lines 98-117 of train_multicoeff.py
    """
    model = model.to(device)
    model.eval()
    test_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            # EXACT same as training evaluation
            t = batch['t'][0].to(device)
            y = batch['y'].to(device).squeeze(-1)
            features = batch['features'].to(device)
            
            y0 = y[:, 0:1]
            
            model.set_current_batch(t, features)
            y_pred = odeint(model, y0, t, method='euler')
            y_pred = y_pred.permute(1, 0, 2).squeeze(-1)
            
            loss = torch.mean((y_pred - y) ** 2)
            test_losses.append(loss.item())
    
    test_loss = np.mean(test_losses)
    
    return test_loss


def main():
    """
    Evaluate all 13 models with EXACT training conditions
    """
    print("="*70)
    print("PROPER EVALUATION - MATCHING TRAINING CONDITIONS")
    print("="*70)
    print("Config: seq_len=100, batch_size=16, overlapping sequences")
    print("="*70)
    
    # Paths
    project_dir = Path(__file__).parent.parent
    models_dir = project_dir / 'analysis' / 'kaggle_13_models' / 'models'
    data_dir = project_dir / 'data' / 'processed' / 'normalized'
    
    # Get CSV files
    csv_files = sorted([
        data_dir / f 
        for f in os.listdir(data_dir) 
        if f.endswith('.csv') and f.startswith('u_wesad')
    ])
    
    print(f"\nFound {len(csv_files)} subjects")
    print(f"Models directory: {models_dir}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    results = []
    
    # Evaluate each fold
    for fold in range(1, 14):  # 13 folds
        model_file = models_dir / f'multicoeff_ude_fold_{fold}.pth'
        
        if not model_file.exists():
            print(f"⚠️  Fold {fold}: Model not found")
            continue
        
        # Load model
        model = UDEMultiCoeff(hidden_dim=64, num_features=18)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        
        # Get corresponding test file
        test_file = csv_files[fold - 1]
        
        print(f"Fold {fold}: Testing on {test_file.name}")
        
        # Create test dataset with EXACT same parameters as training
        test_dataset = StressDataset(str(test_file), seq_len=100)  # seq_len=100 like training
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # batch_size=16 like training
        
        print(f"  Test sequences: {len(test_dataset)} (overlapping)")
        
        # Evaluate with EXACT same method
        test_mse = evaluate_model_exact(model, test_loader, device)
        
        print(f"  Test MSE: {test_mse:.6f}")
        
        results.append({
            'Fold': fold,
            'Subject': test_file.name,
            'Test_MSE': test_mse
        })
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY (PROPER METHOD)")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    
    mean_mse = results_df['Test_MSE'].mean()
    std_mse = results_df['Test_MSE'].std()
    
    print(f"\n{'='*70}")
    print(f"Mean Test MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"{'='*70}")
    
    # Compare with baselines
    lstm_mse = 0.0098  # From literature/previous work
    improvement = ((lstm_mse - mean_mse) / lstm_mse) * 100
    
    print(f"\nComparison with Baselines:")
    print(f"  LSTM MSE:        {lstm_mse:.6f}")
    print(f"  Your Model MSE:  {mean_mse:.6f}")
    print(f"  Improvement:     {improvement:.1f}%")
    
    if improvement > 0:
        print(f"  ✅ Your model is {improvement:.1f}% BETTER than LSTM!")
    else:
        print(f"  ⚠️  Your model is {abs(improvement):.1f}% worse than LSTM")
    
    # Save results
    output_path = project_dir / 'analysis' / 'kaggle_13_models' / 'proper_evaluation_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved results to: {output_path}")
    print("="*70)
    
    return results_df


if __name__ == "__main__":
    results = main()
