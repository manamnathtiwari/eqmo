"""
Explainability Analysis Script
------------------------------
Extracts and visualizes the learned physical parameters (Alpha, Beta) 
from the trained UDE models for each subject.

Parameters:
- Alpha: Sensitivity to Workload (How much workload spikes stress)
- Beta:  Recovery Rate (How fast stress dissipates naturally)

High Alpha = High Sensitivity (Bad)
High Beta  = High Resilience (Good)
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.ude_model import UDE

def analyze_explainability():
    print("="*70)
    print("EXPLAINABILITY ANALYSIS: PERSONALIZED STRESS PROFILES")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, 'results', 'loso_models')
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    
    # Get all subject files to map Fold ID to Subject ID
    all_files = sorted([f for f in os.listdir(data_dir) if f.startswith('u_wesad_') and f.endswith('.csv')])
    
    # Find completed folds
    model_files = [f for f in os.listdir(models_dir) if f.startswith('ude_fold_') and f.endswith('.pth')]
    model_files.sort()
    
    results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Found {len(model_files)} trained models.")
    
    for f in model_files:
        # Parse Fold ID
        fold_idx = int(f.split('_')[-1].split('.')[0]) - 1 # 0-indexed
        subject_file = all_files[fold_idx]
        subject_id = os.path.basename(subject_file).replace('.csv', '')
        
        # Load Model
        model = UDE().to(device)
        model.load_state_dict(torch.load(os.path.join(models_dir, f), map_location=device))
        
        # Extract Parameters using the new method that handles softplus constraints
        params = model.get_interpretable_params()
        alpha = params['alpha']
        beta = params['beta']
        risk_score = params['risk_score']
        
        results.append({
            'Subject': subject_id,
            'Alpha (Sensitivity)': alpha,
            'Beta (Recovery)': beta,
            'Burnout Risk': risk_score
        })
        
    df = pd.DataFrame(results)
    
    # Display Results
    print("\nLearned Physiological Parameters:")
    print("-" * 60)
    print(df.to_string(index=False))
    print("-" * 60)
    
    # Interpretations
    print("\nINTERPRETATION:")
    best_subject = df.loc[df['Burnout Risk'].idxmin()]
    worst_subject = df.loc[df['Burnout Risk'].idxmax()]
    
    print(f"Most Resilient: {best_subject['Subject']}")
    print(f"   - Low Sensitivity ({best_subject['Alpha (Sensitivity)']:.3f})")
    print(f"   - Fast Recovery ({best_subject['Beta (Recovery)']:.3f})")
    
    print(f"\nHighest Burnout Risk: {worst_subject['Subject']}")
    print(f"   - High Sensitivity ({worst_subject['Alpha (Sensitivity)']:.3f})")
    print(f"   - Slow Recovery ({worst_subject['Beta (Recovery)']:.3f})")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    sns.scatterplot(data=df, x='Beta (Recovery)', y='Alpha (Sensitivity)', s=200, hue='Burnout Risk', palette='RdYlGn_r')
    
    # Add labels
    for i in range(df.shape[0]):
        plt.text(df['Beta (Recovery)'][i]+0.002, df['Alpha (Sensitivity)'][i], 
                 df['Subject'][i], fontdict={'size': 12})
        
    plt.title('Personalized Stress Profiles: Sensitivity vs. Recovery', fontsize=14, fontweight='bold')
    plt.xlabel('Beta: Recovery Rate (Higher is Better) →')
    plt.ylabel('Alpha: Stress Sensitivity (Lower is Better) →')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    out_dir = os.path.join(base_dir, 'results', 'explainability')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'stress_profiles.png'), dpi=150)
    print(f"\nPlot saved to: {os.path.join(out_dir, 'stress_profiles.png')}")
    
    # Save CSV
    df.to_csv(os.path.join(out_dir, 'parameters.csv'), index=False)

if __name__ == "__main__":
    analyze_explainability()
