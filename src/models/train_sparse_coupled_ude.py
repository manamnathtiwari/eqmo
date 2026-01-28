import torch
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.sparse_coupled_ude import SparseCoupledUDEModel

def train_sparse_coupled_ude_loso():
    """
    Train Sparse Coupled-UDE using LOSO cross-validation
    
    This is the MAIN model combining:
    - Ridge for feature selection
    - Coupled UDE for mechanistic dynamics
    """
    print("="*70)
    print("SPARSE COUPLED-UDE: LOSO CROSS-VALIDATION")
    print("="*70)
    print("Model: Ridge → Coupled UDE (S ⇄ HRV ⇄ EDA)")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                       if f.startswith('u_wesad_') and f.endswith('.csv')])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Total subjects: {len(all_files)}\n")
    
    results = []
    all_params = []
    
    for fold_idx in range(len(all_files)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/{len(all_files)}")
        print(f"{'='*70}")
        
        test_file = all_files[fold_idx]
        train_files = [f for i, f in enumerate(all_files) if i != fold_idx]
        
        subject_name = os.path.basename(test_file).replace('.csv', '')
        print(f"Test Subject: {subject_name}")
        print(f"Training on: {len(train_files)} subjects")
        
        # Initialize model
        model = SparseCoupledUDEModel(
            ridge_threshold=0.1,
            hidden_size=32,
            device=device
        )
        
        # === STAGE 1: Ridge Feature Selection ===
        print("\n" + "-"*70)
        
        # Load all training data for Ridge
        X_train_list = []
        y_train_list = []
        
        for train_file in train_files:
            df = pd.read_csv(train_file)
            
            # Get features
            feature_cols = [c for c in df.columns if c not in ['time', 'stress', 'label']]
            X = df[feature_cols].values
            y = df['stress'].values
            
            X_train_list.append(X)
            y_train_list.append(y)
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Fit Ridge
        selected_features = model.fit_stage1_ridge(X_train, y_train)
        
        # === STAGE 2: Coupled UDE Training ===
        print("\n" + "-"*70)
        
        model.fit_stage2_coupled_ude(
            train_files=train_files,
            epochs=30,  # Reduce for speed, increase to 50 on Kaggle
            lr=0.005
        )
        
        # === TESTING ===
        print("\n" + "-"*70)
        print("TESTING")
        print("-"*70)
        
        df_test = pd.read_csv(test_file)
        y_test = df_test['stress'].values
        
        try:
            y_pred = model.predict(df_test)
            
            # Trim to same length
            min_len = min(len(y_test), len(y_pred))
            y_test = y_test[:min_len]
            y_pred = y_pred[:min_len]
            
            test_mse = mean_squared_error(y_test, y_pred)
            print(f"\nTest MSE: {test_mse:.6f}")
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            test_mse = np.nan
            y_pred = np.zeros_like(y_test)
        
        # Save results
        results.append({
            'Fold': fold_idx + 1,
            'Subject': subject_name,
            'Test_MSE': test_mse,
            'Selected_Features': ','.join(selected_features),
            'N_Features': len(selected_features)
        })
        
        # Save parameters
        params = model.coupled_ude.get_interpretable_params()
        params['Subject'] = subject_name
        params['Fold'] = fold_idx + 1
        all_params.append(params)
        
        # Save model
        model_dir = os.path.join(base_dir, 'results', 'sparse_coupled_ude', 'models')
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, f'model_fold_{fold_idx+1}.pth'))
        
        print(f"\n✓ Fold {fold_idx+1} complete")
    
    # === SUMMARY ===
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    df_params = pd.DataFrame(all_params)
    
    print("\nPer-Fold Results:")
    print(df_results.to_string(index=False))
    
    valid_mse = df_results['Test_MSE'].dropna()
    if len(valid_mse) > 0:
        print(f"\nMean Test MSE: {valid_mse.mean():.6f} ± {valid_mse.std():.6f}")
    
    # Save results
    out_dir = os.path.join(base_dir, 'results', 'sparse_coupled_ude')
    os.makedirs(out_dir, exist_ok=True)
    
    df_results.to_csv(os.path.join(out_dir, 'loso_results.csv'), index=False)
    df_params.to_csv(os.path.join(out_dir, 'parameters.csv'), index=False)
    
    print(f"\n✅ Results saved to: {out_dir}")
    
    return df_results, df_params

if __name__ == "__main__":
    train_sparse_coupled_ude_loso()
