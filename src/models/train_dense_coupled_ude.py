import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from torchdiffeq import odeint

import sys
sys.path.append(os.path.dirname(__file__))
from dense_coupled_ude import DenseCoupledUDE

def train_dense_coupled_ude_loso():
    """
    Train Dense Coupled-UDE using LOSO cross-validation
    
    Uses ALL 18 features with 4 coupled variables:
    - Stress (target)
    - HRV (primary cardiac)
    - EDA (primary arousal)
    - Workload (cognitive load)
    """
    print("="*70)
    print("DENSE COUPLED-UDE: LOSO CROSS-VALIDATION")
    print("="*70)
    print("Model: 4 Coupled Variables × 18 Features")
    print("Variables: Stress ⇄ HRV ⇄ EDA ⇄ Workload")
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
        
        # Load and prepare data
        all_sequences = []
        all_features = []
        
        for train_file in train_files:
            df = pd.read_csv(train_file)
            
            # Get stress and coupled variables
            stress = df['stress'].values
            hrv = df['hrv_rmssd'].values if 'hrv_rmssd' in df.columns else df.iloc[:, 1].values
            eda = df['eda_mean_norm'].values if 'eda_mean_norm' in df.columns else df.iloc[:, 6].values
            workload = df['workload'].values if 'workload' in df.columns else df.iloc[:, -1].values
            
            # Get all 18 features
            feature_cols = [c for c in df.columns if c not in ['time', 'stress', 'label']][:18]
            features = df[feature_cols].values
            
            # Create sequences
            seq_len = 60
            for i in range(0, len(stress) - seq_len, seq_len // 2):
                y_seq = np.column_stack([
                    stress[i:i+seq_len],
                    hrv[i:i+seq_len],
                    eda[i:i+seq_len],
                    workload[i:i+seq_len]
                ])
                f_seq = features[i:i+seq_len]
                
                if len(y_seq) == seq_len:
                    all_sequences.append(y_seq)
                    all_features.append(f_seq)
        
        print(f"Created {len(all_sequences)} training sequences")
        
        # Convert to tensors
        sequences = torch.FloatTensor(np.array(all_sequences)).to(device)
        features_tensor = torch.FloatTensor(np.array(all_features)).to(device)
        t = torch.linspace(0, 1, 60).to(device)
        
        # Initialize model
        model = DenseCoupledUDE(n_features=18, hidden_size=64).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        # Training loop
        epochs = 50
        print(f"\nTraining for {epochs} epochs...")
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            batch_size = min(32, len(sequences))
            indices = torch.randperm(len(sequences))
            
            for i in range(0, len(sequences), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_y = sequences[batch_idx]
                batch_f = features_tensor[batch_idx]
                
                y0 = batch_y[:, 0, :]
                y_true = batch_y
                
                # ODE solve with features
                optimizer.zero_grad()
                
                # Custom ODE function that includes features
                def ode_func(t_val, y_val):
                    # Get features at this time step
                    t_idx = int((t_val / t[-1]) * (batch_f.shape[1] - 1))
                    t_idx = min(t_idx, batch_f.shape[1] - 1)
                    f_val = batch_f[:, t_idx, :]
                    return model(t_val, y_val, f_val)
                
                y_pred = odeint(ode_func, y0, t, method='euler', options={'step_size': 0.02})
                y_pred = y_pred.permute(1, 0, 2)
                
                loss = torch.mean((y_pred - y_true) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / (len(sequences) / batch_size)
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        
        # Testing
        print("\nTesting...")
        df_test = pd.read_csv(test_file)
        
        stress_test = df_test['stress'].values
        hrv_test = df_test['hrv_rmssd'].values if 'hrv_rmssd' in df_test.columns else df_test.iloc[:, 1].values
        eda_test = df_test['eda_mean_norm'].values if 'eda_mean_norm' in df_test.columns else df_test.iloc[:, 6].values
        workload_test = df_test['workload'].values if 'workload' in df_test.columns else df_test.iloc[:, -1].values
        
        feature_cols_test = [c for c in df_test.columns if c not in ['time', 'stress', 'label']][:18]
        features_test = df_test[feature_cols_test].values
        
        y_test_full = np.column_stack([stress_test, hrv_test, eda_test, workload_test])
        
        model.eval()
        with torch.no_grad():
            y0_test = torch.FloatTensor(y_test_full[0:1]).to(device)
            features_test_tensor = torch.FloatTensor(features_test).to(device)
            t_test = torch.linspace(0, len(stress_test)-1, len(stress_test)).to(device)
            
            def ode_func_test(t_val, y_val):
                t_idx = int((t_val / t_test[-1]) * (len(features_test) - 1))
                t_idx = min(t_idx, len(features_test) - 1)
                f_val = features_test_tensor[t_idx:t_idx+1]
                if f_val.shape[0] == 0:
                    f_val = features_test_tensor[-1:]
                return model(t_val, y_val, f_val)
            
            try:
                y_pred = odeint(ode_func_test, y0_test, t_test[:1000], method='euler', options={'step_size': 0.1})
                stress_pred = y_pred[:, 0, 0].cpu().numpy()
                
                min_len = min(len(stress_test), len(stress_pred))
                test_mse = mean_squared_error(stress_test[:min_len], stress_pred[:min_len])
            except:
                test_mse = np.nan
        
        print(f"Test MSE: {test_mse:.6f}")
        
        # Save results
        params = model.get_interpretable_params()
        params['Subject'] = subject_name
        params['Fold'] = fold_idx + 1
        params['Test_MSE'] = test_mse
        
        results.append({
            'Fold': fold_idx + 1,
            'Subject': subject_name,
            'Test_MSE': test_mse
        })
        
        all_params.append(params)
        
        # Save model
        model_dir = os.path.join(base_dir, 'results', 'dense_coupled_ude', 'models')
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_fold_{fold_idx+1}.pth'))
        
        print(f"\n✓ Fold {fold_idx+1} complete")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    print("\nPer-Fold Results:")
    print(df_results.to_string(index=False))
    
    valid_mse = df_results['Test_MSE'].dropna()
    if len(valid_mse) > 0:
        print(f"\nMean Test MSE: {valid_mse.mean():.6f} ± {valid_mse.std():.6f}")
    
    # Save
    out_dir = os.path.join(base_dir, 'results', 'dense_coupled_ude')
    os.makedirs(out_dir, exist_ok=True)
    
    df_results.to_csv(os.path.join(out_dir, 'loso_results.csv'), index=False)
    
    # Save parameters (flatten feature sensitivities)
    params_flat = []
    for p in all_params:
        p_flat = {k: v for k, v in p.items() if k != 'feature_sensitivities'}
        params_flat.append(p_flat)
    
    df_params = pd.DataFrame(params_flat)
    df_params.to_csv(os.path.join(out_dir, 'parameters.csv'), index=False)
    
    print(f"\n✅ Results saved to: {out_dir}")
    
    return df_results, df_params

if __name__ == "__main__":
    train_dense_coupled_ude_loso()
