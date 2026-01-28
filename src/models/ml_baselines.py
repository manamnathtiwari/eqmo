"""
Comprehensive ML Baselines Suite
---------------------------------
Implements standard ML methods for stress prediction.
Fast training (all run locally in ~30 minutes).

Models:
1. Ridge Regression (Dense linear)
2. LSTM (Deep learning)
3. SVR (Kernel methods)

All use same LOSO protocol for fair comparison.
"""
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Last timestep
        return out

def prepare_data(file_path, lookback=10):
    """Prepare data for ML models"""
    df = pd.read_csv(file_path)
    feature_cols = [c for c in df.columns if c not in ['time', 'stress', 'label']]
    
    # Create sequences
    X_list = []
    y_list = []
    
    features = df[feature_cols].values
    stress = df['stress'].values
    
    for i in range(lookback, len(df)):
        # For LSTM: sequence
        X_seq = features[i-lookback:i]
        # For other models: flatten
        X_flat = np.concatenate([
            stress[i-lookback:i],  # Past stress
            features[i-lookback:i].flatten()  # Past features
        ])
        
        y = stress[i]
        
        X_list.append((X_seq, X_flat))
        y_list.append(y)
    
    X_seq = np.array([x[0] for x in X_list])
    X_flat = np.array([x[1] for x in X_list])
    y = np.array(y_list)
    
    return X_seq, X_flat, y, feature_cols

def train_all_baselines():
    print("="*70)
    print("COMPREHENSIVE ML BASELINES - LOSO CROSS-VALIDATION")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                       if f.startswith('u_wesad_') and f.endswith('.csv')])
    
    print(f"Total subjects: {len(all_files)}\n")
    
    results = {
        'Fold': [],
        'Subject': [],
        'Ridge': [],
        'XGBoost': [],
        'SVR': [],
        'LSTM': []
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold_idx in range(len(all_files)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/{len(all_files)}")
        print(f"{'='*70}")
        
        test_file = all_files[fold_idx]
        train_files = [f for i, f in enumerate(all_files) if i != fold_idx]
        subject_name = os.path.basename(test_file).replace('.csv', '')
        
        print(f"Test: {subject_name}")
        
        # Prepare training data
        X_seq_train_list = []
        X_flat_train_list = []
        y_train_list = []
        
        for train_file in train_files:
            X_seq, X_flat, y, _ = prepare_data(train_file)
            X_seq_train_list.append(X_seq)
            X_flat_train_list.append(X_flat)
            y_train_list.append(y)
        
        X_seq_train = np.vstack(X_seq_train_list)
        X_flat_train = np.vstack(X_flat_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Prepare test data
        X_seq_test, X_flat_test, y_test, feature_cols = prepare_data(test_file)
        
        # Standardize
        scaler = StandardScaler()
        X_flat_train_scaled = scaler.fit_transform(X_flat_train)
        X_flat_test_scaled = scaler.transform(X_flat_test)
        
        # === MODEL 1: Ridge Regression ===
        print("\n  Training Ridge Regression...", end=" ")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_flat_train_scaled, y_train)
        y_pred_ridge = ridge.predict(X_flat_test_scaled)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        print(f"MSE = {mse_ridge:.6f}")
        
        # === MODEL 2: XGBoost ===
        print("  Training XGBoost...", end=" ")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_flat_train, y_train)
        y_pred_xgb = xgb_model.predict(X_flat_test)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        print(f"MSE = {mse_xgb:.6f}")
        
        # === MODEL 3: SVR ===
        print("  Training SVR (RBF kernel)...", end=" ")
        # Subsample for speed (SVR is slow)
        n_train_svr = min(5000, len(X_flat_train_scaled))
        idx = np.random.choice(len(X_flat_train_scaled), n_train_svr, replace=False)
        
        svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr_model.fit(X_flat_train_scaled[idx], y_train[idx])
        y_pred_svr = svr_model.predict(X_flat_test_scaled)
        mse_svr = mean_squared_error(y_test, y_pred_svr)
        print(f"MSE = {mse_svr:.6f}")
        
        # === MODEL 4: LSTM ===
        print("  Training LSTM (10 epochs)...", end=" ")
        lstm_model = LSTMModel(input_size=X_seq_train.shape[2]).to(device)
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Convert to PyTorch
        X_seq_train_t = torch.FloatTensor(X_seq_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        
        # Quick training
        lstm_model.train()
        batch_size = 256
        for epoch in range(10):
            for i in range(0, len(X_seq_train_t), batch_size):
                batch_X = X_seq_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                optimizer.zero_grad()
                pred = lstm_model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        
        # Test
        lstm_model.eval()
        with torch.no_grad():
            X_seq_test_t = torch.FloatTensor(X_seq_test).to(device)
            y_pred_lstm = lstm_model(X_seq_test_t).cpu().numpy().flatten()
        
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)
        print(f"MSE = {mse_lstm:.6f}")
        
        # Store results
        results['Fold'].append(fold_idx + 1)
        results['Subject'].append(subject_name)
        results['Ridge'].append(mse_ridge)
        results['XGBoost'].append(mse_xgb)
        results['SVR'].append(mse_svr)
        results['LSTM'].append(mse_lstm)
    
    # Summary
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("AVERAGE MSE (Lower is Better)")
    print("="*70)
    for model in ['Ridge', 'XGBoost', 'SVR', 'LSTM']:
        mean_mse = df[model].mean()
        std_mse = df[model].std()
        print(f"{model:15s}: {mean_mse:.6f} ± {std_mse:.6f}")
    
    # Save
    out_dir = os.path.join(base_dir, 'results', 'ml_baselines')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'loso_results.csv'), index=False)
    print(f"\n✅ Results saved to: {out_dir}")
    
    return df

if __name__ == "__main__":
    train_all_baselines()
