import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class LSTMStressModel(nn.Module):
    def __init__(self, input_size=18, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMStressModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def prepare_sequences(df, lookback=20):
    feature_cols = [c for c in df.columns if c not in ['time', 'stress', 'label']]
    features = df[feature_cols].values
    stress = df['stress'].values
    
    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(features[i-lookback:i])
        y.append(stress[i])
    
    return np.array(X), np.array(y)

def train_lstm_loso():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                       if f.startswith('u_wesad_') and f.endswith('.csv')])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    for fold_idx in range(len(all_files)):
        test_file = all_files[fold_idx]
        train_files = [f for i, f in enumerate(all_files) if i != fold_idx]
        
        X_train_list, y_train_list = [], []
        for train_file in train_files:
            df = pd.read_csv(train_file)
            X, y = prepare_sequences(df)
            X_train_list.append(X)
            y_train_list.append(y)
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        df_test = pd.read_csv(test_file)
        X_test, y_test = prepare_sequences(df_test)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        model = LSTMStressModel(input_size=X_train.shape[-1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        model.train()
        for epoch in range(30):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            predictions = model(X_test_t).cpu().numpy().flatten()
        
        mse = mean_squared_error(y_test, predictions)
        
        results.append({
            'Fold': fold_idx + 1,
            'Subject': os.path.basename(test_file).replace('.csv', ''),
            'Test_MSE': mse
        })
        
        print(f"Fold {fold_idx+1}/15: MSE = {mse:.6f}")
    
    df = pd.DataFrame(results)
    out_dir = os.path.join(base_dir, 'results', 'lstm_model')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'loso_results.csv'), index=False)
    
    print(f"\nMean MSE: {df['Test_MSE'].mean():.6f} ± {df['Test_MSE'].std():.6f}")
    return df

if __name__ == "__main__":
    train_lstm_loso()
