import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

def prepare_data(df, lookback=10):
    feature_cols = [c for c in df.columns if c not in ['time', 'stress', 'label']]
    features = df[feature_cols].values
    stress = df['stress'].values
    
    X, y = [], []
    for i in range(lookback, len(df)):
        X_row = np.concatenate([stress[i-lookback:i], features[i-lookback:i].flatten()])
        X.append(X_row)
        y.append(stress[i])
    
    return np.array(X), np.array(y)

def train_ridge_loso():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed', 'normalized')
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                       if f.startswith('u_wesad_') and f.endswith('.csv')])
    
    results = []
    
    for fold_idx in range(len(all_files)):
        test_file = all_files[fold_idx]
        train_files = [f for i, f in enumerate(all_files) if i != fold_idx]
        
        X_train_list, y_train_list = [], []
        for train_file in train_files:
            df = pd.read_csv(train_file)
            X, y = prepare_data(df)
            X_train_list.append(X)
            y_train_list.append(y)
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        df_test = pd.read_csv(test_file)
        X_test, y_test = prepare_data(df_test)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        results.append({
            'Fold': fold_idx + 1,
            'Subject': os.path.basename(test_file).replace('.csv', ''),
            'Test_MSE': mse
        })
        
        print(f"Fold {fold_idx+1}/15: MSE = {mse:.6f}")
    
    df = pd.DataFrame(results)
    out_dir = os.path.join(base_dir, 'results', 'ridge_model')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'loso_results.csv'), index=False)
    
    print(f"\nMean MSE: {df['Test_MSE'].mean():.6f} ± {df['Test_MSE'].std():.6f}")
    return df

if __name__ == "__main__":
    train_ridge_loso()
