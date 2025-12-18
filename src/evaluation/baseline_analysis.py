import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_baseline():
    # Use absolute paths relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # burnout_project root
    data_dir = os.path.join(base_dir, 'data', 'processed')
    results_dir = os.path.join(base_dir, 'results', 'baseline')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load a few users
    users = ['u_wesad_002.csv', 'u_wesad_003.csv', 'u_wesad_004.csv']
    
    for u_file in users:
        path = os.path.join(data_dir, u_file)
        if not os.path.exists(path): continue
        
        df = pd.read_csv(path)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df['time'], df['stress'], label='Stress')
        plt.plot(df['time'], df['workload'], label='Workload', alpha=0.5)
        plt.title(f"Baseline Data: {u_file}")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"plot_{u_file.replace('.csv', '.png')}"))
        plt.close()
        
        # 2. Linear Regression Baseline
        # Predict S(t+1) using S(t) and W(t)
        # Features: stress, workload
        # Target: stress_change (dS) or next stress
        
        X = df[['stress', 'workload']].values[:-1]
        y = df['stress'].values[1:]
        
        # Split train/test
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"User {u_file}: Linear Regression MSE={mse:.6f}, R2={r2:.4f}")
        
        # Plot prediction
        plt.figure(figsize=(10, 5))
        plt.plot(y_test[:100], label='True')
        plt.plot(y_pred[:100], label='Predicted (Linear)')
        plt.title(f"Linear Baseline: {u_file}")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"pred_{u_file.replace('.csv', '.png')}"))
        plt.close()

if __name__ == "__main__":
    run_baseline()
