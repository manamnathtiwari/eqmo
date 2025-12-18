import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def preprocess_user(input_path, output_path):
    """
    Load raw user data, compute features, and save processed data.
    """
    df = pd.read_csv(input_path)
    
    # Ensure time is sorted
    df = df.sort_values('time').reset_index(drop=True)
    
    # Feature Engineering
    # 1. Rolling statistics for Stress (S) and Workload (W)
    # Window sizes: 30 mins, 60 mins
    for window in [30, 60]:
        df[f'stress_mean_{window}m'] = df['stress'].rolling(window=window, min_periods=1).mean()
        df[f'stress_std_{window}m'] = df['stress'].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'workload_mean_{window}m'] = df['workload'].rolling(window=window, min_periods=1).mean()
    
    # 2. Lagged features (previous time step)
    df['stress_lag_1'] = df['stress'].shift(1).fillna(df['stress'].iloc[0])
    df['workload_lag_1'] = df['workload'].shift(1).fillna(df['workload'].iloc[0])
    
    # 3. Change in stress (target for some models)
    df['stress_change'] = df['stress'].diff().fillna(0)
    
    # 4. Activity Recognition (Matches Sequence Diagram)
    # Classify state based on workload/stress levels
    # 0: Rest, 1: Low Work, 2: High Strain
    conditions = [
        (df['workload'] < 0.2),
        (df['workload'] >= 0.2) & (df['workload'] < 0.7),
        (df['workload'] >= 0.7)
    ]
    choices = [0, 1, 2] # Labels
    df['activity_class'] = np.select(conditions, choices, default=0)
    
    # Save
    df.to_csv(output_path, index=False)

def main():
    input_dir = os.path.join('data', 'synthetic')
    output_dir = os.path.join('data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of user files (excluding metadata.csv)
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and f != 'metadata.csv']
    
    print(f"Preprocessing {len(files)} files...")
    
    for f in tqdm(files):
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)
        preprocess_user(in_path, out_path)
        
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
