import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math

# Define the expected multi-modal features
FEATURE_COLUMNS = [
    'workload',  # Index 0 - used in f_known term
    'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50', 'hrv_lf_hf',
    'heart_rate',
    'eda_mean', 'eda_std', 'eda_peaks',
    'resp_mean', 'resp_std', 'resp_rate',
    'temp_mean', 'temp_std',
    'activity_level', 'activity_std',
    'emg_mean', 'emg_std' 
]

class StressDataset(Dataset):
    def __init__(self, csv_path, seq_len=60, feature_columns=None):
        """
        Dataset for multi-modal stress prediction.
        
        Args:
            csv_path: Path to processed CSV file
            seq_len: Sequence length for temporal modeling
            feature_columns: List of feature column names (None = use all available)
        """
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        
        # Determine which features are available
        if feature_columns is None:
            # Auto-detect: use all FEATURE_COLUMNS that exist in the dataframe
            self.feature_columns = [col for col in FEATURE_COLUMNS if col in self.df.columns]
            
            # Fallback: if no features found, use legacy columns
            if len(self.feature_columns) == 0:
                print(f"  WARNING: No multi-modal features found in {csv_path}, using legacy mode (workload only)")
                self.feature_columns = ['workload']
        else:
            self.feature_columns = feature_columns
        
        # Load target (stress)
        self.stress = self.df['stress'].values.astype(np.float32)
        
        # Load all features as a matrix
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        
        # Load time
        self.time = self.df['time'].values.astype(np.float32)
        
        self.num_features = len(self.feature_columns)
        
    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        """
        Returns:
            t: (seq_len,) time points
            y: (seq_len, 1) stress values (target)
            features: (seq_len, num_features) all physiological features
        """
        # Extract sequences
        t = self.time[idx:idx+self.seq_len]
        y = self.stress[idx:idx+self.seq_len]
        features = self.features[idx:idx+self.seq_len, :]  # (seq_len, num_features)
        
        # Normalize time to start at 0
        t_start = t[0]
        t = t - t_start
        
        return {
            't': torch.tensor(t, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32).unsqueeze(-1),  # (seq_len, 1)
            'features': torch.tensor(features, dtype=torch.float32),   # (seq_len, num_features)
            't_start': t_start,
            'num_features': self.num_features
        }

def get_workload_interpolator(u_tensor, t_tensor):
    """
    Returns a function u(t) that interpolates the workload tensor.
    u_tensor: (batch, seq_len, 1) or (seq_len, 1)
    t_tensor: (seq_len,) - time points
    
    Returns: A callable that takes a scalar time t and returns interpolated workload
    """
    def interpolate(t_scalar, current_u):
        """
        Interpolate workload at time t_scalar.
        t_scalar: float - query time
        current_u: tensor (batch, seq_len, 1) - workload values
        """
        t_idx = t_scalar
        idx_low = int(math.floor(t_idx))
        idx_high = idx_low + 1
        
        # Clamp to valid range
        max_idx = current_u.shape[1] - 1
        idx_low = max(0, min(idx_low, max_idx))
        idx_high = max(0, min(idx_high, max_idx))
        
        u_low = current_u[:, idx_low, :]
        u_high = current_u[:, idx_high, :]
        
        # Linear interpolation weight
        weight = t_idx - int(math.floor(t_idx))
        
        u_interp = u_low * (1 - weight) + u_high * weight
        return u_interp
    
    return interpolate
