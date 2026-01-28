import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torchdiffeq import odeint
import os
import sys

sys.path.append(os.path.dirname(__file__))
from coupled_ude import CoupledUDE

class SparseCoupledUDEModel:
    """
    Complete Sparse Coupled-UDE Pipeline
    
    Stage 1: Ridge identifies important features (18 → 2-3)
    Stage 2: Coupled UDE models selected features as interdependent system
    
    Result: Interpretable mechanistic equations with feedback loops
    """
    
    def __init__(self, ridge_threshold=0.02, hidden_size=32, device='cpu'):  # Lowered from 0.1 to 0.02
        self.ridge_threshold = ridge_threshold
        self.hidden_size = hidden_size
        self.device = device
        
        # Stage 1: Ridge for feature selection
        self.ridge = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        
        # Stage 2: Coupled UDE
        self.coupled_ude = None
        self.selected_features = None
        self.selected_indices = None
        
        # Feature names (match actual data columns)
        self.all_features = [
            'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50', 'hrv_lf_hf',
            'hr_mean_norm', 'hr_std_norm',
            'eda_mean_norm', 'eda_std_norm', 'eda_peaks_norm',
            'temp_mean_norm', 'temp_std_norm',
            'resp_mean_norm', 'resp_std_norm',
            'activity_mean_norm', 'activity_std_norm',
            'emg_mean_norm', 'emg_std_norm',
            'workload'
        ]
    
    def fit_stage1_ridge(self, X, y):
        """
        Stage 1: Ridge regression for feature selection
        
        X: (n_samples, n_features) - all 18 features
        y: (n_samples,) - stress values
        """
        print("="*70)
        print("STAGE 1: RIDGE FEATURE SELECTION")
        print("="*70)
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Ridge
        self.ridge.fit(X_scaled, y)
        
        # Get coefficients
        coefficients = self.ridge.coef_
        
        # Select important features
        important_mask = np.abs(coefficients) > self.ridge_threshold
        self.selected_indices = np.where(important_mask)[0]
        self.selected_features = [self.all_features[i] for i in self.selected_indices]
        
        # Fallback: If no features selected, pick top 6 by magnitude
        if len(self.selected_features) == 0:
            print(f"\n⚠️  No features above threshold {self.ridge_threshold}")
            print("   Selecting top 6 features by coefficient magnitude...")
            top_indices = np.argsort(np.abs(coefficients))[-6:][::-1]
            self.selected_indices = top_indices
            self.selected_features = [self.all_features[i] for i in top_indices]
        
        print(f"\nRidge Coefficients:")
        for i, (feat, coef) in enumerate(zip(self.all_features, coefficients)):
            marker = "✓" if i in self.selected_indices else " "
            print(f"  [{marker}] {feat:15s}: {coef:+.4f}")
        
        print(f"\nSelected Features ({len(self.selected_features)}/{len(self.all_features)}):")
        for feat in self.selected_features:
            idx = self.all_features.index(feat)
            print(f"  - {feat} (coef: {coefficients[idx]:+.4f})")
        
        # Predictions
        y_pred = self.ridge.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        print(f"\nRidge MSE: {mse:.6f}")
        
        return self.selected_features
    
    def prepare_coupled_data(self, df):
        """
        Prepare data for coupled UDE training
        
        Extracts: Stress, Feature1, Feature2
        Returns: (n_samples, 3) array
        """
        # Get stress
        stress = df['stress'].values
        
        # Get all available feature columns
        available_features = [c for c in df.columns if c not in ['time', 'stress', 'label']]
        
        # Get selected feature values (ensure we have exactly 2)
        feature_data = []
        features_to_use = self.selected_features[:2] if self.selected_features and len(self.selected_features) >= 2 else []
        
        # If we don't have 2 selected features, use first 2 available
        if len(features_to_use) < 2:
            features_to_use = available_features[:2]
            print(f"Warning: Using default features {features_to_use}")
        
        for feat in features_to_use:
            found = False
            # Try exact match first
            if feat in df.columns:
                feature_data.append(df[feat].values)
                found = True
            else:
                # Try partial match (e.g., 'HR_mean' matches 'HR_mean')
                for col in available_features:
                    if feat in col or col in feat:
                        feature_data.append(df[col].values)
                        found = True
                        break
            
            if not found:
                # Use first available feature as fallback
                if available_features:
                    feature_data.append(df[available_features[0]].values)
                    print(f"Warning: {feat} not found, using {available_features[0]}")
                else:
                    feature_data.append(np.zeros_like(stress))
                    print(f"Warning: {feat} not found, using zeros")
        
        # Ensure we have exactly 2 features
        while len(feature_data) < 2:
            feature_data.append(np.zeros_like(stress))
        
        # Stack: [Stress, Feature1, Feature2]
        y_data = np.column_stack([stress] + feature_data[:2])
        
        assert y_data.shape[1] == 3, f"Expected 3 columns, got {y_data.shape[1]}"
        
        return y_data
    
    def fit_stage2_coupled_ude(self, train_files, epochs=50, lr=0.001):
        """
        Stage 2: Train Coupled UDE on selected features
        
        train_files: list of CSV file paths
        """
        print("\n" + "="*70)
        print("STAGE 2: COUPLED UDE TRAINING")
        print("="*70)
        
        if self.selected_features is None:
            raise ValueError("Must run fit_stage1_ridge first")
        
        # Initialize Coupled UDE with top 2 of selected features
        # (Even if 6 are selected, we use the best 2 for the coupled system)
        self.coupled_ude = CoupledUDE(
            selected_features=self.selected_features[:2],  # Use top 2
            hidden_size=self.hidden_size
        ).to(self.device)
        
        optimizer = optim.Adam(self.coupled_ude.parameters(), lr=lr)
        
        # Prepare training data
        print(f"\nLoading {len(train_files)} training files...")
        all_sequences = []
        
        for file_path in train_files:
            df = pd.read_csv(file_path)
            y_seq = self.prepare_coupled_data(df)
            
            # Create sequences (60 timesteps)
            seq_len = 60
            for i in range(0, len(y_seq) - seq_len, seq_len // 2):
                seq = y_seq[i:i+seq_len]
                if len(seq) == seq_len:
                    all_sequences.append(seq)
        
        print(f"Created {len(all_sequences)} training sequences")
        
        # Convert to tensors
        sequences = torch.FloatTensor(np.array(all_sequences)).to(self.device)
        t = torch.linspace(0, 1, 60).to(self.device)
        
        # Training loop
        print(f"\nTraining for {epochs} epochs...")
        for epoch in range(epochs):
            self.coupled_ude.train()
            epoch_loss = 0
            
            # Mini-batch training
            batch_size = min(32, len(sequences))
            indices = torch.randperm(len(sequences))
            
            for i in range(0, len(sequences), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch = sequences[batch_idx]
                
                y0 = batch[:, 0, :]  # Initial state
                y_true = batch
                
                # Solve ODE
                optimizer.zero_grad()
                y_pred = odeint(self.coupled_ude, y0, t, method='euler')
                y_pred = y_pred.permute(1, 0, 2)
                
                # Loss
                loss = torch.mean((y_pred - y_true) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.coupled_ude.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / (len(sequences) / batch_size)
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        
        # Print learned parameters
        print("\n" + "="*70)
        print("LEARNED PARAMETERS")
        print("="*70)
        params = self.coupled_ude.get_interpretable_params()
        
        print("\nRecovery Rates:")
        print(f"  β_stress: {params['beta_stress']:.4f} (half-life: {np.log(2)/params['beta_stress']:.1f} time units)")
        print(f"  β_HRV:    {params['beta_hrv']:.4f}")
        print(f"  β_EDA:    {params['beta_eda']:.4f}")
        
        print("\nCoupling Coefficients:")
        print(f"  HRV → Stress: {params['gamma_hrv_to_stress']:.4f}")
        print(f"  EDA → Stress: {params['gamma_eda_to_stress']:.4f}")
        print(f"  S·HRV interaction: {params['gamma_interaction']:.4f}")
        print(f"  Stress → HRV: {params['gamma_stress_to_hrv']:.4f}")
        print(f"  Stress → EDA: {params['gamma_stress_to_eda']:.4f}")
        
        print("\nFeedback Loop Strengths:")
        print(f"  S ⇄ HRV: {params['feedback_S_HRV']:.4f}")
        print(f"  S ⇄ EDA: {params['feedback_S_EDA']:.4f}")
        
        print("\n" + "="*70)
        print("EQUATIONS")
        print("="*70)
        equations = self.coupled_ude.get_equations_str()
        print("\nStress Equation:", equations['stress'])
        print("\nHRV Equation:", equations['hrv'])
        print("\nEDA Equation:", equations['eda'])
    
    def predict(self, df):
        """Predict stress using coupled system"""
        if self.coupled_ude is None:
            raise ValueError("Must train model first")
        
        self.coupled_ude.eval()
        
        y_data = self.prepare_coupled_data(df)
        y0 = torch.FloatTensor(y_data[0:1]).to(self.device)
        t = torch.linspace(0, len(y_data)-1, len(y_data)).to(self.device)
        
        with torch.no_grad():
            y_pred = odeint(self.coupled_ude, y0, t, method='dopri5')
            stress_pred = y_pred[:, 0, 0].cpu().numpy()
        
        return stress_pred
    
    def save(self, path):
        """Save complete model"""
        torch.save({
            'ridge': self.ridge,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'selected_indices': self.selected_indices,
            'coupled_ude_state': self.coupled_ude.state_dict() if self.coupled_ude else None,
            'all_features': self.all_features
        }, path)
    
    def load(self, path):
        """Load complete model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.ridge = checkpoint['ridge']
        self.scaler = checkpoint['scaler']
        self.selected_features = checkpoint['selected_features']
        self.selected_indices = checkpoint['selected_indices']
        self.all_features = checkpoint['all_features']
        
        if checkpoint['coupled_ude_state']:
            self.coupled_ude = CoupledUDE(
                selected_features=self.selected_features[:2],
                hidden_size=self.hidden_size
            ).to(self.device)
            self.coupled_ude.load_state_dict(checkpoint['coupled_ude_state'])
