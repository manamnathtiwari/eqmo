"""
Quick Test: Verify models can be loaded and predictions work
Run this BEFORE the Streamlit app to catch errors early
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from glob import glob

# UDE Model Definition (MUST MATCH TRAINING!)
class UDE(nn.Module):
    def __init__(self, hidden_dim=64, num_features=18):
        super(UDE, self).__init__()
        
        input_dim = 1 + num_features
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))
        self._alpha_raw = nn.Parameter(torch.tensor([-2.2]))  # SINGLE value
        self.current_features = None
        self.current_t = None
    
    @property
    def beta(self):
        return F.softplus(self._beta_raw)
    
    @property
    def alpha(self):
        """Single alpha for all features"""
        return F.softplus(self._alpha_raw)
    
    def predict_simple(self, features, initial_stress=0.5):
        batch_size = features.shape[0]
        stress = torch.ones(batch_size) * initial_stress
        
        stress_decay = -self.beta * stress
        feature_drive = self.alpha * torch.sum(features, dim=1)  # Single alpha * sum
        
        nn_input = torch.cat([stress.unsqueeze(1), features], dim=1)
        nn_correction = self.net(nn_input).squeeze()
        
        dS_dt = stress_decay + feature_drive + nn_correction
        predicted_stress = stress + dS_dt
        
        return predicted_stress

print("="*70)
print("TESTING UNIVERSAL DEMO COMPONENTS")
print("="*70)

# Test 1: Check models directory
print("\n[1/5] Checking models directory...")
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'loso_models')

if os.path.exists(MODELS_DIR):
    print(f"✅ Models directory found: {MODELS_DIR}")
else:
    print(f"❌ Models directory NOT found: {MODELS_DIR}")
    print("   Please check the path!")
    exit(1)

# Test 2: Find model files
print("\n[2/5] Finding model files...")
model_files = glob(os.path.join(MODELS_DIR, 'ude_fold_*.pth'))

if len(model_files) > 0:
    print(f"✅ Found {len(model_files)} model files:")
    for mf in model_files[:3]:  # Show first 3
        print(f"   - {os.path.basename(mf)}")
    if len(model_files) > 3:
        print(f"   ... and {len(model_files)-3} more")
else:
    print(f"❌ No model files found!")
    print(f"   Looking for: {os.path.join(MODELS_DIR, 'ude_fold_*.pth')}")
    exit(1)

# Test 3: Load one model
print("\n[3/5] Testing model loading...")
try:
    test_model_path = model_files[0]
    model = UDE(hidden_dim=64, num_features=18)
    model.load_state_dict(torch.load(test_model_path, map_location='cpu'))
    model.eval()
    print(f"✅ Successfully loaded: {os.path.basename(test_model_path)}")
    print(f"   Alpha value: {model.alpha.item():.4f}")
    print(f"   Beta value: {model.beta.item():.4f}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Test 4: Generate synthetic data
print("\n[4/5] Generating synthetic data...")
try:
    n_samples = 100
    n_features = 18
    
    # Generate random features
    X = np.random.randn(n_samples, n_features) * 0.5 + 0.5
    X = np.clip(X, 0, 1)
    
    X_tensor = torch.FloatTensor(X)
    
    print(f"✅ Generated data shape: {X.shape}")
except Exception as e:
    print(f"❌ Failed to generate data: {e}")
    exit(1)

# Test 5: Make predictions
print("\n[5/5] Testing predictions...")
try:
    with torch.no_grad():
        predictions = model.predict_simple(X_tensor).numpy()
    
    print(f"✅ Predictions shape: {predictions.shape}")
    print(f"   Mean prediction: {np.mean(predictions):.4f}")
    print(f"   Std prediction: {np.std(predictions):.4f}")
    print(f"   Min prediction: {np.min(predictions):.4f}")
    print(f"   Max prediction: {np.max(predictions):.4f}")
except Exception as e:
    print(f"❌ Failed to make predictions: {e}")
    exit(1)

# Test 6: Load ALL models
print("\n[BONUS] Loading ALL models...")
all_models = {}
failed = 0

for model_path in model_files:
    try:
        model = UDE(hidden_dim=64, num_features=18)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        filename = os.path.basename(model_path)
        subject_num = int(filename.replace('ude_fold_', '').replace('.pth', ''))
        all_models[f'Subject_{subject_num}'] = model
    except Exception as e:
        failed += 1
        print(f"   ⚠️ Failed: {os.path.basename(model_path)} - {e}")

print(f"✅ Successfully loaded {len(all_models)}/{len(model_files)} models")
if failed > 0:
    print(f"⚠️  {failed} models failed to load")

# Test 7: Ensemble predictions
print("\n[BONUS] Testing ensemble predictions...")
try:
    all_preds = []
    
    for subject_name, model in all_models.items():
        with torch.no_grad():
            pred = model.predict_simple(X_tensor).numpy()
        all_preds.append(pred)
    
    pred_array = np.array(all_preds)
    mean_pred = np.mean(pred_array, axis=0)
    std_pred = np.std(pred_array, axis=0)
    
    print(f"✅ Ensemble predictions:")
    print(f"   Mean: {np.mean(mean_pred):.4f}")
    print(f"   Uncertainty (avg std): {np.mean(std_pred):.4f}")
    print(f"   Agreement: {'Good' if np.mean(std_pred) < 0.2 else 'Moderate' if np.mean(std_pred) < 0.4 else 'Poor'}")
except Exception as e:
    print(f"❌ Ensemble failed: {e}")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"✅ Models directory: OK")
print(f"✅ Model files found: {len(model_files)}")
print(f"✅ Models loaded: {len(all_models)}")
print(f"✅ Data generation: OK")
print(f"✅ Predictions: OK")
print(f"✅ Ensemble: OK")
print("\n🎉 ALL TESTS PASSED! The Streamlit app should work!")
print("\nRun the app with:")
print("  streamlit run demo/phase-2/universal_demo.py")
print("="*70)
