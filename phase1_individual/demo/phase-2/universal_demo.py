"""
Universal UDE Demo - Generate Data, Get Predictions from ALL Trained Models
No need to select individual models - ALL models predict on YOUR data!
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
import os
import sys
from glob import glob

# Set page config
st.set_page_config(page_title="Universal UDE Demo", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #2ca02c;}
    .success-box {padding: 1rem; background-color: #d4edda; border-radius: 0.5rem; border-left: 5px solid #28a745;}
    .info-box {padding: 1rem; background-color: #d1ecf1; border-radius: 0.5rem; border-left: 5px solid #17a2b8;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False

# UDE Model Definition (MATCHES YOUR ACTUAL TRAINED MODELS!)
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
        
        # Single recovery rate
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))
        
        # SINGLE alpha (not 18!) - this matches your trained models
        self._alpha_raw = nn.Parameter(torch.tensor([-2.2]))
        
        # Placeholders
        self.current_features = None
        self.current_t = None
        self.num_features = num_features
    
    @property
    def beta(self):
        return F.softplus(self._beta_raw)
    
    @property
    def alpha(self):
        """Single alpha for all features"""
        return F.softplus(self._alpha_raw)
    
    def forward(self, t, y):
        batch_size = y.shape[0]
        
        # Get features at current time
        t_idx = torch.argmin(torch.abs(self.current_t - t))
        features = self.current_features[:, t_idx, :]
        
        # Linear dynamics with SINGLE alpha
        stress_decay = -self.beta * y
        feature_drive = self.alpha * torch.sum(features, dim=1)  # Single alpha * sum of features
        
        # Neural network correction
        nn_input = torch.cat([y.unsqueeze(1), features], dim=1)
        nn_correction = self.net(nn_input).squeeze()
        
        return stress_decay + feature_drive + nn_correction
    
    def predict_simple(self, features, initial_stress=0.5):
        """
        Simple prediction without ODE solving - for demo purposes
        Approximates stress based on current features
        """
        batch_size = features.shape[0]
        stress = torch.ones(batch_size) * initial_stress
        
        # Linear dynamics with SINGLE alpha
        stress_decay = -self.beta * stress
        feature_drive = self.alpha * torch.sum(features, dim=1)  # Single alpha * sum of features
        
        # Neural network correction
        nn_input = torch.cat([stress.unsqueeze(1), features], dim=1)
        nn_correction = self.net(nn_input).squeeze()
        
        # Predicted stress change
        dS_dt = stress_decay + feature_drive + nn_correction
        
        # Simple Euler step (assuming dt=1)
        predicted_stress = stress + dS_dt
        
        return predicted_stress

# Feature names
FEATURE_NAMES = [
    'hrv_rmssd', 'hrv_sdnn', 'hrv_pnn50', 'hrv_lf_hf',
    'hr_mean_norm', 'hr_std_norm',
    'eda_mean_norm', 'eda_std_norm', 'eda_peaks_norm',
    'temp_mean_norm', 'temp_std_norm',
    'resp_mean_norm', 'resp_std_norm',
    'activity_mean_norm', 'activity_std_norm',
    'emg_mean_norm', 'emg_std_norm',
    'workload'
]

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
MODELS_DIR = os.path.join(BASE_DIR, 'results', 'loso_models')

# Load ALL models at startup
@st.cache_resource
def load_all_models():
    """Load all trained models at once"""
    models = {}
    
    if not os.path.exists(MODELS_DIR):
        return models
    
    model_files = glob(os.path.join(MODELS_DIR, 'ude_fold_*.pth'))
    
    for model_path in model_files:
        try:
            model = UDE(hidden_dim=64, num_features=18)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            # Extract subject number
            filename = os.path.basename(model_path)
            subject_num = int(filename.replace('ude_fold_', '').replace('.pth', ''))
            
            models[f'Subject_{subject_num}'] = {
                'model': model,
                'alpha': model.alpha.detach().item(),  # Single value
                'beta': model.beta.detach().item()
            }
        except Exception as e:
            st.sidebar.warning(f"Failed to load {model_path}: {e}")
    
    return models

# Main App
st.markdown('<p class="main-header">🔬 Universal UDE Demo</p>', unsafe_allow_html=True)
st.markdown("**Generate ANY data → Get predictions from ALL your trained models!**")

# Load all models
all_models = load_all_models()

if len(all_models) == 0:
    st.error(f"❌ No trained models found in: {MODELS_DIR}")
    st.info("Please train models first or check the path")
    st.stop()

st.sidebar.success(f"✅ Loaded {len(all_models)} trained models!")

# Sidebar - Configuration
st.sidebar.markdown("## ⚙️ Data Configuration")

# Data generation options
st.sidebar.markdown("### 1️⃣ Generate Synthetic Data")

duration_hours = st.sidebar.selectbox("Duration", [6, 12, 24], index=1)
sampling_rate = st.sidebar.slider("Sampling Rate (samples/hour)", 10, 120, 60)
total_samples = duration_hours * sampling_rate

noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.05, 0.01)

# Main Content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Generate Data", 
    "🤖 All Model Predictions", 
    "🔍 Compare Models", 
    "📈 Ensemble Results",
    "🎯 Equation Discovery"
])

# TAB 1: Generate Data
with tab1:
    st.markdown('<p class="sub-header">Step 1: Generate Synthetic Physiological Data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Data Configuration")
        st.write(f"- **Duration:** {duration_hours} hours")
        st.write(f"- **Sampling Rate:** {sampling_rate} samples/hour")
        st.write(f"- **Total Samples:** {total_samples}")
        st.write(f"- **Features:** 18 physiological signals")
        st.write(f"- **Noise Level:** {noise_level}")
    
    with col2:
        st.markdown("#### Features Generated")
        st.write("- HRV metrics (4)")
        st.write("- Heart Rate (2)")
        st.write("- EDA (3)")
        st.write("- Temperature (2)")
        st.write("- Respiration (2)")
        st.write("- Activity (2)")
        st.write("- EMG (2)")
        st.write("- Workload (1)")
    
    if st.button("🎲 Generate Synthetic Data", type="primary", use_container_width=True):
        with st.spinner("Generating realistic physiological data..."):
            # Generate time series
            time_hours = np.linspace(0, duration_hours, total_samples)
            
            # Generate realistic physiological data
            np.random.seed(42)
            data = {'time_hours': time_hours}
            
            # HRV features (vary with circadian rhythm)
            circadian = np.sin(2 * np.pi * time_hours / 24) * 0.3
            data['hrv_rmssd'] = 0.5 + circadian + np.random.randn(total_samples) * 0.1
            data['hrv_sdnn'] = 0.6 + circadian * 0.8 + np.random.randn(total_samples) * 0.1
            data['hrv_pnn50'] = 0.4 + circadian * 0.5 + np.random.randn(total_samples) * 0.08
            data['hrv_lf_hf'] = 0.5 + np.random.randn(total_samples) * 0.15
            
            # Heart rate (inverse of HRV)
            data['hr_mean_norm'] = 0.6 - circadian * 0.4 + np.random.randn(total_samples) * 0.1
            data['hr_std_norm'] = 0.3 + np.random.randn(total_samples) * 0.05
            
            # EDA (stress indicator)
            stress_pattern = np.sin(2 * np.pi * time_hours / 12) * 0.2 + 0.5
            data['eda_mean_norm'] = stress_pattern + np.random.randn(total_samples) * 0.1
            data['eda_std_norm'] = 0.2 + np.random.randn(total_samples) * 0.05
            data['eda_peaks_norm'] = stress_pattern * 0.8 + np.random.randn(total_samples) * 0.08
            
            # Temperature (stable with small variations)
            data['temp_mean_norm'] = 0.5 + np.random.randn(total_samples) * 0.03
            data['temp_std_norm'] = 0.1 + np.random.randn(total_samples) * 0.02
            
            # Respiration (linked to stress)
            data['resp_mean_norm'] = 0.5 + stress_pattern * 0.3 + np.random.randn(total_samples) * 0.08
            data['resp_std_norm'] = 0.3 + np.random.randn(total_samples) * 0.05
            
            # Activity (varies during day)
            activity_pattern = np.abs(np.sin(2 * np.pi * time_hours / 24)) * 0.6
            data['activity_mean_norm'] = activity_pattern + np.random.randn(total_samples) * 0.1
            data['activity_std_norm'] = 0.2 + np.random.randn(total_samples) * 0.05
            
            # EMG (muscle tension)
            data['emg_mean_norm'] = stress_pattern * 0.7 + np.random.randn(total_samples) * 0.1
            data['emg_std_norm'] = 0.25 + np.random.randn(total_samples) * 0.05
            
            # Workload (cognitive load)
            workload_pattern = np.where((time_hours % 24) < 16, 0.7, 0.2)  # High during work hours
            data['workload'] = workload_pattern + np.random.randn(total_samples) * 0.15
            
            # Add noise
            for key in data:
                if key != 'time_hours':
                    data[key] += np.random.randn(total_samples) * noise_level
                    data[key] = np.clip(data[key], 0, 1)  # Keep in [0, 1] range
            
            # Store in session state
            st.session_state.df = pd.DataFrame(data)
            st.session_state.data_generated = True
            
            st.success("✅ Data generated successfully!")
    
    # Display data if generated
    if st.session_state.data_generated:
        st.markdown("#### Generated Data Preview")
        
        df = st.session_state.df
        
        # Plot top features
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("HRV & Heart Rate", "EDA & Stress Indicators", "Activity & Workload"),
            vertical_spacing=0.1
        )
        
        # Row 1: HRV and HR
        fig.add_trace(go.Scatter(x=df['time_hours'], y=df['hrv_rmssd'], mode='lines', name='HRV RMSSD'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['time_hours'], y=df['hr_mean_norm'], mode='lines', name='HR Mean'), row=1, col=1)
        
        # Row 2: EDA
        fig.add_trace(go.Scatter(x=df['time_hours'], y=df['eda_mean_norm'], mode='lines', name='EDA Mean'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['time_hours'], y=df['emg_mean_norm'], mode='lines', name='EMG Mean'), row=2, col=1)
        
        # Row 3: Activity and Workload
        fig.add_trace(go.Scatter(x=df['time_hours'], y=df['activity_mean_norm'], mode='lines', name='Activity'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['time_hours'], y=df['workload'], mode='lines', name='Workload'), row=3, col=1)
        
        fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
        fig.update_layout(height=700, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Data Table"):
            st.dataframe(df.head(100), use_container_width=True)
            st.write(f"Total samples: {len(df)}")

# TAB 2: All Model Predictions
with tab2:
    st.markdown('<p class="sub-header">Step 2: Get Predictions from ALL Trained Models</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first (Tab 1)")
    else:
        st.markdown(f"#### Predicting with {len(all_models)} Trained Models")
        
        if st.button("🚀 Get Predictions from All Models", type="primary", use_container_width=True):
            df = st.session_state.df
            
            # Prepare features
            X = df[FEATURE_NAMES].values
            X_tensor = torch.FloatTensor(X)
            t_tensor = torch.zeros(len(X), 1)
            
            # Get predictions from all models
            all_predictions = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (subject_name, model_data) in enumerate(all_models.items()):
                status_text.text(f"Predicting with {subject_name}...")
                
                model = model_data['model']
                
                with torch.no_grad():
                    predictions = model.predict_simple(X_tensor).numpy()
                
                all_predictions[subject_name] = predictions
                
                progress_bar.progress((i + 1) / len(all_models))
            
            status_text.text("✅ All predictions complete!")
            
            # Store predictions
            st.session_state.all_predictions = all_predictions
            st.session_state.predictions_made = True
            
            # Calculate statistics
            pred_array = np.array(list(all_predictions.values()))
            mean_pred = np.mean(pred_array, axis=0)
            std_pred = np.std(pred_array, axis=0)
            min_pred = np.min(pred_array, axis=0)
            max_pred = np.max(pred_array, axis=0)
            
            st.session_state.ensemble_stats = {
                'mean': mean_pred,
                'std': std_pred,
                'min': min_pred,
                'max': max_pred
            }
            
            st.success(f"✅ Got predictions from all {len(all_models)} models!")
        
        # Display predictions if made
        if st.session_state.predictions_made:
            st.markdown("#### All Model Predictions")
            
            df = st.session_state.df
            all_preds = st.session_state.all_predictions
            
            # Plot all predictions
            fig = go.Figure()
            
            # Plot each model's predictions
            for subject_name, predictions in all_preds.items():
                fig.add_trace(go.Scatter(
                    x=df['time_hours'],
                    y=predictions,
                    mode='lines',
                    name=subject_name,
                    opacity=0.3,
                    line=dict(width=1)
                ))
            
            # Add ensemble mean
            stats = st.session_state.ensemble_stats
            fig.add_trace(go.Scatter(
                x=df['time_hours'],
                y=stats['mean'],
                mode='lines',
                name='Ensemble Mean',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Stress Predictions from All Trained Models",
                xaxis_title="Time (hours)",
                yaxis_title="Predicted Stress",
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Prediction", f"{np.mean(stats['mean']):.4f}")
            col2.metric("Std Across Models", f"{np.mean(stats['std']):.4f}")
            col3.metric("Min Prediction", f"{np.min(stats['min']):.4f}")
            col4.metric("Max Prediction", f"{np.max(stats['max']):.4f}")

# TAB 3: Compare Models
with tab3:
    st.markdown('<p class="sub-header">Step 3: Compare Learned Parameters</p>', unsafe_allow_html=True)
    
    st.markdown("#### Alpha & Beta Comparison Across All Models")
    
    # Create comparison dataframe
    param_comparison = pd.DataFrame({
        'Subject': list(all_models.keys()),
        'Alpha': [all_models[subj]['alpha'] for subj in all_models.keys()],
        'Beta': [all_models[subj]['beta'] for subj in all_models.keys()]
    })
    
    # Display table
    st.dataframe(param_comparison.style.format({
        'Alpha': '{:.6f}',
        'Beta': '{:.6f}'
    }), use_container_width=True)
    
    # Bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Alpha (Feature Sensitivity)")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=param_comparison['Subject'],
            y=param_comparison['Alpha'],
            marker_color='blue',
            text=[f"{a:.4f}" for a in param_comparison['Alpha']],
            textposition='outside'
        ))
        fig.update_layout(
            xaxis_title="Subject",
            yaxis_title="Alpha Value",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Beta (Recovery Rate)")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=param_comparison['Subject'],
            y=param_comparison['Beta'],
            marker_color='green',
            text=[f"{b:.4f}" for b in param_comparison['Beta']],
            textposition='outside'
        ))
        fig.update_layout(
            xaxis_title="Subject",
            yaxis_title="Beta Value",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("#### Summary Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Alpha Statistics:**")
        st.write(f"- Mean: {param_comparison['Alpha'].mean():.6f}")
        st.write(f"- Std: {param_comparison['Alpha'].std():.6f}")
        st.write(f"- Min: {param_comparison['Alpha'].min():.6f}")
        st.write(f"- Max: {param_comparison['Alpha'].max():.6f}")
    
    with col2:
        st.markdown("**Beta Statistics:**")
        st.write(f"- Mean: {param_comparison['Beta'].mean():.6f}")
        st.write(f"- Std: {param_comparison['Beta'].std():.6f}")
        st.write(f"- Min: {param_comparison['Beta'].min():.6f}")
        st.write(f"- Max: {param_comparison['Beta'].max():.6f}")

# TAB 4: Ensemble Results
with tab4:
    st.markdown('<p class="sub-header">Step 4: Ensemble Predictions & Uncertainty</p>', unsafe_allow_html=True)
    
    if not st.session_state.predictions_made:
        st.warning("⚠️ Please get predictions first (Tab 2)")
    else:
        df = st.session_state.df
        stats = st.session_state.ensemble_stats
        
        # Ensemble prediction with uncertainty
        st.markdown("### 📊 Ensemble Prediction with Uncertainty Bands")
        
        fig = go.Figure()
        
        # Uncertainty band (mean ± std)
        fig.add_trace(go.Scatter(
            x=df['time_hours'],
            y=stats['mean'] + stats['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['time_hours'],
            y=stats['mean'] - stats['std'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0,100,200,0.2)',
            fill='tonexty',
            name='±1 Std',
            hoverinfo='skip'
        ))
        
        # Mean prediction
        fig.add_trace(go.Scatter(
            x=df['time_hours'],
            y=stats['mean'],
            mode='lines',
            name='Ensemble Mean',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="Ensemble Stress Prediction with Uncertainty",
            xaxis_title="Time (hours)",
            yaxis_title="Predicted Stress",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary
        st.markdown("### 📋 Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Ensemble Statistics**")
            st.write(f"- Models Used: {len(all_models)}")
            st.write(f"- Mean Stress: {np.mean(stats['mean']):.4f}")
            st.write(f"- Avg Uncertainty: {np.mean(stats['std']):.4f}")
            st.write(f"- Peak Stress: {np.max(stats['mean']):.4f}")
            st.write(f"- Min Stress: {np.min(stats['mean']):.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Interpretation**")
            
            avg_stress = np.mean(stats['mean'])
            avg_uncertainty = np.mean(stats['std'])
            
            if avg_stress > 0.6:
                st.write("🔴 **High Stress Detected**")
            elif avg_stress > 0.4:
                st.write("🟡 **Moderate Stress**")
            else:
                st.write("🟢 **Low Stress**")
            
            if avg_uncertainty > 0.2:
                st.write("⚠️ High model disagreement")
            else:
                st.write("✅ Models agree well")
            
            st.markdown('</div>', unsafe_allow_html=True)

# TAB 5: Equation Discovery
with tab5:
    st.markdown('<p class="sub-header">Step 5: Discover & Verify Equations</p>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Learned Equation from Your Trained Models")
    
    # Show the equation structure
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.markdown("**Universal Differential Equation:**")
    st.latex(r"\frac{dS}{dt} = -\beta \cdot S + \alpha \cdot \sum_{i=1}^{18} F_i + NN(S, F_1, ..., F_{18})")
    st.markdown("Where:")
    st.write("- **S** = Stress level")
    st.write("- **β** = Recovery rate (how fast stress decays)")
    st.write("- **α** = Feature sensitivity (how features affect stress)")
    st.write("- **Fᵢ** = Physiological features (HRV, EDA, etc.)")
    st.write("- **NN** = Neural network (captures nonlinear dynamics)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show learned parameters
    st.markdown("### 📊 Learned Parameters Across All Subjects")
    
    if len(all_models) > 0:
        # Get all alphas and betas
        alphas = [all_models[subj]['alpha'] for subj in all_models.keys()]
        betas = [all_models[subj]['beta'] for subj in all_models.keys()]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Alpha", f"{np.mean(alphas):.4f}")
        col2.metric("Avg Beta", f"{np.mean(betas):.4f}")
        col3.metric("Alpha Range", f"{np.min(alphas):.4f} - {np.max(alphas):.4f}")
        col4.metric("Beta Range", f"{np.min(betas):.4f} - {np.max(betas):.4f}")
        
        # Simplified equation
        avg_alpha = np.mean(alphas)
        avg_beta = np.mean(betas)
        
        st.markdown("### 🔍 Simplified Ensemble Equation")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Using average parameters:**")
        st.latex(f"\\frac{{dS}}{{dt}} = -{avg_beta:.4f} \\cdot S + {avg_alpha:.4f} \\cdot \\sum F_i + NN(S, F)")
        st.markdown("**Interpretation:**")
        st.write(f"- Stress decays at rate **{avg_beta:.4f}** (natural recovery)")
        st.write(f"- Features drive stress with sensitivity **{avg_alpha:.4f}**")
        st.write(f"- Neural network adds **nonlinear corrections**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Verification section
    st.markdown("### ✅ Equation Verification")
    
    if st.session_state.predictions_made:
        st.markdown("#### Compare: True Equation vs Learned Equation")
        
        # User can define a "true" equation for comparison
        st.markdown("**Define a test equation to verify against:**")
        
        col1, col2 = st.columns(2)
        with col1:
            test_beta = st.number_input("Test β (recovery rate)", value=0.05, step=0.01, format="%.4f")
        with col2:
            test_alpha = st.number_input("Test α (sensitivity)", value=0.10, step=0.01, format="%.4f")
        
        if st.button("🔬 Verify Equation Accuracy"):
            # Generate predictions using test equation
            df = st.session_state.df
            X = df[FEATURE_NAMES].values
            
            # Simple prediction with test parameters
            test_predictions = []
            stress = 0.5  # Initial stress
            
            for features in X:
                # Test equation: dS/dt = -beta*S + alpha*sum(features)
                dS_dt = -test_beta * stress + test_alpha * np.sum(features)
                stress = stress + dS_dt  # Euler step
                test_predictions.append(stress)
            
            test_predictions = np.array(test_predictions)
            
            # Compare with ensemble predictions
            ensemble_pred = st.session_state.ensemble_stats['mean']
            
            # Calculate metrics
            mse = np.mean((test_predictions - ensemble_pred) ** 2)
            mae = np.mean(np.abs(test_predictions - ensemble_pred))
            correlation = np.corrcoef(test_predictions, ensemble_pred)[0, 1]
            
            # Display results
            st.markdown("#### 📈 Verification Results")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{mse:.6f}")
            col2.metric("MAE", f"{mae:.6f}")
            col3.metric("Correlation", f"{correlation:.4f}")
            
            # Plot comparison
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['time_hours'],
                y=ensemble_pred,
                mode='lines',
                name='Learned (Ensemble)',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=df['time_hours'],
                y=test_predictions,
                mode='lines',
                name='Test Equation',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Learned vs Test Equation",
                xaxis_title="Time (hours)",
                yaxis_title="Predicted Stress",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            if correlation > 0.9:
                st.success(f"✅ **Excellent Match!** Correlation = {correlation:.4f}")
                st.write("Your test equation closely matches the learned ensemble!")
            elif correlation > 0.7:
                st.info(f"✓ **Good Match** Correlation = {correlation:.4f}")
                st.write("The equations show similar patterns.")
            else:
                st.warning(f"⚠ **Poor Match** Correlation = {correlation:.4f}")
                st.write("Try adjusting β and α to better match the learned equation.")
            
            # Suggestions
            st.markdown("#### 💡 Suggestions")
            st.write(f"- **Learned avg β:** {avg_beta:.4f}")
            st.write(f"- **Learned avg α:** {avg_alpha:.4f}")
            st.write(f"- **Your test β:** {test_beta:.4f}")
            st.write(f"- **Your test α:** {test_alpha:.4f}")
            
            if abs(test_beta - avg_beta) > 0.02:
                st.write(f"→ Try β closer to {avg_beta:.4f}")
            if abs(test_alpha - avg_alpha) > 0.02:
                st.write(f"→ Try α closer to {avg_alpha:.4f}")
    else:
        st.info("⚠️ Generate predictions first (Tab 2) to verify equations")

# Footer
st.markdown("---")
st.markdown(f"**Universal UDE Demo** | Using {len(all_models)} trained models | Date: 27-12-2024")
