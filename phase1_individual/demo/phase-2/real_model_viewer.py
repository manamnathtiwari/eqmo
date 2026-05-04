"""
Real UDE Model Viewer - Uses Your Trained WESAD Models
Same interface as demo_app.py but with REAL trained models
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Set page config
st.set_page_config(page_title="Real UDE Model Viewer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (same as demo_app)
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .sub-header {font-size: 1.5rem; font-weight: bold; color: #2ca02c;}
    .success-box {padding: 1rem; background-color: #d4edda; border-radius: 0.5rem; border-left: 5px solid #28a745;}
    .info-box {padding: 1rem; background-color: #d1ecf1; border-radius: 0.5rem; border-left: 5px solid #17a2b8;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'equation_extracted' not in st.session_state:
    st.session_state.equation_extracted = False

# UDE Model Definition
class UDEModel(nn.Module):
    def __init__(self, n_features=18):
        super().__init__()
        self.n_features = n_features
        self.linear_coeffs = nn.Parameter(torch.randn(n_features) * 0.1)
        self.nn = nn.Sequential(
            nn.Linear(n_features + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, features, t):
        linear = torch.sum(self.linear_coeffs * features, dim=-1, keepdim=True)
        state = torch.cat([features, t.expand_as(features[:, :1])], dim=-1)
        nonlinear = self.nn(state)
        return (linear + nonlinear).squeeze(-1)

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
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'normalized')
RESULTS_FILE = os.path.join(MODELS_DIR, 'loso_results.csv')

# Symbolic Regression
def symbolic_regression(coeffs, var_names):
    """Extract symbolic equation from learned coefficients"""
    symbols = [sp.Symbol(name) for name in var_names]
    equation = sum(float(coeffs[i]) * symbols[i] for i in range(len(symbols)))
    equation = sp.simplify(equation)
    return equation, symbols

# Main App
st.markdown('<p class="main-header">🔬 Real UDE Model Viewer</p>', unsafe_allow_html=True)
st.markdown("**View your trained WESAD stress models - Same interface as demo, but with REAL results!**")

# Sidebar - Configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Step 1: Model Selection
st.sidebar.markdown("### 1️⃣ Select Trained Model")

# Check if models exist
if not os.path.exists(MODELS_DIR):
    st.sidebar.error(f"❌ Models not found")
    st.error(f"Models directory not found: {MODELS_DIR}")
    st.stop()

# Get available models
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
model_files.sort()

if not model_files:
    st.sidebar.error("❌ No models found")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Choose Subject Model",
    model_files,
    format_func=lambda x: x.replace('ude_fold_', 'Subject ').replace('.pth', '')
)

fold_num = int(selected_model.replace('ude_fold_', '').replace('.pth', ''))

# Get subject info
subject_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
subject_files.sort()

if fold_num <= len(subject_files):
    subject_file = subject_files[fold_num - 1]
    subject_name = subject_file.replace('.csv', '')
else:
    subject_name = f"Subject {fold_num}"
    subject_file = None

st.sidebar.info(f"**Subject:** {subject_name}")

# Step 2: Data Options
st.sidebar.markdown("### 2️⃣ Data Options")
show_samples = st.sidebar.slider("Samples to Display", 100, 2000, 1000, 100)

# Main Content (same tabs as demo_app)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Load Model & Data", "🤖 Model Analysis", "🔍 Equation Discovery", "📈 Results"])

# TAB 1: Load Model & Data
with tab1:
    st.markdown('<p class="sub-header">Step 1: Load Trained Model & Real Data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Model Information")
        st.code(f"""
Model File: {selected_model}
Subject: {subject_name}
Fold: {fold_num}/15
Features: 18 physiological signals
Architecture: Linear (18 coeffs) + Neural Network (64→32→1)
        """, language="text")
        
        st.markdown("#### WESAD Features")
        st.write("- **HRV:** rmssd, sdnn, pnn50, lf_hf")
        st.write("- **Heart Rate:** mean, std")
        st.write("- **EDA:** mean, std, peaks")
        st.write("- **Temperature:** mean, std")
        st.write("- **Respiration:** mean, std")
        st.write("- **Activity:** mean, std")
        st.write("- **EMG:** mean, std")
        st.write("- **Workload:** cognitive load")
    
    with col2:
        st.markdown("#### Training Info")
        
        # Load results if available
        if os.path.exists(RESULTS_FILE):
            results_df = pd.read_csv(RESULTS_FILE)
            fold_results = results_df[results_df['Fold'] == fold_num]
            
            if not fold_results.empty:
                test_mse = fold_results['Test_MSE'].values[0]
                st.metric("Test MSE", f"{test_mse:.6f}")
                
                mean_mse = results_df['Test_MSE'].mean()
                st.metric("Average MSE (All)", f"{mean_mse:.6f}")
                
                rank = (results_df['Test_MSE'] <= test_mse).sum()
                st.metric("Rank", f"{rank}/15")
    
    if st.button("📥 Load Model & Data", type="primary", use_container_width=True):
        with st.spinner("Loading trained model and real data..."):
            # Load model
            model_path = os.path.join(MODELS_DIR, selected_model)
            model = UDEModel(n_features=18)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            st.session_state.model = model
            st.session_state.model_loaded = True
            
            # Load data
            if subject_file:
                data_path = os.path.join(DATA_DIR, subject_file)
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)
                    
                    # Get available features
                    available_features = [f for f in FEATURE_NAMES if f in df.columns]
                    
                    # Prepare features
                    X = df[available_features].values
                    
                    # Pad if needed
                    if X.shape[1] < 18:
                        X = np.pad(X, ((0, 0), (0, 18 - X.shape[1])), mode='constant')
                    
                    st.session_state.df = df
                    st.session_state.X = X
                    st.session_state.available_features = available_features
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Loaded model and {len(df)} data samples!")
                else:
                    st.error(f"❌ Data file not found: {data_path}")
            else:
                st.warning("⚠️ Subject data file not found")
    
    # Display data if loaded
    if st.session_state.data_loaded:
        st.markdown("#### Real WESAD Data Preview")
        
        df = st.session_state.df
        
        # Plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Stress Over Time", "Top Features Over Time"),
            vertical_spacing=0.15
        )
        
        # Stress (if available)
        if 'stress' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index[:show_samples], y=df['stress'][:show_samples], 
                          mode='lines', name='Stress', line=dict(color='red', width=2)),
                row=1, col=1
            )
        
        # Top 3 features
        coeffs = st.session_state.model.linear_coeffs.detach().numpy()
        top_indices = np.argsort(np.abs(coeffs))[-3:]
        
        colors = ['blue', 'green', 'orange']
        for i, idx in enumerate(top_indices):
            if idx < len(st.session_state.available_features):
                feat = st.session_state.available_features[idx]
                if feat in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index[:show_samples], y=df[feat][:show_samples],
                                  mode='lines', name=feat, line=dict(color=colors[i])),
                        row=2, col=1
                    )
        
        fig.update_xaxes(title_text="Sample Index", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_layout(height=600, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Data Table"):
            st.dataframe(df.head(100), use_container_width=True)
            st.write(f"Total samples: {len(df)}")

# TAB 2: Model Analysis
with tab2:
    st.markdown('<p class="sub-header">Step 2: Analyze Trained Model</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load model first (Tab 1)")
    else:
        model = st.session_state.model
        
        st.markdown("#### Model Performance")
        
        if st.session_state.data_loaded:
            df = st.session_state.df
            X = st.session_state.X
            
            # Make predictions
            X_tensor = torch.FloatTensor(X)
            t_tensor = torch.zeros(len(X), 1)
            
            with torch.no_grad():
                predictions = model(X_tensor, t_tensor).numpy()
            
            st.session_state.predictions = predictions
            
            # Calculate metrics if true stress available
            if 'stress' in df.columns:
                y_true = df['stress'].values
                
                mse = np.mean((predictions - y_true) ** 2)
                mae = np.mean(np.abs(predictions - y_true))
                r2 = 1 - (np.sum((y_true - predictions) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:.6f}")
                col2.metric("MAE", f"{mae:.6f}")
                col3.metric("R² Score", f"{r2:.4f}")
                
                # Predictions vs True
                st.markdown("#### Predictions vs True Stress")
                
                fig = go.Figure()
                
                sample_indices = np.linspace(0, len(df)-1, show_samples, dtype=int)
                
                fig.add_trace(go.Scatter(
                    x=sample_indices,
                    y=y_true[sample_indices],
                    mode='lines',
                    name='True Stress',
                    line=dict(color='green', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=sample_indices,
                    y=predictions[sample_indices],
                    mode='lines',
                    name='Predicted Stress',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    xaxis_title="Sample Index",
                    yaxis_title="Stress Level",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ True stress values not available in data")
                
                # Just show predictions
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=predictions[:show_samples],
                    mode='lines',
                    name='Predicted Stress',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    xaxis_title="Sample Index",
                    yaxis_title="Predicted Stress",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load data in Tab 1 to see predictions")

# TAB 3: Equation Discovery
with tab3:
    st.markdown('<p class="sub-header">Step 3: Extract Learned Equation</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load model first (Tab 1)")
    else:
        st.markdown("#### Learned Coefficients")
        
        model = st.session_state.model
        learned_coeffs = model.linear_coeffs.detach().numpy()
        
        # Comparison table
        comparison_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Coefficient': learned_coeffs,
            'Abs_Coefficient': np.abs(learned_coeffs)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        st.dataframe(comparison_df.style.format({
            'Coefficient': '{:.6f}',
            'Abs_Coefficient': '{:.6f}'
        }), use_container_width=True)
        
        # Bar chart
        fig = go.Figure()
        
        top_10 = comparison_df.head(10)
        colors = ['green' if c > 0 else 'red' for c in top_10['Coefficient']]
        
        fig.add_trace(go.Bar(
            x=top_10['Feature'],
            y=top_10['Coefficient'],
            marker_color=colors,
            text=[f"{c:.4f}" for c in top_10['Coefficient']],
            textposition='outside'
        ))
        
        fig.update_layout(
            xaxis=dict(tickangle=-45),
            yaxis_title="Coefficient Value",
            title="Top 10 Feature Coefficients",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔍 Extract Symbolic Equation", type="primary", use_container_width=True):
            # Symbolic regression
            equation, symbols = symbolic_regression(learned_coeffs, FEATURE_NAMES)
            
            st.session_state.symbolic_equation = equation
            st.session_state.equation_extracted = True
            
            st.markdown("#### Discovered Stress Equation")
            
            # Display in nice format
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.latex(f"\\frac{{dS}}{{dt}} = {sp.latex(equation)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Python format
            st.markdown("#### Python Format")
            st.code(f"dS_dt = {equation}", language="python")
            
            # Simplified version (top 5 features)
            top_5 = comparison_df.head(5)
            simplified_eq = " + ".join([
                f"{row['Coefficient']:.4f}·{row['Feature']}" 
                for _, row in top_5.iterrows()
            ])
            
            st.markdown("#### Simplified (Top 5 Features)")
            st.code(f"dS/dt ≈ {simplified_eq} + ...", language="python")
            
            # Interpretation
            st.markdown("#### 💡 Interpretation")
            
            top_positive = comparison_df[comparison_df['Coefficient'] > 0].head(3)
            top_negative = comparison_df[comparison_df['Coefficient'] < 0].head(3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Stress Increasing Factors:**")
                for _, row in top_positive.iterrows():
                    st.write(f"- **{row['Feature']}**: +{row['Coefficient']:.4f}")
            
            with col2:
                st.markdown("**Stress Decreasing Factors:**")
                for _, row in top_negative.iterrows():
                    st.write(f"- **{row['Feature']}**: {row['Coefficient']:.4f}")

# TAB 4: Results
with tab4:
    st.markdown('<p class="sub-header">Step 4: Complete Results</p>', unsafe_allow_html=True)
    
    if not st.session_state.equation_extracted:
        st.warning("⚠️ Please extract equation first (Tab 3)")
    else:
        # Summary
        st.markdown("### 📊 Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Model Info**")
            st.write(f"- Subject: {subject_name}")
            st.write(f"- Features: 18")
            st.write(f"- Samples: {len(st.session_state.df) if st.session_state.data_loaded else 'N/A'}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Performance**")
            
            if st.session_state.data_loaded and 'stress' in st.session_state.df.columns:
                df = st.session_state.df
                y_true = df['stress'].values
                predictions = st.session_state.predictions
                
                mse = np.mean((predictions - y_true) ** 2)
                r2 = 1 - (np.sum((y_true - predictions) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
                
                st.write(f"- MSE: {mse:.6f}")
                st.write(f"- R²: {r2:.4f}")
            else:
                st.write("- Performance metrics N/A")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Top Feature**")
            
            comparison_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Coefficient': st.session_state.model.linear_coeffs.detach().numpy()
            }).sort_values('Coefficient', key=lambda x: abs(x), ascending=False)
            
            top_feat = comparison_df.iloc[0]
            st.write(f"- {top_feat['Feature']}")
            st.write(f"- Coeff: {top_feat['Coefficient']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Discovered Equation
        st.markdown("### 🎯 Discovered Stress Equation")
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.latex(f"\\frac{{dS}}{{dt}} = {sp.latex(st.session_state.symbolic_equation)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # All subjects comparison
        if os.path.exists(RESULTS_FILE):
            st.markdown("### 📊 All Subjects Comparison")
            
            results_df = pd.read_csv(RESULTS_FILE)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=results_df['Subject'],
                y=results_df['Test_MSE'],
                marker_color='blue',
                text=[f"{mse:.6f}" for mse in results_df['Test_MSE']],
                textposition='outside'
            ))
            
            # Highlight current subject
            current_idx = fold_num - 1
            if current_idx < len(results_df):
                fig.add_trace(go.Bar(
                    x=[results_df['Subject'].iloc[current_idx]],
                    y=[results_df['Test_MSE'].iloc[current_idx]],
                    marker_color='red',
                    name='Current Subject',
                    showlegend=True
                ))
            
            fig.update_layout(
                xaxis_title="Subject",
                yaxis_title="Test MSE",
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Real UDE Model Viewer** | Using YOUR trained WESAD models | Date: 27-12-2024")
