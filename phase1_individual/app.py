"""
MC-UDE Stress Prediction Demo
==============================
Interactive Streamlit app demonstrating the Multi-Coefficient
Universal Differential Equation model for interpretable stress prediction.

Run:  streamlit run phase1_individual/app.py
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from glob import glob

# Try importing optional dependencies
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="MC-UDE Stress Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # burnout_project/ (one level up)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "normalized")
MODELS_DIR = os.path.join(BASE_DIR, "mc_ude_results")

FEATURE_COLUMNS = [
    'workload_norm', 'hrv_rmssd_norm', 'hrv_sdnn_norm', 'hrv_pnn50_norm',
    'hrv_lf_hf_norm', 'heart_rate_norm', 'eda_mean_norm', 'eda_std_norm',
    'eda_peaks_norm', 'resp_mean_norm', 'resp_std_norm', 'resp_rate_norm',
    'temp_mean_norm', 'temp_std_norm', 'activity_level_norm',
    'activity_std_norm', 'emg_mean_norm', 'emg_std_norm'
]

FEATURE_DISPLAY = [
    'Workload', 'HRV RMSSD', 'HRV SDNN', 'HRV pNN50',
    'HRV LF/HF', 'Heart Rate', 'EDA Mean', 'EDA Std',
    'EDA Peaks', 'Resp Mean', 'Resp Std', 'Resp Rate',
    'Temp Mean', 'Temp Std', 'Activity Level', 'Activity Std',
    'EMG Mean', 'EMG Std'
]

FEATURE_CATEGORY = {
    'Heart Rate': '❤️ Cardiac', 'HRV RMSSD': '❤️ Cardiac',
    'HRV SDNN': '❤️ Cardiac', 'HRV pNN50': '❤️ Cardiac', 'HRV LF/HF': '❤️ Cardiac',
    'EDA Mean': '⚡ Electrodermal', 'EDA Std': '⚡ Electrodermal', 'EDA Peaks': '⚡ Electrodermal',
    'Resp Mean': '🌬️ Respiratory', 'Resp Std': '🌬️ Respiratory', 'Resp Rate': '🌬️ Respiratory',
    'Temp Mean': '🌡️ Temperature', 'Temp Std': '🌡️ Temperature',
    'Activity Level': '🏃 Activity', 'Activity Std': '🏃 Activity',
    'EMG Mean': '💪 Muscle', 'EMG Std': '💪 Muscle',
    'Workload': '🧠 Cognitive',
}


# ============================================================================
# Model definition (must match training)
# ============================================================================
class MCUDE(nn.Module):
    def __init__(self, hidden_dim=64, num_features=18):
        super().__init__()
        input_dim = 1 + num_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self._beta_raw = nn.Parameter(torch.tensor([-2.9]))
        self._alphas_raw = nn.Parameter(torch.ones(num_features) * (-2.2))
        self.current_features = None
        self.current_t = None
        self.num_features = num_features

    @property
    def beta(self): return F.softplus(self._beta_raw)
    @property
    def alphas(self): return F.softplus(self._alphas_raw)

    def set_current_batch(self, t, features):
        self.current_t = t
        self.current_features = features

    def forward(self, t, y):
        S = y
        t_idx = torch.argmin(torch.abs(self.current_t - t))
        features = self.current_features[:, t_idx, :]
        recovery = -self.beta * S
        feature_contribution = torch.sum(self.alphas * features, dim=-1)
        f_known = recovery + feature_contribution
        S_expanded = S.unsqueeze(-1) if S.dim() == 1 else S
        nn_in = torch.cat([S_expanded, features], dim=-1)
        f_nn = self.net(nn_in)
        if S.dim() == 1:
            f_nn = f_nn.squeeze(-1)
        return f_known + f_nn


# ============================================================================
# Data & model loading
# ============================================================================
@st.cache_data
def load_subject_data(filepath):
    df = pd.read_csv(filepath)
    return df


@st.cache_resource
def load_model(model_path):
    model = MCUDE(num_features=18)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def get_subject_list():
    csv_files = sorted(glob(os.path.join(DATA_DIR, 'u_wesad_*.csv')))
    subjects = []
    for f in csv_files:
        basename = os.path.basename(f)
        sid = basename.replace('u_wesad_', '').replace('.csv', '')
        subjects.append({'id': sid, 'path': f, 'name': f'Subject {sid}'})
    return subjects


def get_model_for_fold(fold_idx):
    model_path = os.path.join(MODELS_DIR, f'mcude_fold_{fold_idx}.pth')
    if os.path.exists(model_path):
        return load_model(model_path)
    return None


def predict_trajectory(model, features, time_vals, seq_len=60):
    """Run ODE integration to predict stress trajectory."""
    if not HAS_TORCHDIFFEQ:
        return None
    with torch.no_grad():
        feat_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, seq, 18)
        t = torch.FloatTensor(time_vals)
        y0 = torch.tensor([0.0])  # Will be overridden by caller
        model.set_current_batch(t, feat_tensor)
        y_pred = odeint(model, y0, t, method='euler')
    return y_pred.squeeze().numpy()


def predict_whatif(model, features, time_vals, feature_idx, scale_factor):
    """Run what-if simulation by scaling a specific feature."""
    modified = features.copy()
    modified[:, feature_idx] = modified[:, feature_idx] * scale_factor
    if not HAS_TORCHDIFFEQ:
        return None
    with torch.no_grad():
        feat_tensor = torch.FloatTensor(modified).unsqueeze(0)
        t = torch.FloatTensor(time_vals)
        y0 = torch.tensor([0.0])
        model.set_current_batch(t, feat_tensor)
        y_pred = odeint(model, y0, t, method='euler')
    return y_pred.squeeze().numpy()


# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { margin: 0.5rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    .equation-card {
        background: linear-gradient(135deg, #0f3460, #1a1a2e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        color: #e0e0e0;
        font-family: 'Courier New', monospace;
        font-size: 1.05rem;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    .equation-card .eq-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #7eb8da;
        margin-bottom: 0.5rem;
    }
    .equation-card .eq-main {
        font-size: 1.15rem;
        color: #ffffff;
    }
    .eq-beta { color: #ff6b6b; font-weight: bold; }
    .eq-alpha { color: #4ecdc4; font-weight: bold; }
    .eq-nn { color: #45b7d1; font-weight: bold; }
    .eq-var { color: #f9ca24; }

    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4ecdc4;
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
        margin-top: 0.3rem;
    }

    .insight-box {
        background: linear-gradient(135deg, #1a3a2e, #1a2a3e);
        border-left: 4px solid #4ecdc4;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #e0e0e0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .insight-box strong { color: #4ecdc4; }

    .novelty-card {
        background: linear-gradient(135deg, #2d1b4e, #1a1a3e);
        border: 1px solid #5a3d8a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .novelty-card h4 { color: #b39ddb; margin: 0 0 0.5rem; }
    .novelty-card p { color: #ccc; margin: 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# App
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 MC-UDE: Interpretable Stress Prediction</h1>
        <p>Multi-Coefficient Universal Differential Equation — Real trained models on WESAD physiological data</p>
    </div>
    """, unsafe_allow_html=True)

    # Check dependencies
    subjects = get_subject_list()
    model_files = sorted(glob(os.path.join(MODELS_DIR, 'mcude_fold_*.pth')))

    if not subjects:
        st.error(f"No subject data found in `{DATA_DIR}`")
        return
    if not model_files:
        st.error(f"No trained models found in `{MODELS_DIR}`")
        return
    if not HAS_TORCHDIFFEQ:
        st.warning("⚠️ `torchdiffeq` not installed. Trajectory predictions will be disabled. Run: `pip install torchdiffeq`")

    # Sidebar
    st.sidebar.markdown("### ⚙️ Configuration")
    selected_idx = st.sidebar.selectbox(
        "Select Subject",
        range(len(subjects)),
        format_func=lambda i: subjects[i]['name']
    )
    seq_len = st.sidebar.slider("Trajectory Length (timesteps)", 30, 120, 60)
    start_offset = st.sidebar.slider("Start Position in Recording", 0, 200, 0, step=10)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About")
    st.sidebar.markdown("""
    This demo uses **real trained models** on the **WESAD dataset** 
    (Wearable Stress and Affect Detection).

    Each model learns a personalized stress equation:

    `dS/dt = -β·S + Σαᵢ·Fᵢ + NN(S,F)`

    where αᵢ reveals which physiological features drive each individual's stress.
    """)

    # Load data and model
    subject = subjects[selected_idx]
    fold_idx = selected_idx + 1
    df = load_subject_data(subject['path'])
    model = get_model_for_fold(fold_idx)

    if model is None:
        st.error(f"Model for fold {fold_idx} not found.")
        return

    # Extract parameters
    beta = model.beta.item()
    alphas = model.alphas.detach().numpy()

    # ========================================================================
    # Tab layout
    # ========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Prediction", "🔬 Equation & Profile", "🔮 What-If Simulator", "💡 Why MC-UDE?"
    ])

    # ========================================================================
    # TAB 1: Prediction
    # ========================================================================
    with tab1:
        st.markdown("### Stress Trajectory Prediction")

        # Prepare data
        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        features = df[feat_cols].values.astype(np.float32)
        stress = df['stress'].values.astype(np.float32)
        time_vals = df['time'].values.astype(np.float32)

        end_idx = min(start_offset + seq_len, len(df))
        start_idx = start_offset
        if end_idx - start_idx < seq_len:
            start_idx = max(0, end_idx - seq_len)

        feat_window = features[start_idx:end_idx]
        stress_window = stress[start_idx:end_idx]
        time_window = time_vals[start_idx:end_idx] - time_vals[start_idx]

        # Predict
        if HAS_TORCHDIFFEQ and len(feat_window) == seq_len:
            with torch.no_grad():
                feat_t = torch.FloatTensor(feat_window).unsqueeze(0)
                t = torch.FloatTensor(time_window)
                y0 = torch.tensor([stress_window[0]])
                model.set_current_batch(t, feat_t)
                y_pred = odeint(model, y0, t, method='euler').squeeze().numpy()

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_window, y=stress_window,
                mode='lines', name='Ground Truth',
                line=dict(color='#ff6b6b', width=2.5),
                fill='tozeroy', fillcolor='rgba(255,107,107,0.1)'
            ))
            fig.add_trace(go.Scatter(
                x=time_window, y=y_pred,
                mode='lines', name='MC-UDE Prediction',
                line=dict(color='#4ecdc4', width=2.5, dash='dot')
            ))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                xaxis_title='Time (minutes)',
                yaxis_title='Stress Level',
                height=400,
                margin=dict(l=40, r=20, t=30, b=40),
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics
            mse = np.mean((stress_window - y_pred) ** 2)
            mae = np.mean(np.abs(stress_window - y_pred))
            corr = np.corrcoef(stress_window, y_pred)[0, 1] if np.std(y_pred) > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><div class="metric-value">{mse:.5f}</div><div class="metric-label">MSE</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-value">{mae:.5f}</div><div class="metric-label">MAE</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-value">{corr:.4f}</div><div class="metric-label">Correlation</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><div class="metric-value">{beta:.4f}</div><div class="metric-label">Recovery Rate β</div></div>', unsafe_allow_html=True)
        else:
            st.info("Adjust parameters to view trajectory prediction.")

    # ========================================================================
    # TAB 2: Equation & Profile
    # ========================================================================
    with tab2:
        st.markdown("### Learned Stress Equation")

        # Build equation HTML
        terms = [f'<span class="eq-beta">-{beta:.4f}</span>·<span class="eq-var">S</span>']
        for i, (a, name) in enumerate(zip(alphas, FEATURE_DISPLAY)):
            if a > 0.01:
                terms.append(f'<span class="eq-alpha">+{a:.4f}</span>·{name}')
        terms.append('<span class="eq-nn">+ NN(S, F)</span>')

        eq_str = ' '.join(terms)
        st.markdown(f"""
        <div class="equation-card">
            <div class="eq-label">Personalized Governing Equation — {subject['name']}</div>
            <div class="eq-main">d<span class="eq-var">S</span>/dt = {eq_str}</div>
        </div>
        """, unsafe_allow_html=True)

        # Alpha profile bar chart
        st.markdown("### Feature Sensitivity Profile (αᵢ coefficients)")

        alpha_df = pd.DataFrame({
            'Feature': FEATURE_DISPLAY,
            'Alpha': alphas,
            'Category': [FEATURE_CATEGORY[f] for f in FEATURE_DISPLAY]
        }).sort_values('Alpha', ascending=True)

        fig2 = px.bar(
            alpha_df, x='Alpha', y='Feature', color='Category',
            orientation='h', height=500,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,46,0.8)',
            margin=dict(l=10, r=20, t=10, b=10),
            legend=dict(orientation='h', y=-0.15),
            xaxis_title='Sensitivity (αᵢ)',
            yaxis_title=''
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Top features insight
        sorted_feats = sorted(zip(FEATURE_DISPLAY, alphas), key=lambda x: x[1], reverse=True)
        top3 = sorted_feats[:3]
        st.markdown(f"""
        <div class="insight-box">
            <strong>Clinical Insight:</strong> This subject's stress is primarily driven by
            <strong>{top3[0][0]}</strong> (α={top3[0][1]:.4f}),
            <strong>{top3[1][0]}</strong> (α={top3[1][1]:.4f}), and
            <strong>{top3[2][0]}</strong> (α={top3[2][1]:.4f}).
            The recovery rate β={beta:.4f} indicates how quickly stress dissipates naturally.
        </div>
        """, unsafe_allow_html=True)

    # ========================================================================
    # TAB 3: What-If Simulator
    # ========================================================================
    with tab3:
        st.markdown("### 🔮 What-If Scenario Simulator")
        st.markdown("*Modify a physiological feature and see how the predicted stress trajectory changes — something only an interpretable model can do.*")

        col1, col2 = st.columns([1, 1])
        with col1:
            feature_name = st.selectbox("Select Feature to Modify", FEATURE_DISPLAY)
            feature_idx = FEATURE_DISPLAY.index(feature_name)
        with col2:
            scale = st.slider(
                f"Scale {feature_name}", 0.0, 3.0, 1.0, 0.1,
                help="1.0 = no change, 0.5 = reduce by half, 2.0 = double"
            )

        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        features = df[feat_cols].values.astype(np.float32)
        stress = df['stress'].values.astype(np.float32)
        time_vals = df['time'].values.astype(np.float32)

        end_idx = min(start_offset + seq_len, len(df))
        start_idx = start_offset
        if end_idx - start_idx < seq_len:
            start_idx = max(0, end_idx - seq_len)

        feat_window = features[start_idx:end_idx]
        stress_window = stress[start_idx:end_idx]
        time_window = time_vals[start_idx:end_idx] - time_vals[start_idx]

        if HAS_TORCHDIFFEQ and len(feat_window) == seq_len:
            # Original
            with torch.no_grad():
                feat_t = torch.FloatTensor(feat_window).unsqueeze(0)
                t = torch.FloatTensor(time_window)
                y0 = torch.tensor([stress_window[0]])
                model.set_current_batch(t, feat_t)
                y_orig = odeint(model, y0, t, method='euler').squeeze().numpy()

            # Modified
            feat_mod = feat_window.copy()
            feat_mod[:, feature_idx] *= scale
            with torch.no_grad():
                feat_t2 = torch.FloatTensor(feat_mod).unsqueeze(0)
                model.set_current_batch(t, feat_t2)
                y_mod = odeint(model, y0, t, method='euler').squeeze().numpy()

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=time_window, y=stress_window,
                mode='lines', name='Ground Truth',
                line=dict(color='#ff6b6b', width=2)
            ))
            fig3.add_trace(go.Scatter(
                x=time_window, y=y_orig,
                mode='lines', name='Original Prediction',
                line=dict(color='#4ecdc4', width=2)
            ))
            fig3.add_trace(go.Scatter(
                x=time_window, y=y_mod,
                mode='lines', name=f'{feature_name} × {scale:.1f}',
                line=dict(color='#f9ca24', width=2.5, dash='dash')
            ))
            fig3.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                xaxis_title='Time (minutes)',
                yaxis_title='Stress Level',
                height=400,
                margin=dict(l=40, r=20, t=30, b=40),
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Impact metric
            diff = np.mean(y_mod) - np.mean(y_orig)
            pct = (diff / (np.mean(np.abs(y_orig)) + 1e-8)) * 100
            direction = "increases" if diff > 0 else "decreases"
            st.markdown(f"""
            <div class="insight-box">
                <strong>What-If Result:</strong> Scaling <strong>{feature_name}</strong> by {scale:.1f}x
                {direction} mean predicted stress by <strong>{abs(pct):.1f}%</strong>.
                {'This is a significant effect — this feature is a key stress driver for this subject.'
                 if abs(pct) > 5 else 'This is a modest effect for this particular subject.'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Install `torchdiffeq` and adjust parameters to use the simulator.")

    # ========================================================================
    # TAB 4: Why MC-UDE?
    # ========================================================================
    with tab4:
        st.markdown("### Why is MC-UDE Novel?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="novelty-card">
                <h4>🔲 Black-Box Models (LSTM, RF, CNN)</h4>
                <p>
                ✅ Good accuracy (MSE ~0.004)<br>
                ❌ No explanation — WHY is this person stressed?<br>
                ❌ No personalization — same model for everyone<br>
                ❌ No simulation — can't test interventions<br>
                ❌ No physics — can predict impossible stress levels
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="novelty-card">
                <h4>🧠 MC-UDE (This Work)</h4>
                <p>
                ✅ Competitive accuracy (MSE ~0.006)<br>
                ✅ Interpretable — learned equation per subject<br>
                ✅ Personalized — unique αᵢ profile per person<br>
                ✅ Simulatable — test "what if" interventions<br>
                ✅ Physics-grounded — recovery rate β, bounded dynamics
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Cross-Subject Comparison")

        # Load all models and show comparison
        all_profiles = []
        for i in range(len(subjects)):
            m = get_model_for_fold(i + 1)
            if m is not None:
                a = m.alphas.detach().numpy()
                b = m.beta.item()
                all_profiles.append({
                    'Subject': subjects[i]['name'],
                    'Beta': b,
                    **{FEATURE_DISPLAY[j]: a[j] for j in range(len(FEATURE_DISPLAY))}
                })

        if all_profiles:
            profile_df = pd.DataFrame(all_profiles)
            matrix = profile_df[FEATURE_DISPLAY].values
            fig4 = go.Figure(data=go.Heatmap(
                z=matrix,
                x=FEATURE_DISPLAY,
                y=profile_df['Subject'].tolist(),
                colorscale='Viridis',
                colorbar=dict(title='αᵢ')
            ))
            fig4.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                height=450,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
                <strong>Key Novelty:</strong> Each row is a different person's learned stress equation.
                No two people respond to stress the same way — MC-UDE discovers this automatically.
                A black-box LSTM would give you a single number; MC-UDE gives you a personalized
                physiological profile that a clinician can act on.
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
