# -*- coding: utf-8 -*-
"""
StressLens - MC-UDE Interactive Demo
=====================================
Run: streamlit run stresslens_app.py
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from glob import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from torchdiffeq import odeint
    HAS_ODE = True
except ImportError:
    HAS_ODE = False

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data", "processed", "normalized")
P1_DIR   = os.path.join(BASE, "phase1_individual", "mc_ude_results")
P2_DIR   = os.path.join(BASE, "phase2_cohort", "cohort_results", "cohort_ude_results")
EVAL_CSV = os.path.join(BASE, "phase2_cohort", "cohort_results", "cohort_evaluation", "cohort_vs_individual.csv")
META_JSON= os.path.join(BASE, "phase2_cohort", "cluster_results", "cohort_metadata.json")
FIG_DIR  = os.path.join(BASE, "phase2_cohort", "analysis")
CLUST_DIR= os.path.join(BASE, "phase2_cohort", "cluster_results")
EVAL_PNG = os.path.join(BASE, "phase2_cohort", "cohort_results", "cohort_evaluation", "cohort_comparison.png")

# Subject → file number → fold mapping
SUBJECTS = ['S002','S003','S004','S005','S006','S007','S008',
            'S009','S010','S011','S013','S014','S015','S016','S017']
SUBJ_FILE = {s: s.replace('S','0') for s in SUBJECTS}   # S002 → 002

# Cohort 0 = Cardiac Responder (14 subjects), Cohort 1 = Cognitive Responder (S003)
COHORT_MAP = {s: '0' for s in SUBJECTS}
COHORT_MAP['S003'] = '1'
COHORT_NAMES = {'0': 'Cardiac Responder', '1': 'Cognitive Responder'}

FEATURE_COLS = [
    'workload_norm','hrv_rmssd_norm','hrv_sdnn_norm','hrv_pnn50_norm',
    'hrv_lf_hf_norm','heart_rate_norm','eda_mean_norm','eda_std_norm',
    'eda_peaks_norm','resp_mean_norm','resp_std_norm','resp_rate_norm',
    'temp_mean_norm','temp_std_norm','activity_level_norm',
    'activity_std_norm','emg_mean_norm','emg_std_norm'
]
FEAT_NAMES = [
    'Workload','HRV RMSSD','HRV SDNN','HRV pNN50','HRV LF/HF',
    'Heart Rate','EDA Mean','EDA Std','EDA Peaks',
    'Resp Mean','Resp Std','Resp Rate',
    'Temp Mean','Temp Std','Activity Level','Activity Std',
    'EMG Mean','EMG Std'
]

# ─── MODEL ────────────────────────────────────────────────────────────────────
class MCUDE(nn.Module):
    def __init__(self, hidden_dim=64, num_features=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + num_features, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))
        self._beta_raw   = nn.Parameter(torch.tensor([-2.9]))
        self._alphas_raw = nn.Parameter(torch.ones(num_features) * (-2.2))
        self.current_features = self.current_t = None
        self.num_features = num_features

    @property
    def beta(self):   return F.softplus(self._beta_raw)
    @property
    def alphas(self): return F.softplus(self._alphas_raw)

    def set_current_batch(self, t, feat):
        self.current_t = t
        self.current_features = feat

    def forward(self, t, y):
        t_idx = torch.argmin(torch.abs(self.current_t - t))
        feat  = self.current_features[:, t_idx, :]
        f_kn  = -self.beta * y + torch.sum(self.alphas * feat, dim=-1)
        S_exp = y.unsqueeze(-1) if y.dim() == 1 else y
        f_nn  = self.net(torch.cat([S_exp, feat], dim=-1))
        if y.dim() == 1:
            f_nn = f_nn.squeeze(-1)
        return f_kn + f_nn


@st.cache_resource
def load_model(path):
    m = MCUDE()
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    m.eval()
    return m

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def get_subject_data(sid):
    num = sid.replace('S', '')
    path = os.path.join(DATA_DIR, f'u_wesad_{num}.csv')
    return load_csv(path)

def run_ode(model, feat_np, time_np, y0):
    if not HAS_ODE:
        return None
    n = len(time_np)
    t    = torch.FloatTensor(time_np)
    feat = torch.FloatTensor(feat_np).unsqueeze(0)
    y0t  = torch.tensor([float(y0)])
    model.set_current_batch(t, feat)
    with torch.no_grad():
        yp = odeint(model, y0t, t, method='euler').squeeze().numpy()
    return yp

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StressLens - MC-UDE Demo",
    page_icon="https://img.icons8.com/emoji/48/brain-emoji.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

  html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(140deg, #0d0d1a 0%, #1a1a3e 50%, #0d2137 100%);
    color: #e8e8f0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
  }

  /* Header */
  .app-header { margin-bottom: 8px; }
  .app-title {
    font-size: 2.6rem; font-weight: 900; letter-spacing: -1.5px;
    background: linear-gradient(90deg, #7f7fd5, #86a8e7, #91eae4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: inline;
  }
  .app-sub {
    color: #888; font-size: 1rem; margin-top: 2px;
  }

  /* Cards */
  .kpi-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(8px);
    margin-bottom: 8px;
  }
  .kpi-num  { font-size: 2rem; font-weight: 800; color: #91eae4; }
  .kpi-label{ font-size: 0.82rem; color: #aaa; margin-top: 4px; }

  /* Equation box */
  .eq-box {
    background: rgba(127,127,213,0.12);
    border-left: 4px solid #7f7fd5;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    font-family: 'Courier New', monospace;
    font-size: 1.05rem;
    color: #d8d8ff;
    margin: 10px 0;
    word-break: break-word;
  }

  /* Info box */
  .info-box {
    background: rgba(134,168,231,0.1);
    border-left: 4px solid #86a8e7;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    color: #d0e4ff;
    margin: 10px 0;
    font-size: 0.95rem;
    line-height: 1.6;
  }

  /* Success box */
  .success-box {
    background: rgba(46,204,113,0.1);
    border-left: 4px solid #2ecc71;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    color: #d0ffd0;
    margin: 10px 0;
    font-size: 0.95rem;
  }

  /* Step chip */
  .step-chip {
    display: inline-block;
    background: linear-gradient(90deg,#7f7fd5,#86a8e7);
    color: white; border-radius: 20px;
    padding: 3px 14px; font-size: 0.78rem;
    font-weight: 700; margin-right: 8px;
    letter-spacing: 0.5px;
  }

  /* Result card */
  .result-card {
    background: rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
  }
  .result-icon { font-size: 2.8rem; line-height: 1.2; }
  .result-label { font-size: 1.4rem; font-weight: 800; margin-top: 8px; }
  .result-conf  { font-size: 0.9rem; color: #aaa; margin-top: 4px; }

  /* Divider */
  hr { border-color: rgba(255,255,255,0.08); }

  /* Streamlit overrides */
  [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 800 !important; }
  button[kind="primary"] {
    background: linear-gradient(90deg,#7f7fd5,#86a8e7) !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
  }
  .stSelectbox label, .stSlider label, .stRadio label { color: #ccc !important; font-weight: 600; }
  .stTabs [data-baseweb="tab"] { font-weight: 700; font-size: 0.9rem; letter-spacing: 0.3px; }
  .stTabs [data-baseweb="tab-highlight"] { background: #7f7fd5 !important; }
</style>
""", unsafe_allow_html=True)

# ─── APP HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span class="app-title">StressLens</span>
  &nbsp;&nbsp;<span style="font-size:1.5rem;">&#128300;</span>
</div>
<div class="app-sub">Multi-Coefficient Universal Differential Equation &mdash; Live Demo &amp; Explainer</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation Guide")
    st.markdown("""
| Tab | Topic |
|-----|-------|
| **The Story** | Problem to solution |
| **The Equation** | MC-UDE math explained |
| **Simulation** | Real-time stress replay |
| **Cohorts** | Phase 2 cohort system |
| **Cold-Start** | Assign new person in 10 min |
| **What-If** | Intervention simulator |
""")
    st.divider()
    st.markdown("### Key Results")
    c1, c2 = st.columns(2)
    c1.metric("Indiv. MSE",  "0.00636")
    c2.metric("Cohort MSE", "0.00263", delta="-59%", delta_color="normal")
    c1.metric("Cold-Start", "93.3%")
    c2.metric("Degradation", "0.49x", help="<1.0 = cohort BEATS individual")
    st.divider()
    st.caption("Dataset: WESAD Wearable Stress | 15 subjects | 18 features")

# ─── TABS ─────────────────────────────────────────────────────────────────────
T = st.tabs([
    "The Story",
    "The Equation",
    "Live Simulation",
    "Cohort System",
    "Cold-Start Demo",
    "What-If Simulator"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — THE STORY
# ═══════════════════════════════════════════════════════════════════════════════
with T[0]:
    st.markdown("## The Problem &rarr; Our Solution")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="kpi-card">
          <div style="font-size:2.2rem;">&#128165;</div>
          <div class="kpi-label">Burnout affects 76% of healthcare workers</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="kpi-card">
          <div style="font-size:2.2rem;">&#8987;</div>
          <div class="kpi-label">Wearables stream rich real-time physiology</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="kpi-card">
          <div style="font-size:2.2rem;">&#10067;</div>
          <div class="kpi-label">Existing ML models are black boxes</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Why Not LSTM or Random Forest?")
        st.markdown("""<div class="info-box">
        Standard ML models can <b>predict</b> stress but cannot <b>explain</b> it.<br><br>
        &bull; Cannot answer: <i>"What if this nurse's heart rate dropped 20%?"</i><br>
        &bull; Cannot produce an interpretable equation for a clinician<br>
        &bull; Cannot simulate future stress without real data<br>
        &bull; Black box &mdash; no trust in clinical settings
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("### Our Solution: MC-UDE")
        st.markdown("""<div class="success-box">
        The <b>Multi-Coefficient Universal Differential Equation</b> combines:<br><br>
        &bull; <b>Interpretable ODE</b> &mdash; you can read the equation<br>
        &bull; <b>Sparse feature weights</b> (alpha) &mdash; automatic feature selection<br>
        &bull; <b>Neural residual term</b> &mdash; captures nonlinear dynamics<br>
        &bull; <b>Cohort grouping</b> &mdash; cold-start from minute 1<br>
        &bull; <b>What-if simulation</b> &mdash; intervention planning
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### The MC-UDE Equation")
    st.markdown("""<div class="eq-box">
    dS/dt = &minus;&beta;&middot;S(t) &nbsp;+&nbsp; &sum;<sub>i</sub> &alpha;<sub>i</sub>&middot;F<sub>i</sub>(t) &nbsp;+&nbsp; NN(S, F)
    </div>
    <div style="margin-top:10px;color:#aaa;font-size:0.9rem;">
    <b>S(t)</b> = stress level &nbsp;|&nbsp;
    <b>&beta;</b> = natural recovery rate &nbsp;|&nbsp;
    <b>&alpha;<sub>i</sub></b> = feature sensitivity (sparse) &nbsp;|&nbsp;
    <b>NN</b> = neural residual
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Two-Phase Approach")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<span class="step-chip">PHASE 1</span>', unsafe_allow_html=True)
        st.markdown("""
**Individual MC-UDE**
- Train one model per person (LOSO cross-validation)
- Each person gets their own equation
- Tests 15 subjects individually
- Mean MSE: **0.006355**
- Limitation: needs days of data per person
        """)
    with p2:
        st.markdown('<span class="step-chip">PHASE 2</span>', unsafe_allow_html=True)
        st.markdown("""
**Cohort MC-UDE**
- Cluster people by stress signature (alpha profiles)
- Pool data per cohort, train one shared equation
- Cold-start: just 10 min of baseline needed
- Mean MSE: **0.002627** (58% better!)
- Cold-start accuracy: **93.3%**
        """)

    st.markdown("---")
    st.markdown("### Phase 2 Pipeline Visualization")
    dend = os.path.join(CLUST_DIR, 'fig1_dendrogram.png')
    if os.path.exists(dend):
        st.image(dend, caption="Hierarchical clustering of alpha-profiles identifies cohorts", use_container_width=True)
    else:
        st.info("Dendrogram image not found — run 01_cluster_profiles.py first.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — THE EQUATION
# ═══════════════════════════════════════════════════════════════════════════════
with T[1]:
    st.markdown("## The MC-UDE Equation &mdash; Interactive Explainer")

    st.markdown("""<div class="eq-box" style="font-size:1.2rem;text-align:center;">
    dS/dt &nbsp;=&nbsp; <span style="color:#e74c3c;">&minus;&beta;&middot;S</span>
    &nbsp;+&nbsp; <span style="color:#2ecc71;">&sum; &alpha;<sub>i</sub>&middot;F<sub>i</sub></span>
    &nbsp;+&nbsp; <span style="color:#f39c12;">NN(S, F)</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Click each term to understand it:")
    t1, t2, t3 = st.columns(3)

    with t1:
        with st.expander("RECOVERY TERM:   -beta x S", expanded=False):
            st.markdown("""
**What it means:** Stress decays naturally over time.

- **beta = 0.062** (learned per subject)
- If S=0.8, recovery = -0.062 x 0.8 = -0.050 per step
- Models the body's natural homeostasis
- **Constrained** to [0.01, 5.0] by physics loss

Higher beta = faster recovery from stress.
            """)
            b_val = st.slider("Try beta:", 0.01, 0.20, 0.062, 0.005, key="beta_demo")
            t_arr = np.linspace(0, 30, 300)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_arr, y=np.exp(-b_val * t_arr),
                mode='lines', line=dict(color='#e74c3c', width=3),
                name=f'Free decay beta={b_val:.3f}'))
            fig.update_layout(
                title=f"Stress free decay with beta={b_val:.3f}",
                xaxis_title="Time (min)", yaxis_title="Stress",
                template='plotly_dark', height=260,
                margin=dict(l=20,r=20,t=40,b=40))
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        with st.expander("STRESSOR TERM:   sum(alpha_i x F_i)", expanded=False):
            st.markdown("""
**What it means:** Each sensor feature drives stress up.

- **alpha_i** = sensitivity to feature i
- L1 regularization makes most alpha_i shrink to near-zero
- Only the most important features stay active
- Different cohorts have different top features

**Cardiac Responder:** Heart Rate, Workload drive stress  
**Cognitive Responder:** Resp Rate, Temperature drive stress
            """)
            demo_alphas = {
                'Heart Rate': 0.088, 'Workload': 0.088, 'EDA Std': 0.081,
                'EDA Mean': 0.081, 'Resp Std': 0.080, 'HRV pNN50': 0.077,
                'EMG Mean': 0.065, 'Temp Std': 0.026
            }
            fig = go.Figure(go.Bar(
                x=list(demo_alphas.values()),
                y=list(demo_alphas.keys()),
                orientation='h',
                marker_color=['#e74c3c' if v > 0.083 else '#3498db' for v in demo_alphas.values()]))
            fig.update_layout(title="Example: Cardiac Responder alpha profile",
                template='plotly_dark', height=280,
                xaxis_title="alpha value",
                margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

    with t3:
        with st.expander("NEURAL RESIDUAL:   NN(S, F)", expanded=False):
            st.markdown("""
**What it means:** A small neural network captures complex nonlinear dynamics 
that the linear ODE cannot express.

**Architecture:**
```
Input: [S(t), F1, F2, ..., F18]  (19 dims)
  -> Linear(64) -> Tanh
  -> Linear(64) -> Tanh
  -> Linear(1)  -> dS/dt residual
```

**Why both ODE + NN?**
- ODE part = **interpretable** (you can read it)
- NN part = **accurate** (captures remaining patterns)
- Together = Universal Differential Equation (UDE)

The NN term typically contributes <30% of the total signal.
            """)

    st.markdown("---")
    st.markdown("### View a Real Learned Equation")

    subj_eq = st.selectbox("Select subject:", SUBJECTS, key="eq_subj")
    fold_eq  = SUBJECTS.index(subj_eq) + 1
    prof     = load_json(os.path.join(P1_DIR, f'profile_fold_{fold_eq}.json'))

    if prof:
        st.markdown(f"""<div class="eq-box">
        {prof.get('equation', 'N/A')}
        </div>""", unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Beta", f"{prof['beta']:.4f}")
        mc2.metric("Active Features", f"{prof['n_active']}/18")
        mc3.metric("Sparsity", f"{prof['sparsity_pct']:.0f}%")
        mc4.metric("Test MSE", f"{prof.get('test_mse', 0):.5f}")

        alphas   = np.array(prof['all_alphas'])
        sort_idx = np.argsort(alphas)[::-1]
        fig = go.Figure(go.Bar(
            x=[FEAT_NAMES[i] for i in sort_idx],
            y=alphas[sort_idx],
            marker=dict(
                color=alphas[sort_idx],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="alpha"))))
        fig.update_layout(
            title=f"{subj_eq} — Feature Sensitivity Profile",
            xaxis_tickangle=-45, template='plotly_dark', height=360,
            yaxis_title="alpha (sensitivity)",
            margin=dict(b=120))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Profile not found — ensure Phase 1 results are in phase1_individual/mc_ude_results/")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
with T[2]:
    st.markdown("## Live Stress Simulation")
    st.markdown("""<div class="info-box">
    Select a subject and time window. We load their real wearable sensor data, 
    run it through the trained MC-UDE, and plot predicted vs actual stress &mdash;
    like a real-time physiological monitor.
    </div>""", unsafe_allow_html=True)

    if not HAS_ODE:
        st.error("torchdiffeq not installed. Run:  pip install torchdiffeq")
    else:
        col_ctrl, col_plot = st.columns([1, 2.5])

        with col_ctrl:
            sim_subj = st.selectbox("Subject", SUBJECTS, key='sim_subj')
            model_choice = st.radio("Model", ["Individual (Phase 1)", "Cohort (Phase 2)"],
                                    help="Individual: trained only on this person (LOSO)\nCohort: trained on the whole cohort")
            w_start = st.slider("Window start (rows)", 0, 800, 100, 50)
            w_len   = st.slider("Window length (rows)", 30, 120, 60, 10)
            run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

        with col_plot:
            if run_btn:
                df = get_subject_data(sim_subj)
                if df is None:
                    st.error(f"Data not found for {sim_subj}")
                else:
                    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
                    features  = df[feat_cols].values.astype(np.float32)
                    stress    = df['stress'].values.astype(np.float32)
                    time_arr  = df['time'].values.astype(np.float32)

                    end = min(w_start + w_len, len(df))
                    f_seg = features[w_start:end]
                    s_seg = stress[w_start:end]
                    t_seg = time_arr[w_start:end] - time_arr[w_start]

                    fold_sim = SUBJECTS.index(sim_subj) + 1
                    cohort_id = COHORT_MAP[sim_subj]

                    if "Individual" in model_choice:
                        mpath  = os.path.join(P1_DIR, f'mcude_fold_{fold_sim}.pth')
                        ppath  = os.path.join(P1_DIR, f'profile_fold_{fold_sim}.json')
                        mlabel = f"{sim_subj} Individual"
                        mcolor = '#2ecc71'
                    else:
                        # LOO cohort model for this subject (if available), else full cohort model
                        loo_path = os.path.join(P2_DIR, f'cohort{cohort_id}_{sim_subj}.pth')
                        full_path= os.path.join(P2_DIR, f'cohort{cohort_id}_full.pth')
                        mpath  = loo_path if os.path.exists(loo_path) else full_path
                        ppath  = os.path.join(P2_DIR, f'cohort{cohort_id}_{sim_subj}_profile.json')
                        if not os.path.exists(ppath):
                            ppath = os.path.join(P2_DIR, f'cohort{cohort_id}_full_profile.json')
                        mlabel = f"{COHORT_NAMES[cohort_id]} (Cohort)"
                        mcolor = '#3498db'

                    if not os.path.exists(mpath):
                        st.error(f"Model not found: {mpath}")
                    else:
                        model = load_model(mpath)
                        with st.spinner("Solving ODE..."):
                            y_pred = run_ode(model, f_seg, t_seg, s_seg[0])

                        mse = float(np.mean((y_pred - s_seg[:len(y_pred)])**2)) if y_pred is not None else None

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=t_seg[:len(s_seg)], y=s_seg,
                            mode='lines', name='Actual Stress',
                            line=dict(color='#f39c12', width=2.5)))
                        if y_pred is not None:
                            fig.add_trace(go.Scatter(
                                x=t_seg[:len(y_pred)], y=y_pred,
                                mode='lines', name=mlabel,
                                line=dict(color=mcolor, width=2.5, dash='dash')))
                        fig.update_layout(
                            title=f"Stress Trajectory: {sim_subj} | {mlabel}",
                            xaxis_title="Time (normalized)", yaxis_title="Stress Level",
                            template='plotly_dark', height=380,
                            legend=dict(orientation='h', y=-0.2),
                            margin=dict(b=80))
                        st.plotly_chart(fig, use_container_width=True)

                        if mse is not None:
                            rc1, rc2 = st.columns(2)
                            rc1.metric("Prediction MSE", f"{mse:.5f}")
                            prof_sim = load_json(ppath)
                            if prof_sim:
                                rc2.metric("Active Features", f"{prof_sim['n_active']}/18")
                                st.markdown(f"""<div class="eq-box">
                                {prof_sim.get('equation','')}
                                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="height:380px;display:flex;align-items:center;
                justify-content:center;border:2px dashed rgba(255,255,255,0.1);
                border-radius:14px;color:#555;font-size:1rem;">
                Configure options on the left and click Run Simulation
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COHORT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
with T[3]:
    st.markdown("## Phase 2: Cohort-Level MC-UDE")
    st.markdown("""<div class="info-box">
    <b>The Insight:</b> Instead of training one model per person (which needs lots of 
    individual data), we cluster people by their stress response signature and train 
    <b>one shared equation per cluster</b>. Pooling 13 subjects gives 12x more training 
    data, reducing overfitting.<br><br>
    Clustering is done on the <b>learned alpha-vectors</b> from Phase 1 &mdash; not on 
    raw sensor data. This groups people who respond to stress through the same physiological channels.
    </div>""", unsafe_allow_html=True)

    st.markdown("### Step-by-Step Process")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown('<span class="step-chip">STEP 1</span>', unsafe_allow_html=True)
        st.markdown("**Extract alpha profiles**  \nEach of 15 subjects has an 18-dim alpha vector from Phase 1. These are their unique stress signatures.")
    with s2:
        st.markdown('<span class="step-chip">STEP 2</span>', unsafe_allow_html=True)
        st.markdown("**Hierarchical clustering**  \nAgglomerative clustering on alpha-space. K=2 chosen by silhouette score = 0.82 (excellent).")
    with s3:
        st.markdown('<span class="step-chip">STEP 3</span>', unsafe_allow_html=True)
        st.markdown("**Train Cohort MC-UDE**  \nPool all members' data into one dataset. Train one MC-UDE per cohort. Evaluate with LOO.")

    st.markdown("---")
    figs_col1, figs_col2 = st.columns(2)
    with figs_col1:
        fp = os.path.join(FIG_DIR, 'fig1_cohort_equations.png')
        if os.path.exists(fp):
            st.image(fp, caption="Cohort alpha profiles: Cardiac vs Cognitive Responder", use_container_width=True)
    with figs_col2:
        hp = os.path.join(CLUST_DIR, 'fig2_cohort_heatmap.png')
        if os.path.exists(hp):
            st.image(hp, caption="Per-subject alpha heatmap grouped by cohort", use_container_width=True)

    st.markdown("---")
    st.markdown("### Results: Cohort vs Individual Performance")

    eval_df = load_csv(EVAL_CSV)
    if eval_df is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indiv. MSE (mean)", f"{eval_df['Individual_MSE'].mean():.5f}")
        m2.metric("Cohort MSE (mean)", f"{eval_df['Cohort_MSE'].mean():.5f}",
                  delta=f"-{(1 - eval_df['Cohort_MSE'].mean()/eval_df['Individual_MSE'].mean())*100:.1f}%",
                  delta_color="normal")
        m3.metric("Degradation Ratio", f"{eval_df['Degradation_Ratio'].mean():.2f}x",
                  help="<1.0 means cohort BEATS individual")
        m4.metric("Subjects improved", f"{(eval_df['Degradation_Ratio'] < 1).sum()}/15")

        # Bar chart comparison
        ev = eval_df.sort_values('Degradation_Ratio').reset_index(drop=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Individual UDE', x=ev['Subject'], y=ev['Individual_MSE'],
                             marker_color='#e74c3c', opacity=0.85))
        fig.add_trace(go.Bar(name='Cohort UDE',     x=ev['Subject'], y=ev['Cohort_MSE'],
                             marker_color='#3498db', opacity=0.85))
        fig.add_trace(go.Bar(name='Naive Baseline', x=ev['Subject'], y=ev['Naive_MSE'],
                             marker_color='#7f8c8d', opacity=0.6))
        fig.update_layout(barmode='group', template='plotly_dark', height=380,
            title="Individual vs Cohort vs Naive MSE (lower = better)",
            xaxis_title="Subject", yaxis_title="MSE",
            legend=dict(orientation='h', y=-0.25), margin=dict(b=100))
        st.plotly_chart(fig, use_container_width=True)

        # Degradation ratio
        colors = ['#2ecc71' if r < 0.5 else '#f39c12' if r < 0.75 else '#e67e22'
                  for r in ev['Degradation_Ratio']]
        fig2 = go.Figure(go.Bar(
            x=ev['Subject'], y=ev['Degradation_Ratio'],
            marker_color=colors,
            text=[f"{r:.2f}x" for r in ev['Degradation_Ratio']],
            textposition='outside'))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="white", line_width=2,
                       annotation_text="Break-even (1.0)", annotation_position="top right")
        fig2.add_hline(y=1.5, line_dash="dot", line_color="orange",
                       annotation_text="Acceptable limit (1.5)")
        fig2.update_layout(template='plotly_dark', height=340,
            title="Degradation Ratio: Cohort MSE / Individual MSE (all below 1.0 = cohort wins)",
            yaxis_title="Ratio", yaxis=dict(range=[0, 1.1]),
            margin=dict(b=60, t=60))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""<div class="success-box">
        <b>Key Finding:</b> The cohort model outperforms the individual model for 
        <b>all 15 subjects</b> (degradation ratio &lt; 1.0 for every case). 
        Pooling more data reduces overfitting better than per-person training.
        Best result: <b>S008 (0.12x)</b> &mdash; cohort is 8x more accurate than individual.
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Evaluation CSV not found at: " + EVAL_CSV)

    # Show real comparison chart from Kaggle
    if os.path.exists(EVAL_PNG):
        st.image(EVAL_PNG, caption="Kaggle-generated comparison: Cohort (blue) consistently dominates Individual (green)", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COLD-START DEMO
# ═══════════════════════════════════════════════════════════════════════════════
with T[4]:
    st.markdown("## Cold-Start Cohort Assignment")
    st.markdown("""<div class="info-box">
    <b>The Problem:</b> A new nurse joins today. We have zero history on them.<br>
    <b>The Solution:</b> Collect just 10 minutes of wearable baseline data &rarr; 
    classify their cohort &rarr; immediately predict their stress using the cohort equation.<br><br>
    <b>Result:</b> <b>93.3% LOO accuracy</b> on 15 subjects. Top discriminating feature: 
    Temperature variability (novel &mdash; most literature focuses on HR/EDA).
    </div>""", unsafe_allow_html=True)

    st.markdown("### Try It: Simulate a New Person's Baseline")
    st.markdown("Adjust the baseline physiological values (10-min averages from wearable):")

    col_s, col_r = st.columns([1, 1])
    with col_s:
        hr       = st.slider("Heart Rate (normalized 0-1)",  0.0, 1.0, 0.55, 0.05, key="cs_hr")
        temp_std = st.slider("Temp Variability (normalized 0-1)", 0.0, 1.0, 0.10, 0.05, key="cs_ts",
                             help="Top cold-start feature! Importance=0.20")
        eda      = st.slider("EDA Mean (normalized 0-1)", 0.0, 1.0, 0.40, 0.05, key="cs_eda")
        resp     = st.slider("Resp Rate (normalized 0-1)", 0.0, 1.0, 0.45, 0.05, key="cs_resp")
        temp_m   = st.slider("Temp Mean (normalized 0-1)", 0.0, 1.0, 0.60, 0.05, key="cs_tm")
        wrkld    = st.slider("Workload (normalized 0-1)",  0.0, 1.0, 0.50, 0.05, key="cs_wk")

    with col_r:
        # Simple rule-based classifier using feature importances from RF result:
        # temp_std=0.20, hr=0.08, eda=0.08, eda_std=0.08 → cardiac
        # high resp, high temp_mean → cognitive
        cardiac_score  = (hr * 0.30 + eda * 0.25 + wrkld * 0.15 + (1 - temp_std) * 0.30)
        cognitive_score= (resp * 0.35 + temp_m * 0.30 + wrkld * 0.15 + (1 - hr) * 0.20)

        total = cardiac_score + cognitive_score + 1e-9
        cardiac_pct  = cardiac_score  / total * 100
        cognitive_pct= cognitive_score / total * 100
        assigned = "Cardiac Responder"  if cardiac_pct >= cognitive_pct else "Cognitive Responder"
        conf     = max(cardiac_pct, cognitive_pct)
        icon     = "&#9829;" if assigned == "Cardiac Responder" else "&#129504;"
        color    = "#2ecc71" if conf > 70 else "#f39c12"

        st.markdown(f"""<div class="result-card">
          <div class="result-icon">{icon}</div>
          <div class="result-label" style="color:{color};">{assigned}</div>
          <div class="result-conf">Confidence: {conf:.0f}%</div>
        </div>""", unsafe_allow_html=True)

        # Confidence bar
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf,
            title={'text': "Assignment Confidence (%)"},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [50, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [50, 65], 'color': 'rgba(231,76,60,0.2)'},
                    {'range': [65, 80], 'color': 'rgba(243,156,18,0.2)'},
                    {'range': [80, 100],'color': 'rgba(46,204,113,0.2)'}
                ],
                'threshold': {'line': {'color': 'white', 'width': 3}, 'value': 80}
            }))
        fig_gauge.update_layout(template='plotly_dark', height=260, margin=dict(t=40,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("**Assigned Equation:**")
        if assigned == "Cardiac Responder":
            st.markdown("""<div class="eq-box">
            dS/dt = -0.062&middot;S + <b>0.088&middot;HeartRate</b>
            + <b>0.088&middot;Workload</b> + 0.081&middot;EDA_Std + ... + NN(S,F)
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="eq-box">
            dS/dt = -0.056&middot;S + <b>0.110&middot;Resp_Mean</b>
            + <b>0.108&middot;Resp_Rate</b> + 0.108&middot;Temp_Mean + ... + NN(S,F)
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Feature importance bar
    importance = {
        'Temp Variability': 0.200,
        'Heart Rate': 0.080,
        'EDA Mean': 0.080,
        'EDA Std Mean': 0.080,
        'EDA Std Std': 0.080,
        'Workload': 0.060,
        'Resp Rate': 0.055,
        'Resp Mean': 0.045,
    }
    fig_imp = go.Figure(go.Bar(
        x=list(importance.values()),
        y=list(importance.keys()),
        orientation='h',
        marker=dict(color=['#e74c3c' if v == max(importance.values()) else '#3498db'
                           for v in importance.values()])))
    fig_imp.update_layout(
        title="Cold-Start Feature Importances (from Random Forest Classifier)",
        template='plotly_dark', height=320,
        xaxis_title="Importance Score",
        margin=dict(l=10, r=10, t=60, b=40))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""<div class="info-box">
    <b>Novel Finding:</b> Temperature variability (not Heart Rate or EDA) is the <b>top discriminating feature</b>
    for cohort assignment with importance=0.20. This has not been previously reported in wearable stress literature,
    which typically focuses on cardiac and electrodermal measures.
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with T[5]:
    st.markdown("## What-If Intervention Simulator")
    st.markdown("""<div class="info-box">
    This turns the MC-UDE into a <b>planning tool</b>. Because we have an explicit equation,
    we can ask: "If we reduce this nurse's workload by 30%, how quickly will their stress recover?"
    Adjust any parameter and see the stress trajectory update instantly.
    </div>""", unsafe_allow_html=True)

    cohort_sel = st.selectbox("Select cohort:", ['Cardiac Responder', 'Cognitive Responder'], key="wi_cohort")

    COHORT_DATA = {
        'Cardiac Responder': {
            'beta': 0.062,
            'alphas': {
                'Heart Rate': 0.088, 'Workload': 0.088, 'EDA Std': 0.081,
                'EDA Mean': 0.081, 'Resp Std': 0.080, 'Resp Rate': 0.080,
                'EDA Peaks': 0.077, 'HRV pNN50': 0.077, 'Activity': 0.076,
                'Temp Mean': 0.074, 'HRV SDNN': 0.074, 'HRV RMSSD': 0.053
            }
        },
        'Cognitive Responder': {
            'beta': 0.056,
            'alphas': {
                'Resp Mean': 0.110, 'Resp Rate': 0.108, 'Temp Mean': 0.108,
                'EDA Peaks': 0.107, 'Activity': 0.105, 'HRV LF/HF': 0.105,
                'Heart Rate': 0.104, 'Workload': 0.103, 'HRV RMSSD': 0.102
            }
        }
    }

    cohort_info = COHORT_DATA[cohort_sel]
    all_features = list(cohort_info['alphas'].keys())

    c_ctrl, c_plot = st.columns([1, 2])

    with c_ctrl:
        st.markdown("### Intervention Controls")
        beta_override = st.slider("Recovery rate (beta)", 0.01, 0.30,
                                   cohort_info['beta'], 0.005, key="wi_beta",
                                   help="Higher = faster recovery from stress")
        s0 = st.slider("Initial stress S0", 0.1, 1.0, 0.75, 0.05, key="wi_s0")
        duration = st.slider("Duration (minutes)", 5, 40, 20, 5, key="wi_dur")

        st.markdown("**Feature scaling (1.0 = no change):**")
        scales = {}
        for feat in all_features[:6]:  # Show top 6 features
            scales[feat] = st.slider(feat, 0.0, 2.0, 1.0, 0.1, key=f"wi_{feat}")

    with c_plot:
        # Simulate
        dt     = 0.05
        t_arr  = np.arange(0, duration, dt)
        beta_b = cohort_info['beta']

        baseline_contrib = sum(cohort_info['alphas'].values()) * 0.5

        S_base, S_mod = [s0], [s0]
        for _ in t_arr[1:]:
            scaled_contrib = sum(
                cohort_info['alphas'].get(f, 0) * scales.get(f, 1.0) * 0.5
                for f in all_features)
            S_base.append(max(0, min(1, S_base[-1] + dt * (-beta_b * S_base[-1] + baseline_contrib))))
            S_mod.append(max(0, min(1, S_mod[-1]  + dt * (-beta_override * S_mod[-1] + scaled_contrib))))

        fig_wi = go.Figure()
        fig_wi.add_trace(go.Scatter(x=t_arr, y=S_base, mode='lines',
            name='Baseline (no change)', line=dict(color='#e74c3c', width=2.5)))
        fig_wi.add_trace(go.Scatter(x=t_arr, y=S_mod, mode='lines',
            name='With intervention', line=dict(color='#2ecc71', width=2.5, dash='dash')))
        fig_wi.add_hline(y=0.5, line_dash="dot", line_color='#f39c12', line_width=1.5,
                         annotation_text="Alert threshold (0.5)")
        fig_wi.update_layout(
            title=f"Stress Trajectory: {cohort_sel}",
            xaxis_title="Time (minutes)", yaxis_title="Stress S(t)",
            yaxis=dict(range=[0, 1.05]),
            template='plotly_dark', height=380,
            legend=dict(orientation='h', y=-0.2),
            margin=dict(b=80))
        st.plotly_chart(fig_wi, use_container_width=True)

        # Impact metrics
        final_b, final_m = S_base[-1], S_mod[-1]
        peak_b,  peak_m  = max(S_base), max(S_mod)
        t_under_b = sum(1 for s in S_base if s < 0.5) * dt
        t_under_m = sum(1 for s in S_mod  if s < 0.5) * dt

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Final stress (baseline)",    f"{final_b:.3f}")
        mc2.metric("Final stress (intervened)",  f"{final_m:.3f}",
                   delta=f"{final_m-final_b:+.3f}", delta_color="inverse")
        mc3.metric("Time below threshold",
                   f"+{t_under_m - t_under_b:.1f} min",
                   delta_color="normal",
                   help="Extra minutes spent below the 0.5 alert threshold")

    st.markdown("---")
    fig_whatif = os.path.join(FIG_DIR, 'fig3_cohort_whatif.png')
    if os.path.exists(fig_whatif):
        st.image(fig_whatif,
                 caption="Feature intervention impact from trained models (red=2x, green=0.5x)",
                 use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""<div style="text-align:center;color:#555;font-size:0.8rem;">
StressLens &nbsp;&bull;&nbsp; Multi-Coefficient UDE &nbsp;&bull;&nbsp;
WESAD Dataset &nbsp;&bull;&nbsp; 15 subjects &nbsp;&bull;&nbsp; 18 features &nbsp;&bull;&nbsp;
Phase 1: Individual LOSO &nbsp;&bull;&nbsp; Phase 2: Cohort Cold-Start
</div>""", unsafe_allow_html=True)
