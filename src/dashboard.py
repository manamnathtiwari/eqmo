import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchdiffeq import odeint
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ude_model import UDE
from src.models.pde_model import StressPDE, create_grid_graph, create_similarity_graph

st.title("Burnout Dynamics: UDE + PDE Model")

@st.cache_resource
def load_model():
    model = UDE()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'results', 'ude_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model()

st.sidebar.header("Simulation Settings")
mode = st.sidebar.selectbox("Mode", ["Individual", "Group (PDE)"])

if mode == "Individual":
    st.subheader("Individual Trajectory")
    user_id = st.sidebar.selectbox("Select User", [f"u_{i:03d}" for i in range(50)])
    
    data_path = os.path.join('data', 'processed', f'{user_id}.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.line_chart(df[['stress', 'workload']])
        
        # Predict
        if st.button("Run UDE Prediction"):
            seq_len = 200
            t = torch.linspace(0, seq_len-1, seq_len)
            u = torch.tensor(df['workload'].values[:seq_len], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            y0 = torch.tensor(df['stress'].values[0], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            model.set_current_batch(t, u)
            with torch.no_grad():
                y_pred = odeint(model, y0, t, method='rk4')
                y_pred = y_pred.permute(1, 0, 2).numpy().flatten()
                
            pred_df = pd.DataFrame({
                'time': t.numpy(),
                'Predicted Stress': y_pred,
                'Actual Stress': df['stress'].values[:seq_len]
            })
            st.line_chart(pred_df.set_index('time'))
            
    else:
        st.error("User data not found.")

elif mode == "Group (PDE)":
    st.subheader("Group Diffusion Simulation")
    diffusion = st.sidebar.slider("Diffusion Coefficient", 0.0, 0.5, 0.05)
    graph_type = st.sidebar.selectbox("Graph Topology", ["Grid (Spatial)", "Similarity (Cohort)"])
    
    if st.button("Run Group Simulation"):
        # Load first 20 users
        data_dir = os.path.join('data', 'processed')
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])[:20]
        
        all_u = []
        all_y0 = []
        all_features = []
        
        for f in files:
            df = pd.read_csv(os.path.join(data_dir, f))
            all_u.append(df['workload'].values[:200])
            all_y0.append(df['stress'].values[0])
            # Feature for similarity
            all_features.append([df['stress'].mean(), df['workload'].mean()])
            
        u_tensor = torch.tensor(np.array(all_u), dtype=torch.float32).unsqueeze(-1)
        y0_tensor = torch.tensor(np.array(all_y0), dtype=torch.float32).unsqueeze(-1)
        t_tensor = torch.linspace(0, 199, 200)
        
        model.set_current_batch(t_tensor, u_tensor)
        
        if graph_type == "Grid (Spatial)":
            adj = create_grid_graph(4, 5)
        else:
            # Similarity
            adj = create_similarity_graph(np.array(all_features), k=3)
            
        pde = StressPDE(20, adj, model, diffusion_coeff=diffusion)
        
        with torch.no_grad():
            y_coupled = odeint(pde, y0_tensor, t_tensor, method='rk4')
            
        # Visualize average stress
        avg_stress = y_coupled.mean(dim=1).squeeze().numpy()
        st.line_chart(avg_stress)
        st.caption("Average Group Stress over Time")
        
        # Heatmap of final state
        final_state = y_coupled[-1, :, 0].numpy()
        
        # If grid, reshape. If similarity, just show bar chart or 1D heatmap
        if graph_type == "Grid (Spatial)":
            final_state = final_state.reshape(4, 5)
            fig, ax = plt.subplots()
            im = ax.imshow(final_state, cmap='hot', vmin=0, vmax=1)
            plt.colorbar(im)
            st.pyplot(fig)
        else:
            st.bar_chart(final_state)
            st.caption("Final Stress Levels per User (Sorted by ID)")

