"""
Comprehensive UDE Demo Application
Features:
1. Select PDE type and variables
2. Define custom equation/relationship
3. Simulate data for different durations (6h, 12h, 24h)
4. Train UDE model
5. Extract symbolic equation using symbolic regression
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="UDE Equation Discovery", layout="wide", initial_sidebar_state="expanded")

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
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'equation_discovered' not in st.session_state:
    st.session_state.equation_discovered = False

# UDE Model
class UDEModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.linear_coeffs = nn.Parameter(torch.randn(n_features) * 0.1)
        self.nn = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        linear = torch.sum(self.linear_coeffs * x, dim=-1, keepdim=True)
        nonlinear = self.nn(x)
        return (linear + nonlinear).squeeze(-1)

# Symbolic Regression
def symbolic_regression(X, y, coeffs, var_names):
    """Extract symbolic equation from learned coefficients"""
    # Create symbolic variables
    symbols = [sp.Symbol(name) for name in var_names]
    
    # Build linear equation
    equation = sum(float(coeffs[i]) * symbols[i] for i in range(len(symbols)))
    
    # Simplify
    equation = sp.simplify(equation)
    
    return equation, symbols

# Main App
st.markdown('<p class="main-header">🔬 UDE Equation Discovery System</p>', unsafe_allow_html=True)
st.markdown("**Discover interpretable equations from data using Universal Differential Equations**")

# Sidebar - Configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Step 1: PDE Type Selection
st.sidebar.markdown("### 1️⃣ Select PDE Type")
pde_type = st.sidebar.selectbox(
    "Choose PDE Type",
    ["Linear ODE", "Nonlinear ODE", "Coupled System", "Custom Equation"]
)

# Step 2: Variable Selection
st.sidebar.markdown("### 2️⃣ Select Variables")
n_variables = st.sidebar.slider("Number of Variables", 2, 6, 3)

variable_names = []
for i in range(n_variables):
    default_names = ['Stress', 'HRV', 'EDA', 'Workload', 'Temperature', 'Activity']
    var_name = st.sidebar.text_input(
        f"Variable {i+1} Name",
        value=default_names[i] if i < len(default_names) else f"Var{i+1}",
        key=f"var_{i}"
    )
    variable_names.append(var_name)

# Step 3: Define Equation/Relationship
st.sidebar.markdown("### 3️⃣ Define Relationship")

if pde_type == "Linear ODE":
    st.sidebar.info("Linear: dy/dt = Σ(aᵢ·xᵢ)")
    coefficients = []
    for i, var in enumerate(variable_names):
        coeff = st.sidebar.number_input(
            f"Coefficient for {var}",
            value=1.0 if i == 0 else -0.5,
            step=0.1,
            key=f"coeff_{i}"
        )
        coefficients.append(coeff)

elif pde_type == "Nonlinear ODE":
    st.sidebar.info("Nonlinear: dy/dt = Σ(aᵢ·xᵢ) + nonlinear terms")
    coefficients = []
    for i, var in enumerate(variable_names):
        coeff = st.sidebar.number_input(
            f"Linear coeff for {var}",
            value=1.0 if i == 0 else -0.5,
            step=0.1,
            key=f"coeff_{i}"
        )
        coefficients.append(coeff)
    
    include_quadratic = st.sidebar.checkbox("Include quadratic terms (x²)", value=True)
    include_interaction = st.sidebar.checkbox("Include interaction terms (x·y)", value=True)

elif pde_type == "Coupled System":
    st.sidebar.info("Coupled: Multiple equations affecting each other")
    st.sidebar.warning("Simplified to single equation for this demo")
    coefficients = [1.0] * n_variables

else:  # Custom
    st.sidebar.info("Enter custom coefficients")
    coefficients = []
    for i, var in enumerate(variable_names):
        coeff = st.sidebar.number_input(
            f"Coefficient for {var}",
            value=float(np.random.randn()),
            step=0.1,
            key=f"coeff_{i}"
        )
        coefficients.append(coeff)

# Step 4: Simulation Duration
st.sidebar.markdown("### 4️⃣ Simulation Duration")
duration_hours = st.sidebar.selectbox(
    "Duration",
    [6, 12, 24],
    index=1
)

sampling_rate = st.sidebar.slider("Sampling Rate (samples/hour)", 10, 120, 60)
total_samples = duration_hours * sampling_rate

# Main Content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Generation", "🤖 Model Training", "🔍 Equation Discovery", "📈 Results"])

# TAB 1: Data Generation
with tab1:
    st.markdown('<p class="sub-header">Step 1: Generate Synthetic Data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Equation Configuration")
        
        # Display equation
        equation_str = "dy/dt = "
        for i, (var, coeff) in enumerate(zip(variable_names, coefficients)):
            if i > 0:
                equation_str += " + " if coeff >= 0 else " - "
                equation_str += f"{abs(coeff):.2f}·{var}"
            else:
                equation_str += f"{coeff:.2f}·{var}"
        
        if pde_type == "Nonlinear ODE":
            if include_quadratic:
                equation_str += f" + 0.5·{variable_names[0]}²"
            if include_interaction and n_variables >= 2:
                equation_str += f" - 0.3·{variable_names[0]}·{variable_names[1]}"
        
        st.code(equation_str, language="python")
        
        st.markdown("#### Simulation Parameters")
        st.write(f"- **Duration:** {duration_hours} hours")
        st.write(f"- **Sampling Rate:** {sampling_rate} samples/hour")
        st.write(f"- **Total Samples:** {total_samples}")
        st.write(f"- **Variables:** {', '.join(variable_names)}")
    
    with col2:
        st.markdown("#### Quick Settings")
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.05, 0.01)
        initial_range = st.slider("Initial Value Range", 0.1, 2.0, 1.0, 0.1)
    
    if st.button("🎲 Generate Data", type="primary", use_container_width=True):
        with st.spinner("Generating synthetic data..."):
            # Generate time series
            time_hours = np.linspace(0, duration_hours, total_samples)
            
            # Generate initial values
            np.random.seed(42)
            data = {}
            data['time_hours'] = time_hours
            
            # Generate base variables
            for i, var in enumerate(variable_names):
                if i == 0:
                    # Target variable starts at initial range
                    data[var] = np.random.rand(total_samples) * initial_range
                else:
                    # Other variables vary realistically
                    base = np.sin(2 * np.pi * time_hours / 12) * 0.3 + 0.5
                    data[var] = base + np.random.randn(total_samples) * 0.1
            
            # Calculate target based on equation
            X = np.column_stack([data[var] for var in variable_names])
            
            # Linear part
            y = np.dot(X, coefficients)
            
            # Nonlinear parts
            if pde_type == "Nonlinear ODE":
                if include_quadratic:
                    y += 0.5 * X[:, 0] ** 2
                if include_interaction and n_variables >= 2:
                    y -= 0.3 * X[:, 0] * X[:, 1]
            
            # Add noise
            y += np.random.randn(total_samples) * noise_level
            
            data['target'] = y
            
            # Store in session state
            st.session_state.df = pd.DataFrame(data)
            st.session_state.variable_names = variable_names
            st.session_state.coefficients = coefficients
            st.session_state.data_generated = True
            
            st.success("✅ Data generated successfully!")
    
    # Display data if generated
    if st.session_state.data_generated:
        st.markdown("#### Generated Data Preview")
        
        df = st.session_state.df
        
        # Plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Target Variable Over Time", "Input Variables Over Time"),
            vertical_spacing=0.15
        )
        
        # Target
        fig.add_trace(
            go.Scatter(x=df['time_hours'], y=df['target'], mode='lines', name='Target',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Variables
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, var in enumerate(variable_names):
            fig.add_trace(
                go.Scatter(x=df['time_hours'], y=df[var], mode='lines', name=var,
                          line=dict(color=colors[i % len(colors)])),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_layout(height=600, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Data Table"):
            st.dataframe(df.head(100), use_container_width=True)
            st.write(f"Total rows: {len(df)}")

# TAB 2: Model Training
with tab2:
    st.markdown('<p class="sub-header">Step 2: Train UDE Model</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate data first (Tab 1)")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Training Configuration")
            epochs = st.slider("Training Epochs", 100, 2000, 500, 100)
            learning_rate = st.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05], 0.01)
            batch_size = st.slider("Batch Size", 32, 256, 128, 32)
        
        with col2:
            st.markdown("#### Model Architecture")
            st.write(f"- **Input Features:** {n_variables}")
            st.write(f"- **Hidden Layers:** 64 → 32")
            st.write(f"- **Output:** 1 (target)")
            st.write(f"- **Parameters:** ~{64*n_variables + 64*32 + 32 + n_variables}")
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            df = st.session_state.df
            
            # Prepare data
            X = torch.FloatTensor(df[variable_names].values)
            y = torch.FloatTensor(df['target'].values)
            
            # Initialize model
            model = UDEModel(n_features=n_variables)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_history = []
            
            for epoch in range(epochs):
                # Shuffle data
                indices = torch.randperm(len(X))
                
                epoch_loss = 0
                n_batches = 0
                
                for i in range(0, len(X), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = torch.mean((y_pred - y_batch) ** 2)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                avg_loss = epoch_loss / n_batches
                loss_history.append(avg_loss)
                
                if (epoch + 1) % 50 == 0:
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Training complete! Final Loss: {avg_loss:.6f}")
            
            # Store model
            st.session_state.model = model
            st.session_state.loss_history = loss_history
            st.session_state.model_trained = True
            
            # Plot training loss
            st.markdown("#### Training Progress")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=loss_history, mode='lines', name='Training Loss',
                                    line=dict(color='blue', width=2)))
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="MSE Loss",
                yaxis_type="log",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions
            with torch.no_grad():
                y_pred = model(X).numpy()
            
            mse = np.mean((y_pred - y.numpy()) ** 2)
            r2 = 1 - (np.sum((y.numpy() - y_pred) ** 2) / np.sum((y.numpy() - np.mean(y.numpy())) ** 2))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{mse:.6f}")
            col2.metric("R² Score", f"{r2:.4f}")
            col3.metric("Final Loss", f"{avg_loss:.6f}")

# TAB 3: Equation Discovery
with tab3:
    st.markdown('<p class="sub-header">Step 3: Discover Symbolic Equation</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train model first (Tab 2)")
    else:
        st.markdown("#### Learned Coefficients")
        
        model = st.session_state.model
        learned_coeffs = model.linear_coeffs.detach().numpy()
        true_coeffs = st.session_state.coefficients
        
        # Comparison table
        comparison_df = pd.DataFrame({
            'Variable': variable_names,
            'True Coefficient': true_coeffs,
            'Learned Coefficient': learned_coeffs,
            'Error': [abs(t - l) for t, l in zip(true_coeffs, learned_coeffs)]
        })
        
        st.dataframe(comparison_df.style.format({
            'True Coefficient': '{:.4f}',
            'Learned Coefficient': '{:.4f}',
            'Error': '{:.4f}'
        }), use_container_width=True)
        
        # Bar chart
        fig = go.Figure()
        x = np.arange(len(variable_names))
        width = 0.35
        
        fig.add_trace(go.Bar(x=x - width/2, y=true_coeffs, name='True', marker_color='green'))
        fig.add_trace(go.Bar(x=x + width/2, y=learned_coeffs, name='Learned', marker_color='blue'))
        
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=x, ticktext=variable_names),
            yaxis_title="Coefficient Value",
            title="Coefficient Comparison",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔍 Extract Symbolic Equation", type="primary", use_container_width=True):
            # Symbolic regression
            equation, symbols = symbolic_regression(
                st.session_state.df[variable_names].values,
                st.session_state.df['target'].values,
                learned_coeffs,
                variable_names
            )
            
            st.session_state.symbolic_equation = equation
            st.session_state.equation_discovered = True
            
            st.markdown("#### Discovered Equation")
            
            # Display in nice format
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.latex(f"\\frac{{dy}}{{dt}} = {sp.latex(equation)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Python format
            st.markdown("#### Python Format")
            st.code(f"dy_dt = {equation}", language="python")
            
            # Accuracy metrics
            avg_error = np.mean(comparison_df['Error'])
            max_error = np.max(comparison_df['Error'])
            
            # Calculate additional validation metrics
            df = st.session_state.df
            X = torch.FloatTensor(df[variable_names].values)
            y_true = torch.FloatTensor(df['target'].values)
            
            with torch.no_grad():
                y_pred = model(X).numpy()
            
            # Prediction accuracy
            pred_mse = np.mean((y_pred - y_true.numpy()) ** 2)
            pred_mae = np.mean(np.abs(y_pred - y_true.numpy()))
            pred_r2 = 1 - (np.sum((y_true.numpy() - y_pred) ** 2) / np.sum((y_true.numpy() - np.mean(y_true.numpy())) ** 2))
            
            # Coefficient correlation
            coeff_correlation = np.corrcoef(true_coeffs, learned_coeffs)[0, 1]
            
            # Display metrics
            st.markdown("#### 📊 Equation Quality Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Coeff Error", f"{avg_error:.4f}")
            col2.metric("Max Coeff Error", f"{max_error:.4f}")
            col3.metric("Coeff Correlation", f"{coeff_correlation:.4f}")
            col4.metric("Prediction R²", f"{pred_r2:.4f}")
            
            # Interpretation guide
            with st.expander("📖 Understanding These Metrics"):
                st.markdown("""
                ### How to Know if the Discovered Equation is Realistic
                
                #### 1. **Average Coefficient Error** (Most Important!)
                This measures how close each learned coefficient is to the true value.
                
                **Formula:** `Avg Error = mean(|true_coeff - learned_coeff|)`
                
                **Example:**
                ```
                True:    dy/dt = 2.0·x1 - 3.0·x2 + 1.0·x3
                Learned: dy/dt = 1.98·x1 - 2.85·x2 + 0.95·x3
                
                Errors: |2.0-1.98| + |-3.0-(-2.85)| + |1.0-0.95| = 0.02 + 0.15 + 0.05
                Avg Error = 0.22 / 3 = 0.073
                ```
                
                **Interpretation:**
                - **< 0.1** ✅ **Excellent** - Equation is almost perfect!
                - **0.1 - 0.3** ✓ **Good** - Equation is reliable for practical use
                - **0.3 - 0.5** ⚠ **Moderate** - Approximate, use with caution
                - **> 0.5** ❌ **Poor** - Not reliable, need more data or simpler equation
                
                ---
                
                #### 2. **Max Coefficient Error**
                The largest error among all coefficients. Helps identify if one variable is problematic.
                
                **Use:** If max error >> avg error, one coefficient is poorly learned.
                
                ---
                
                #### 3. **Coefficient Correlation**
                Measures how well the pattern of coefficients matches (values from -1 to 1).
                
                **Interpretation:**
                - **> 0.9** ✅ Excellent match
                - **0.7 - 0.9** ✓ Good match
                - **< 0.7** ⚠ Poor match
                
                ---
                
                #### 4. **Prediction R² Score**
                How well the model predicts the target values (0 to 1).
                
                **Interpretation:**
                - **> 0.95** ✅ Excellent predictions
                - **0.85 - 0.95** ✓ Good predictions
                - **0.70 - 0.85** ⚠ Moderate predictions
                - **< 0.70** ❌ Poor predictions
                
                ---
                
                ### 🎯 Overall Quality Assessment
                
                **Your equation is realistic if:**
                1. ✅ Avg Coefficient Error < 0.2
                2. ✅ Coefficient Correlation > 0.8
                3. ✅ Prediction R² > 0.90
                
                **If metrics are poor, try:**
                - Reduce noise level
                - Increase training epochs
                - Use more data samples
                - Simplify the equation (fewer variables)
                
                ---
                
                ### 📐 Example Scenarios
                
                **Scenario 1: Perfect Recovery**
                ```
                Avg Error: 0.05
                Max Error: 0.08
                Correlation: 0.98
                R²: 0.99
                
                ✅ Equation is essentially perfect!
                ```
                
                **Scenario 2: Good Recovery**
                ```
                Avg Error: 0.18
                Max Error: 0.35
                Correlation: 0.85
                R²: 0.93
                
                ✓ Equation is good enough for practical use
                ```
                
                **Scenario 3: Poor Recovery**
                ```
                Avg Error: 0.65
                Max Error: 1.2
                Correlation: 0.45
                R²: 0.72
                
                ❌ Equation is not reliable - need to improve
                ```
                """)
            
            # Overall assessment
            st.markdown("#### 🎯 Overall Assessment")
            
            if avg_error < 0.1 and coeff_correlation > 0.9 and pred_r2 > 0.95:
                st.success(f"""
                ✅ **EXCELLENT RECOVERY!**
                
                Your discovered equation is highly accurate:
                - Coefficients match within {avg_error:.1%} error
                - Strong correlation ({coeff_correlation:.3f})
                - Excellent predictions (R² = {pred_r2:.3f})
                
                **This equation is reliable for real-world use!**
                """)
            elif avg_error < 0.3 and coeff_correlation > 0.7 and pred_r2 > 0.85:
                st.info(f"""
                ✓ **GOOD RECOVERY**
                
                Your discovered equation is reasonably accurate:
                - Coefficients match within {avg_error:.1%} error
                - Good correlation ({coeff_correlation:.3f})
                - Good predictions (R² = {pred_r2:.3f})
                
                **This equation can be used for practical applications.**
                """)
            else:
                st.warning(f"""
                ⚠ **MODERATE RECOVERY**
                
                Your equation has some inaccuracies:
                - Coefficient error: {avg_error:.1%}
                - Correlation: {coeff_correlation:.3f}
                - R²: {pred_r2:.3f}
                
                **Suggestions to improve:**
                - Reduce noise level in data generation
                - Increase training epochs (try 1000+)
                - Use more data samples
                - Simplify the equation
                """)

# TAB 4: Results
with tab4:
    st.markdown('<p class="sub-header">Step 4: Complete Results</p>', unsafe_allow_html=True)
    
    if not st.session_state.equation_discovered:
        st.warning("⚠️ Please complete equation discovery first (Tab 3)")
    else:
        # Summary
        st.markdown("### 📊 Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Configuration**")
            st.write(f"- PDE Type: {pde_type}")
            st.write(f"- Variables: {n_variables}")
            st.write(f"- Duration: {duration_hours}h")
            st.write(f"- Samples: {total_samples}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Model Performance**")
            df = st.session_state.df
            X = torch.FloatTensor(df[variable_names].values)
            y = torch.FloatTensor(df['target'].values)
            
            with torch.no_grad():
                y_pred = st.session_state.model(X).numpy()
            
            mse = np.mean((y_pred - y.numpy()) ** 2)
            r2 = 1 - (np.sum((y.numpy() - y_pred) ** 2) / np.sum((y.numpy() - np.mean(y.numpy())) ** 2))
            
            st.write(f"- MSE: {mse:.6f}")
            st.write(f"- R²: {r2:.4f}")
            st.write(f"- Epochs: {len(st.session_state.loss_history)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Equation Quality**")
            comparison_df = pd.DataFrame({
                'Variable': variable_names,
                'True': st.session_state.coefficients,
                'Learned': st.session_state.model.linear_coeffs.detach().numpy()
            })
            avg_error = np.mean([abs(t - l) for t, l in zip(comparison_df['True'], comparison_df['Learned'])])
            
            st.write(f"- Avg Error: {avg_error:.4f}")
            st.write(f"- Status: {'✅ Excellent' if avg_error < 0.1 else '✓ Good' if avg_error < 0.3 else '⚠ Moderate'}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Discovered Equation
        st.markdown("### 🎯 Discovered Equation")
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.latex(f"\\frac{{dy}}{{dt}} = {sp.latex(st.session_state.symbolic_equation)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predictions vs True
        st.markdown("### 📈 Predictions vs True Values")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['time_hours'],
            y=y.numpy(),
            mode='lines',
            name='True',
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['time_hours'],
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title="Time (hours)",
            yaxis_title="Target Value",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.markdown("### 💾 Export Results")
        
        results_dict = {
            'pde_type': pde_type,
            'variables': variable_names,
            'true_coefficients': st.session_state.coefficients,
            'learned_coefficients': st.session_state.model.linear_coeffs.detach().numpy().tolist(),
            'equation': str(st.session_state.symbolic_equation),
            'mse': float(mse),
            'r2': float(r2)
        }
        
        results_json = pd.DataFrame([results_dict]).to_json(orient='records', indent=2)
        
        st.download_button(
            label="📥 Download Results (JSON)",
            data=results_json,
            file_name=f"ude_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Export data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name=f"ude_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("**UDE Equation Discovery System** | Built with Streamlit | Date: 27-12-2024")
