# UDE Equation Discovery Demo
**Date:** 27-12-2024  
**Phase:** 2 - Review 1

## 🎯 Features

### 1. PDE Type Selection
- Linear ODE
- Nonlinear ODE
- Coupled System
- Custom Equation

### 2. Variable Configuration
- Choose 2-6 variables
- Custom variable names
- Define coefficients for each variable

### 3. Data Simulation
- **Duration Options:** 6 hours, 12 hours, 24 hours
- **Sampling Rate:** 10-120 samples/hour
- **Noise Control:** Adjustable noise level
- **Realistic Data:** Time-varying patterns

### 4. Equation Customization
- Set linear coefficients
- Add quadratic terms (x²)
- Add interaction terms (x·y)
- Full control over relationships

### 5. UDE Training
- Configurable epochs (100-2000)
- Learning rate selection
- Batch size control
- Real-time training progress
- Loss visualization

### 6. Symbolic Equation Discovery
- Extract learned coefficients
- Compare with true values
- Generate symbolic equation using SymPy
- LaTeX and Python formats
- Accuracy metrics

### 7. Complete Results
- Performance summary
- Equation quality assessment
- Predictions vs true values plot
- Export results (JSON)
- Export data (CSV)

## 🚀 How to Run

### Install Dependencies:
```bash
pip install streamlit torch numpy pandas plotly sympy
```

### Run Demo:
```bash
streamlit run demo/phase-2/demo_app.py
```

## 📋 Usage Flow

### Step 1: Configure (Sidebar)
1. Select PDE type
2. Choose number of variables
3. Name your variables
4. Define equation coefficients
5. Set simulation duration

### Step 2: Generate Data (Tab 1)
1. Review equation
2. Adjust noise and initial range
3. Click "Generate Data"
4. View time series plots

### Step 3: Train Model (Tab 2)
1. Set training parameters
2. Click "Train Model"
3. Watch training progress
4. View loss curve and metrics

### Step 4: Discover Equation (Tab 3)
1. Review learned coefficients
2. Compare with true values
3. Click "Extract Symbolic Equation"
4. See discovered equation in LaTeX

### Step 5: View Results (Tab 4)
1. See complete summary
2. View predictions vs true
3. Download results
4. Export data

## 🎨 Example Configurations

### Example 1: Stress Model
```
PDE Type: Linear ODE
Variables: Stress, HRV, EDA
Coefficients: [-0.2, 0.8, 0.5]
Duration: 12 hours
Equation: dS/dt = -0.2·S + 0.8·HRV + 0.5·EDA
```

### Example 2: Nonlinear System
```
PDE Type: Nonlinear ODE
Variables: x, y
Coefficients: [2.0, -3.0]
Quadratic: Yes
Interaction: Yes
Duration: 24 hours
Equation: dy/dt = 2·x - 3·y + 0.5·x² - 0.3·x·y
```

### Example 3: Multi-Variable
```
PDE Type: Custom
Variables: Var1, Var2, Var3, Var4
Coefficients: [1.5, -0.8, 0.3, -0.5]
Duration: 6 hours
```

## ✅ Expected Results

- **Data Generation:** Realistic time series with noise
- **Training:** MSE < 0.01 for linear, < 0.1 for nonlinear
- **Equation Discovery:** Average error < 0.2
- **R² Score:** > 0.95 for good recovery

## 🔧 Troubleshooting

**Issue:** Model not converging  
**Solution:** Increase epochs or adjust learning rate

**Issue:** Poor equation recovery  
**Solution:** Reduce noise level or increase samples

**Issue:** Slow training  
**Solution:** Reduce batch size or epochs

## 📊 Output Files

- `ude_results_YYYYMMDD_HHMMSS.json` - Model results
- `ude_data_YYYYMMDD_HHMMSS.csv` - Generated data

## 🎯 Demo Objectives

1. ✅ Show UDE can recover equations
2. ✅ Demonstrate customizable relationships
3. ✅ Prove scalability (6h to 24h)
4. ✅ Extract interpretable symbolic equations
5. ✅ Provide complete workflow

**Ready for Review!** 🚀
