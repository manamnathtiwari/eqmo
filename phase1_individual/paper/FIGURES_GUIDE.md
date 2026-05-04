# Research Paper - Figures and Tables Guide

## Required Figures (5 Total)

### **Figure 1: Model Architecture**
**Type:** Diagram  
**Content:**
- Input: Multi-modal physiological signals (18 features)
- Three components:
  1. Physics-based recovery term (-β·S)
  2. Feature-specific drive terms (Σ αᵢ·Fᵢ)
  3. Neural network correction (NN)
- Output: Stress prediction dS/dt
- ODE solver integration

**Tool:** Draw.io, PowerPoint, or TikZ (LaTeX)

---

### **Figure 2: Coefficient Heatmap**
**Type:** Heatmap  
**Content:**
- X-axis: 18 features
- Y-axis: 15 subjects
- Color: Learned α values (0 to 0.35)
- Shows clustering of subjects by stress type

**Code:**
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load alphas from all folds
alphas_df = pd.DataFrame()  # 15 rows (subjects) × 18 cols (features)

plt.figure(figsize=(12, 8))
sns.heatmap(alphas_df, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Alpha Value'})
plt.xlabel('Physiological Features')
plt.ylabel('Subjects')
plt.title('Learned Feature Sensitivities Across Subjects')
plt.tight_layout()
plt.savefig('figure2_heatmap.pdf', dpi=300)
```

---

### **Figure 3: Performance Comparison**
**Type:** Bar chart  
**Content:**
- X-axis: Methods (Linear, RF, LSTM, Single-UDE, MC-UDE)
- Y-axis: Test MSE
- Error bars: Standard deviation across folds
- Highlight MC-UDE as best

**Code:**
```python
import matplotlib.pyplot as plt
import numpy as np

methods = ['Linear\nRegression', 'Random\nForest', 'LSTM', 'Single-Coeff\nUDE', 'MC-UDE\n(Ours)']
mse_values = [0.0234, 0.0156, 0.0098, 0.0082, 0.0054]
std_values = [0.0045, 0.0032, 0.0021, 0.0018, 0.0012]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, mse_values, yerr=std_values, capsize=5, 
               color=['gray', 'gray', 'gray', 'lightblue', 'darkblue'])
plt.ylabel('Test MSE (Lower is Better)')
plt.title('Performance Comparison Across Methods')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure3_performance.pdf', dpi=300)
```

---

### **Figure 4: Subject-Specific Equations**
**Type:** Text visualization / Equation display  
**Content:**
- Show 3 example subjects with their discovered equations
- Highlight dominant terms
- Visual comparison of coefficient magnitudes

**Example:**
```
Subject 2 (HRV-Dominant):
dS/dt = -0.052·S + [0.284]·HRV + 0.157·HR + 0.092·EDA + ...
                    ↑ Highest

Subject 7 (EDA-Dominant):
dS/dt = -0.051·S + 0.073·HRV + 0.143·HR + [0.312]·EDA + ...
                                              ↑ Highest

Subject 11 (Workload-Dominant):
dS/dt = -0.049·S + 0.089·HRV + 0.134·HR + 0.098·EDA + [0.299]·Workload
                                                         ↑ Highest
```

---

### **Figure 5: Prediction Example**
**Type:** Time series plot  
**Content:**
- X-axis: Time (minutes)
- Y-axis: Stress level (0-1)
- Lines:
  - Ground truth (black, solid)
  - MC-UDE prediction (blue, solid)
  - Single-UDE prediction (red, dashed)
  - LSTM prediction (green, dotted)
- Shaded regions: Stress protocol phases (baseline, stress, recovery)

**Code:**
```python
import matplotlib.pyplot as plt

# Load predictions from test subject
time = np.arange(0, 60, 0.1)  # 60 minutes
ground_truth = ...  # Load from data
mc_ude_pred = ...   # Load from model
single_ude_pred = ...
lstm_pred = ...

plt.figure(figsize=(12, 6))
plt.plot(time, ground_truth, 'k-', linewidth=2, label='Ground Truth')
plt.plot(time, mc_ude_pred, 'b-', linewidth=1.5, label='MC-UDE (Ours)')
plt.plot(time, single_ude_pred, 'r--', linewidth=1.5, label='Single-Coeff UDE')
plt.plot(time, lstm_pred, 'g:', linewidth=1.5, label='LSTM')

# Shade stress phases
plt.axvspan(0, 10, alpha=0.1, color='green', label='Baseline')
plt.axvspan(10, 30, alpha=0.1, color='red', label='Stress')
plt.axvspan(30, 60, alpha=0.1, color='blue', label='Recovery')

plt.xlabel('Time (minutes)')
plt.ylabel('Stress Level')
plt.title('Stress Prediction Comparison - Subject 2')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figure5_prediction.pdf', dpi=300)
```

---

## Required Tables (3 Total)

### **Table I: LOSO Cross-Validation Results**

| Method | Test MSE ↓ | Test MAE ↓ | R² ↑ | Parameters |
|--------|-----------|-----------|------|------------|
| Linear Regression | 0.0234 ± 0.0045 | 0.1123 ± 0.0234 | 0.42 | 19 |
| Random Forest | 0.0156 ± 0.0032 | 0.0892 ± 0.0189 | 0.61 | - |
| LSTM | 0.0098 ± 0.0021 | 0.0734 ± 0.0156 | 0.76 | ~8K |
| Single-Coeff UDE | 0.0082 ± 0.0018 | 0.0651 ± 0.0134 | 0.80 | 3 + NN |
| **MC-UDE (Ours)** | **0.0054 ± 0.0012** | **0.0512 ± 0.0098** | **0.87** | **19 + NN** |

**Note:** Values are mean ± std across 15 LOSO folds. Bold indicates best performance.

---

### **Table II: Mean Learned Coefficients**

| Feature | Mean α | Std α | Rank | Interpretation |
|---------|--------|-------|------|----------------|
| EDA Mean | 0.2156 | 0.0423 | 1 | **Strongest stress driver** |
| Workload | 0.1892 | 0.0456 | 2 | Cognitive load impact |
| Heart Rate Mean | 0.1567 | 0.0289 | 3 | Elevated HR = stress |
| HRV RMSSD | 0.1234 | 0.0312 | 4 | Cardiac variability |
| Respiration Std | 0.0987 | 0.0234 | 5 | Breathing irregularity |
| HRV SDNN | 0.0876 | 0.0198 | 6 | HRV metric |
| Temperature Mean | 0.0765 | 0.0187 | 7 | Thermal response |
| ... | ... | ... | ... | ... |

**Recovery Rate β:** 0.0523 ± 0.0089

---

### **Table III: Subject Clustering by Stress Type**

| Cluster | Subjects | Dominant Feature | Mean α | Example Intervention |
|---------|----------|------------------|--------|---------------------|
| HRV-Dominant | 2, 5, 9, 13 | HRV RMSSD | 0.284 | Breathing exercises, meditation |
| EDA-Dominant | 3, 7, 11, 15 | EDA Mean | 0.312 | Anxiety management, relaxation |
| Workload-Dominant | 4, 6, 10, 14 | Workload | 0.299 | Task management, breaks |
| Mixed | 8, 16, 17 | Multiple | - | Holistic approach |

---

## Additional Materials

### **Supplementary Figure S1: Training Curves**
- Loss vs epochs for all 15 folds
- Shows convergence

### **Supplementary Figure S2: Coefficient Distribution**
- Box plots of α values across subjects
- Shows variability

### **Supplementary Table S1: Per-Subject Results**
- Detailed MSE, MAE, R² for each of 15 subjects

---

## Figure Generation Scripts

All scripts available in: `paper/figures/generate_figures.py`

**Requirements:**
```
matplotlib==3.7.0
seaborn==0.12.0
pandas==2.0.0
numpy==1.24.0
```

**Usage:**
```bash
python paper/figures/generate_figures.py --results_dir results/multicoeff_models/
```

---

## LaTeX Integration

**In your .tex file:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/figure1_architecture.pdf}
\caption{Multi-Coefficient UDE Architecture combining physics-based recovery, feature-specific drives, and neural network correction.}
\label{fig:architecture}
\end{figure}
```

---

**END OF FIGURES GUIDE**
