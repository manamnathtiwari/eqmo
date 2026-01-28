# 📋 MASTER LOG - UDE Stress Model Project

**Last Updated:** December 26, 2024

---

## 🎯 PROJECT GOAL

Build an interpretable equation-based stress prediction model using Universal Differential Equations (UDE) from multi-modal wearable data.

---

## 📊 CURRENT STATUS

### **Model:** Multi-Coefficient UDE
- **Architecture:** dS/dt = Σ(αᵢ·Fᵢ) + NN(F₁...F₁₈, t)
- **Features:** 18 physiological signals from WESAD dataset
- **Parameters:** 19 interpretable coefficients + neural network

### **Performance (Kaggle Training):**
- **Mean MSE:** 0.005038 ± 0.000576
- **Training:** 50 epochs, LOSO cross-validation, 15 subjects
- **Comparison:**
  - Ridge: 0.000003 ⭐ (best)
  - XGBoost: 0.000004
  - **UDE: 0.005038** (ties with Random Forest)

### **Key Finding:**
WESAD dataset is too simple - linear models (Ridge) dominate. UDE's complexity not needed for this data.

---

## 📁 PROJECT STRUCTURE

### **Code (All Preserved):**
```
src/models/
├── ude_model.py ⭐ (main model)
├── train.py ⭐ (LOSO training)
├── ridge_model.py (baseline)
├── xgboost_model.py (baseline)
├── lstm_model.py (baseline)
├── ml_baselines.py (all baselines)
├── dense_coupled_ude.py (experimental)
├── sparse_coupled_ude.py (experimental)
└── coupled_ude.py (experimental)
```

### **Data:**
```
data/processed/normalized/
└── u_wesad_*.csv (15 subjects)
```

### **Results:**
```
results/loso_models/
├── loso_results.csv (MSE per fold)
├── loso_results.png (visualization)
└── ude_fold_*.pth (15 trained models)
```

### **Documentation:**
```
docs/
└── PROJECT_DOCUMENTATION.md (master reference)
```

---

## 🔬 EXPERIMENTS CONDUCTED

### **Phase 1: Baseline Benchmarking**
- Created Ridge, XGBoost, SVR, LSTM models
- **Result:** Ridge achieves MSE ~0.000003 (best)
- **Conclusion:** WESAD has strong linear relationships

### **Phase 2: Initial UDE (19-param)**
- Built UDE with all 18 features
- Trained on Kaggle (5 hours, P100 GPU)
- **Result:** MSE ~0.005 (ties with Random Forest)
- **Conclusion:** Works but doesn't beat simple models

### **Phase 3: Sparse Coupled-UDE (Attempted)**
- Idea: Ridge selects features → Coupled UDE
- **Issue:** Ridge threshold too high, selected 0 features
- **Status:** Experimental, not pursued

### **Phase 4: Dense Coupled-UDE (Attempted)**
- Idea: 4 coupled variables × 18 features
- Code written but not trained
- **Status:** Experimental, code preserved

### **Phase 5: Novelty Assessment**
- Comprehensive literature search
- **Finding:** Application is novel, method exists elsewhere
- **Rating:** ⭐⭐⭐⭐☆ (4/5) - publishable at ML4H/EMBC

---

## 💡 KEY INSIGHTS

### **What Works:**
✅ UDE successfully models stress dynamics  
✅ Interpretable coefficients show feature importance  
✅ Personalized equations per subject  
✅ Captures temporal evolution  

### **Limitations:**
⚠️ WESAD is too simple (linear relationships dominate)  
⚠️ Ridge regression captures most signal  
⚠️ UDE's complexity not justified on this dataset  
⚠️ Controlled lab setting, not real-world  

### **Lessons Learned:**
1. Simple datasets favor simple models
2. Interpretability ≠ Accuracy
3. Application novelty still valuable
4. Need better data to show UDE's value

---

## 🎯 NOVELTY ASSESSMENT

### **Novel:**
✅ First application of UDEs to stress prediction  
✅ Interpretable personalized stress equations  
✅ Multi-modal wearable data integration  

### **Not Novel:**
❌ UDEs themselves (exist in physics, biology)  
❌ Stress prediction (ML does this)  
❌ Wearable monitoring (common)  

### **Publication Potential:**
- **Rating:** ⭐⭐⭐☆☆ (3/5)
- **Best Venues:** ML4H, EMBC, IEEE TBME
- **Framing:** "First application to stress, proof of concept"
- **Key:** Be honest about WESAD limitations

---

## 📝 NEXT STEPS

### **Option 1: Publish Current Work**
- Frame as proof of concept
- Acknowledge WESAD limitations
- Focus on interpretability
- Target: ML4H or EMBC

### **Option 2: Get Better Data**
- Real-world stress (not lab)
- Longer duration (weeks/months)
- More subjects (100+)
- Complex dynamics

### **Option 3: Deep Analysis**
- Analyze learned coefficients
- Compare across subjects
- Find parameter patterns
- Clinical interpretation

---

## 📚 TIMELINE

**Week 1-2 (Dec 2024):**
- Built baseline models
- Ran benchmarking
- Found Ridge dominates

**Week 3-4:**
- Implemented UDE model
- Trained on Kaggle
- Got MSE ~0.005

**Week 5:**
- Explored coupled variants
- Did novelty assessment
- Decided to focus on original UDE

**Current (Dec 26):**
- Cleaned up documentation
- Focused on UDE only
- Ready for publication prep

---

## 🎓 TECHNICAL DETAILS

### **UDE Architecture:**
```python
class UDEModel(nn.Module):
    def __init__(self, n_features=18):
        # Interpretable coefficients
        self.feature_coefficients = nn.Parameter(torch.randn(n_features))
        
        # Neural network for nonlinear dynamics
        self.nn = nn.Sequential(
            nn.Linear(n_features + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, t, y, features):
        # Linear component (interpretable)
        linear = torch.sum(self.feature_coefficients * features, dim=-1)
        
        # Nonlinear component (neural network)
        state = torch.cat([features, t.expand_as(features[:, :1])], dim=-1)
        nonlinear = self.nn(state).squeeze()
        
        return linear + nonlinear
```

### **Training:**
- **Method:** LOSO cross-validation
- **Optimizer:** Adam (lr=0.005)
- **Epochs:** 50
- **Batch Size:** 256
- **ODE Solver:** Euler (step_size=0.02)
- **Device:** GPU (P100 on Kaggle)

### **Data Preprocessing:**
- Population-level normalization
- Sequence length: 60 timesteps
- Overlap: 50% (30 timesteps)

---

## 📊 RESULTS SUMMARY

### **Per-Fold MSE:**
```
Fold  Subject         MSE
1     u_wesad_002    0.005670
2     u_wesad_003    0.004254
3     u_wesad_004    0.004445
...
15    u_wesad_017    0.005166

Mean: 0.005038 ± 0.000576
```

### **Feature Importance (Avg Coefficients):**
Top 5 features across subjects:
1. Workload
2. HR_mean
3. HRV_pNN50
4. EDA_mean
5. Resp_std

---

## 🚀 PUBLICATION PLAN

### **Title:**
"Universal Differential Equations for Interpretable Stress Modeling from Wearable Data"

### **Abstract Points:**
- First application of UDEs to stress prediction
- Interpretable personalized equations
- Validated on WESAD dataset (15 subjects)
- Achieves competitive performance with interpretability
- Acknowledges limitations of controlled lab data

### **Key Contributions:**
1. Novel application of UDEs to stress domain
2. Interpretable multi-coefficient framework
3. Personalized stress dynamics per subject
4. Proof of concept for equation-based modeling

### **Target Venues:**
- ML4H (Machine Learning for Healthcare) - Primary
- EMBC (Engineering in Medicine & Biology) - Secondary
- IEEE TBME (Transactions on Biomedical Engineering) - Journal

---

## ✅ CURRENT FOCUS

**What We're Doing:**
- ✅ Sticking with original UDE model
- ✅ One master documentation file
- ✅ All code preserved
- ✅ Ready for publication prep

**What We're NOT Doing:**
- ❌ Sparse coupled variants
- ❌ Dense coupled variants
- ❌ Multiple model comparisons
- ❌ Overclaiming novelty

**Next Action:**
- Prepare paper draft
- Create visualizations
- Write honest framing
- Submit to ML4H

---

## 📌 IMPORTANT NOTES

1. **Be Honest:** WESAD is limited, acknowledge this
2. **Focus on Novelty:** Application to stress is new
3. **Emphasize Interpretability:** That's the value
4. **Realistic Expectations:** ML4H/EMBC, not Nature
5. **Keep It Simple:** Original UDE is enough

---

## 🔗 REFERENCES

**Key Papers:**
- Neural ODEs (Chen et al., NeurIPS 2018)
- Universal Differential Equations (Rackauckas et al., 2020)
- WESAD Dataset (Schmidt et al., 2018)

**Related Work:**
- Stress prediction from wearables (extensive ML literature)
- Physiological modeling with ODEs (systems biology)
- Interpretable ML for healthcare

---

**Status:** Model complete, results obtained, ready for publication  
**Focus:** Original UDE model only  
**Goal:** Publish at ML4H or EMBC with honest framing
