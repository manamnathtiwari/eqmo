# 🎯 UDE Stress Model - Project Documentation

## 📊 PROJECT OVERVIEW

**Goal:** Interpretable equation-based stress prediction using Universal Differential Equations (UDE)

**Dataset:** WESAD (15 subjects, multi-modal wearable data, 18 features)

**Approach:** Neural ODE with interpretable parameters for personalized stress modeling

---

## 🏗️ CURRENT MODEL: Multi-Coefficient UDE

### **Architecture:**
```
dS/dt = Σ(αᵢ·Fᵢ) + NN(F₁, F₂, ..., F₁₈, t)

Where:
- S = Stress (target)
- Fᵢ = 18 physiological features
- αᵢ = Interpretable coefficients (learned)
- NN = Neural network for nonlinear dynamics
```

### **Features (18):**
- HRV: rmssd, sdnn, pnn50, lf_hf
- Heart Rate: mean, std
- EDA: mean, std, peaks
- Temperature: mean, std
- Respiration: mean, std
- Activity: mean, std
- EMG: mean, std
- Workload

### **Parameters:**
- 18 feature coefficients (interpretable)
- Neural network weights (captures nonlinearities)
- Total: ~19 interpretable parameters

---

## 📈 RESULTS (From Kaggle Training)

### **Performance:**
- **Mean MSE:** 0.005038 ± 0.000576
- **Training:** 50 epochs, LOSO cross-validation
- **Device:** P100 GPU
- **Time:** ~5 hours for 15 folds

### **Comparison to Baselines:**
- Ridge: 0.000003 (best)
- XGBoost: 0.000004
- UDE: 0.005038
- Random Forest: ~0.005

**Conclusion:** UDE ties with RF but doesn't beat simple linear models on WESAD

---

## 💡 KEY INSIGHTS

### **What Works:**
- ✅ UDE successfully models stress dynamics
- ✅ Interpretable coefficients show feature importance
- ✅ Personalized equations per subject
- ✅ Captures temporal evolution

### **Limitations:**
- ⚠️ WESAD is too simple (linear relationships dominate)
- ⚠️ Ridge regression already captures most signal
- ⚠️ UDE's complexity not needed for this dataset

### **Why WESAD Results Are Limited:**
1. Controlled lab setting (not real-world)
2. Short duration (limited dynamics)
3. Linear relationships dominate
4. Small sample size (15 subjects)

---

## 🎯 NOVELTY ASSESSMENT

### **What's Novel:**
- ✅ Application of UDEs to stress modeling (first)
- ✅ Interpretable parameters from wearable data
- ✅ Personalized stress equations

### **What's Not Novel:**
- ❌ UDEs themselves (exist in other domains)
- ❌ Stress prediction (ML does this well)
- ❌ Wearable-based stress monitoring (common)

### **Publication Potential:**
- **Rating:** ⭐⭐⭐☆☆ (3/5)
- **Best Venues:** ML4H, EMBC, IEEE TBME
- **Framing:** "First application of UDEs to stress, proof of concept"

---

## 📁 CODE STRUCTURE

### **Models:**
```
src/models/
├── ude_model.py              # Main UDE implementation
├── train.py                  # LOSO training script
├── ridge_model.py            # Ridge baseline
├── xgboost_model.py          # XGBoost baseline
├── lstm_model.py             # LSTM baseline
├── ml_baselines.py           # All baselines together
├── dense_coupled_ude.py      # Coupled version (experimental)
├── sparse_coupled_ude.py     # Sparse coupled (experimental)
└── coupled_ude.py            # 2-var coupled (experimental)
```

### **Data:**
```
data/processed/normalized/
└── u_wesad_*.csv             # 15 subject files
```

### **Results:**
```
results/loso_models/
├── loso_results.csv          # MSE per fold
├── loso_results.png          # Visualization
└── ude_fold_*.pth            # Trained models (15)
```

---

## 🚀 HOW TO RUN

### **Local Testing:**
```bash
cd "c:\Users\Manamnath tiwari\OneDrive\Desktop\Equation Model\burnout_project"
python src/models/train.py
```

### **Kaggle Training:**
1. Upload `ude_model.py` and `train.py` to dataset
2. Create notebook with cells from previous runs
3. Run training (5 hours on P100)
4. Download results

---

## 📊 NEXT STEPS

### **Option 1: Accept Current Results**
- Use UDE as proof of concept
- Frame as "first application to stress"
- Publish at ML4H or EMBC
- Acknowledge WESAD limitations

### **Option 2: Get Better Data**
- Real-world stress data (not lab)
- Longer duration (weeks/months)
- More subjects (100+)
- More complex dynamics

### **Option 3: Focus on Interpretability**
- Analyze learned coefficients
- Compare across subjects
- Find patterns in parameters
- Clinical interpretation

---

## 📝 PUBLICATION STRATEGY

### **Title:**
"Universal Differential Equations for Interpretable Stress Modeling from Wearable Data"

### **Key Contributions:**
1. First application of UDEs to stress prediction
2. Interpretable personalized stress equations
3. Proof of concept on WESAD dataset

### **Honest Framing:**
- Acknowledge WESAD limitations
- Position as methodological contribution
- Focus on interpretability over accuracy
- Suggest future work with better data

### **Target Venues:**
- ML4H (Machine Learning for Healthcare)
- EMBC (Engineering in Medicine & Biology)
- IEEE TBME (Transactions on Biomedical Engineering)

---

## 🎓 LESSONS LEARNED

1. **Simple datasets favor simple models** - WESAD is too linear for UDEs to shine
2. **Interpretability ≠ Accuracy** - UDE gives equations but not better predictions
3. **Application novelty matters** - First use in domain is still valuable
4. **Be honest about limitations** - Acknowledge when complex methods aren't needed

---

## 📚 REFERENCES

**Key Papers:**
- Neural ODEs (Chen et al., NeurIPS 2018)
- Universal Differential Equations (Rackauckas et al., 2020)
- WESAD Dataset (Schmidt et al., 2018)

**Related Work:**
- Stress prediction from wearables (extensive ML literature)
- Physiological modeling with ODEs (systems biology)
- Personalized health monitoring (digital health)

---

## ✅ FINAL STATUS

**What We Have:**
- ✅ Working UDE model
- ✅ Trained on WESAD (15 subjects)
- ✅ Interpretable parameters
- ✅ Publishable results (with honest framing)

**What We Don't Have:**
- ❌ Better accuracy than Ridge
- ❌ Revolutionary novelty
- ❌ Real-world validation

**Recommendation:**
- Focus on interpretability
- Frame as proof of concept
- Publish at ML4H/EMBC
- Plan better study with real-world data

---

**Last Updated:** December 26, 2024  
**Status:** Model complete, ready for publication with honest framing
