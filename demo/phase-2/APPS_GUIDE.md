# Demo Apps - Two Options

## 🎯 You Now Have TWO Apps:

### 1. **demo_app.py** - Educational Demo
**Purpose:** Learn how UDE works  
**Data:** Synthetic (generated)  
**Models:** Trains new models each time  
**Use:** Understanding the concept  

**Run:**
```bash
streamlit run demo/phase-2/demo_app.py
```

---

### 2. **real_model_viewer.py** - Your Trained Models ⭐
**Purpose:** View your actual WESAD results  
**Data:** Real WESAD dataset  
**Models:** Uses your pre-trained models  
**Use:** See real stress predictions  

**Run:**
```bash
streamlit run demo/phase-2/real_model_viewer.py
```

---

## 📊 What Each App Shows:

### **demo_app.py** (Educational):
- ✅ Choose 2-6 variables
- ✅ Define custom equations
- ✅ Generate synthetic data (6h, 12h, 24h)
- ✅ Train UDE from scratch
- ✅ See equation recovery
- ✅ Learn how UDE works

**Best for:** Demonstrations, learning, experiments

---

### **real_model_viewer.py** (Real Results) ⭐:
- ✅ Load your 15 trained models
- ✅ View learned stress equations
- ✅ See real predictions on WESAD data
- ✅ Compare all subjects
- ✅ Analyze feature importance
- ✅ Show actual performance (MSE, R²)

**Best for:** Research, analysis, presentations

---

## 🚀 Quick Start:

### For Learning:
```bash
streamlit run demo/phase-2/demo_app.py
```
Play with different equations and see UDE learn them!

### For Real Results:
```bash
streamlit run demo/phase-2/real_model_viewer.py
```
View your actual trained WESAD models!

---

## 📁 Requirements:

### **demo_app.py**:
- Just needs: `streamlit`, `torch`, `numpy`, `pandas`, `plotly`, `sympy`
- No data files needed (generates synthetic)

### **real_model_viewer.py**:
- Needs: `streamlit`, `torch`, `numpy`, `pandas`, `plotly`
- **Requires:**
  - `results/loso_models/` folder with `.pth` files
  - `data/processed/normalized/` folder with WESAD CSVs
  - `results/loso_models/loso_results.csv`

---

## ✅ Which App to Use When:

| Scenario | Use This App |
|----------|--------------|
| Demo for tomorrow's presentation | `demo_app.py` |
| Show your actual results | `real_model_viewer.py` ⭐ |
| Explain how UDE works | `demo_app.py` |
| Analyze WESAD performance | `real_model_viewer.py` ⭐ |
| Test different equations | `demo_app.py` |
| See learned stress equations | `real_model_viewer.py` ⭐ |
| Quick experiment | `demo_app.py` |
| Research/Publication | `real_model_viewer.py` ⭐ |

---

## 🎯 For Tomorrow's Demo:

**Recommended Flow:**

1. **Start with `demo_app.py`** (5 min)
   - Show how UDE discovers equations
   - Use simple 3-variable example
   - Prove the concept works

2. **Switch to `real_model_viewer.py`** (5 min)
   - "Now here are the REAL results"
   - Load a trained model
   - Show learned stress equation
   - Display predictions on real data

**This shows both concept AND real application!** 🚀

---

## 📝 Notes:

- Both apps are independent
- `demo_app.py` doesn't need your trained models
- `real_model_viewer.py` ONLY works with your trained models
- Keep both - they serve different purposes!

**You're all set!** ✅
