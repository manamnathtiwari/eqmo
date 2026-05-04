# 🚀 Kaggle Deployment Guide — MC-UDE

## Overview

Three scripts to run sequentially on Kaggle GPU notebooks.

| Script | Purpose | Runtime | GPU? |
|--------|---------|---------|------|
| `01_train.py` | Train 15 L1-sparse MC-UDE models (LOSO) | ~4-6 hours | ✅ Yes |
| `02_evaluate.py` | Fair trajectory comparison vs baselines | ~1-2 hours | ✅ Yes |
| `03_analyze.py` | Sparse profiles + publication figures | ~5 min | ❌ No |

---

## Step 1: Upload Data

1. Go to **Kaggle → Datasets → New Dataset**
2. Upload the `data/processed/normalized/` folder containing all `u_wesad_*.csv` files
3. Name the dataset: `wesad-normalized`
4. The CSV files should be at: `/kaggle/input/wesad-normalized/normalized/u_wesad_*.csv`

> **If your path is different**, edit the `CONFIG['DATA_DIR']` at the top of each script.

---

## Step 2: Train Models (01_train.py)

1. Create a new **Kaggle Notebook**
2. Enable **GPU** (Settings → Accelerator → GPU P100)
3. Add your `wesad-normalized` dataset as input
4. Paste the contents of `01_train.py` into a code cell
5. Run the cell

**What it does:**
- Trains 15 MC-UDE models using Leave-One-Subject-Out cross-validation
- Uses L1 regularization (λ=0.001) for automatic feature selection
- Uses physics constraints on recovery rate (β bounded to [0.01, 5.0])
- Saves checkpoints after each fold (survives Kaggle timeout)
- Outputs to `/kaggle/working/mc_ude_results/`

**Checkpointing:** If Kaggle times out after 9 hours, just restart. It automatically skips completed folds.

**Expected output:**
- 15 model files: `mcude_fold_1.pth` ... `mcude_fold_15.pth`
- 15 profile files: `profile_fold_1.json` ... `profile_fold_15.json`
- Summary: `loso_results.csv`

---

## Step 3: Download & Re-upload Models

1. After `01_train.py` completes, download the `mc_ude_results/` folder from `/kaggle/working/`
2. Create a new Kaggle Dataset: `mc-ude-models`
3. Upload the `mc_ude_results/` folder
4. Models will be at: `/kaggle/input/mc-ude-models/mc_ude_results/`

---

## Step 4: Run Fair Comparison (02_evaluate.py)

1. Create a new **Kaggle Notebook** (GPU optional)
2. Add BOTH datasets as input:
   - `wesad-normalized` (data)
   - `mc-ude-models` (trained models)
3. Paste `02_evaluate.py` into a code cell
4. Run

**What it does:**
- All models predict the same 60-step trajectory: S(0) → S(60)
- NO model sees past stress values as input features (fair!)
- UDE: Natural ODE integration
- Ridge/RF: 1-step training, autoregressive unrolling at test time
- Outputs comparison CSV and plot

---

## Step 5: Generate Publication Figures (03_analyze.py)

1. Same notebook as Step 4 (or new one)
2. Add `mc-ude-models` dataset
3. Paste `03_analyze.py` and run

**Produces:**
- `fig1_alpha_heatmap.png` — Per-subject feature sensitivity heatmap
- `fig2_recovery_and_importance.png` — Recovery rates + top predictors
- `fig3_sparsity_map.png` — L1 feature selection visualization
- `subject_profiles.csv` — Stress responder classification
- `learned_equations.txt` — Human-readable equations per subject

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: No CSV files` | Check `CONFIG['DATA_DIR']` path matches your Kaggle dataset |
| `KeyError: column not found` | Your CSV may have different column names. Check `FEATURE_COLUMNS` list |
| Kaggle timeout (9 hours) | Restart notebook, it auto-resumes from last checkpoint |
| `ModuleNotFoundError: torchdiffeq` | Add `!pip install torchdiffeq` as the first cell |
| Out of GPU memory | Reduce `BATCH_SIZE` from 32 to 16 |

---

## Dependencies

Add this as the first cell of every notebook:

```python
!pip install torchdiffeq
```

All other dependencies (torch, pandas, numpy, sklearn, matplotlib, seaborn) are pre-installed on Kaggle.
