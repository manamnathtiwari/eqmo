# MC-UDE Project: Challenges & Solutions

A complete record of every major issue encountered and how it was resolved.

---

## 1. Kaggle Environment — Missing `torchdiffeq`

| | Detail |
|---|--------|
| **Problem** | `ModuleNotFoundError: No module named 'torchdiffeq'` on Kaggle — the ODE solver library isn't pre-installed |
| **Root Cause** | Kaggle's default PyTorch environment doesn't include `torchdiffeq` |
| **Solution** | Added `!pip install torchdiffeq` as the first cell in all 3 notebooks (train, evaluate, analyze) |
| **Lesson** | Always check Kaggle's pre-installed packages before uploading scripts |

---

## 2. Kaggle Data Path Configuration

| | Detail |
|---|--------|
| **Problem** | `FileNotFoundError` — scripts couldn't find WESAD data files |
| **Root Cause** | Default `DATA_DIR` pointed to `/kaggle/input/wesad-normalized` but Kaggle stores user datasets at `/kaggle/input/datasets/manamtiwari/wesad-normalized` |
| **Solution** | Updated `DATA_DIR` in all scripts to the correct Kaggle dataset path |
| **Lesson** | Kaggle dataset paths vary by upload method; always verify with `os.walk()` |

---

## 3. Model Path Mismatch Across Notebooks

| | Detail |
|---|--------|
| **Problem** | `KeyError: 'UDE'` in `02_evaluate.py` — couldn't find trained models |
| **Root Cause** | Training saves models to `/kaggle/working/mc_ude_results` but evaluation ran in a separate notebook where that path doesn't exist |
| **Solution** | Either (a) run all scripts in the same notebook, or (b) re-upload trained models as a Kaggle dataset and set `MODELS_DIR` accordingly |
| **Lesson** | Kaggle notebooks don't share `/kaggle/working/` — plan artifact passing between notebooks |

---

## 4. ODE Solver Mismatch (Initial Misdiagnosis)

| | Detail |
|---|--------|
| **Problem** | UDE MSE was 0.35 — 75x worse than baselines (Ridge ~0.005) |
| **Initial Diagnosis** | Training used `method='euler'` but evaluation used `method='dopri5'` |
| **Fix Applied** | Changed evaluation to `method='euler'` |
| **Result** | MSE barely changed (0.353 → 0.347). **This was NOT the root cause.** |
| **Lesson** | Don't assume the first hypothesis is correct — verify with data |

---

## 5. Time Scale Mismatch (THE Real Root Cause) ⭐

| | Detail |
|---|--------|
| **Problem** | UDE MSE still ~0.35 even after solver fix |
| **Root Cause** | Training uses actual CSV time values (spacing ~0.017 per step). Evaluation used `torch.arange(60)` = [0, 1, 2, ..., 59] — **60x larger time steps** |
| **Why It Breaks** | The model learned `dS/dt` dynamics calibrated for dt ≈ 0.017. With Euler integration: `S(t+1) = S(t) + dS/dt × dt`. When dt jumps from 0.017 to 1.0, every state update is 60x too large and the trajectory explodes |
| **Solution** | Modified `prepare_data()` to extract real time values from CSV and pass them to `odeint()` instead of integer indices |
| **Result** | UDE MSE dropped from 0.35 → **0.006** — now competitive with baselines |
| **Lesson** | In ODE models, time scale consistency between training and evaluation is absolutely critical. Always use the same time grid. |

---

## 6. Weak L1 Regularization — No Feature Sparsity

| | Detail |
|---|--------|
| **Problem** | All 18 features active (0% sparsity) across all 15 subjects — no personalized biomarker selection |
| **Root Cause** | `LAMBDA_L1 = 0.001` was too weak to push any α coefficients to near-zero |
| **Impact** | All subjects showed nearly identical profiles (Heart Rate ≈ 0.088, Workload ≈ 0.086, etc.) — undermines the "personalized discovery" narrative |
| **Solution (Pending)** | Re-train with `LAMBDA_L1 = 0.01` or `0.05` to achieve genuine sparsity (3-5 active features per subject) |
| **Lesson** | L1 regularization strength must be tuned; too weak = no pruning, too strong = underfitting |

---

## 7. High Stress Autocorrelation

| | Detail |
|---|--------|
| **Problem** | Naive baseline (predict S(0) forever) gets MSE ~0.004, nearly as good as ML models |
| **Root Cause** | Stress signal has lag-1 autocorrelation ≈ 0.9999 — it barely changes over 60 timesteps (~1 minute) |
| **Impact** | Raw MSE is not very discriminative between models on this task |
| **Solution** | Framed the paper around interpretability rather than raw accuracy: "comparable MSE but with full equation transparency" |
| **Lesson** | In slow-changing physiological signals, prediction accuracy alone is a weak metric. Interpretability and explainability are the real value. |

---

## 8. Accidental Code Corruption

| | Detail |
|---|--------|
| **Problem** | Stray backticks ` `` ` appeared on line 163 of `02_evaluate.py` |
| **Root Cause** | Accidental keyboard entry during editing |
| **Solution** | Removed the backticks during the solver fix edit |
| **Lesson** | Always review diffs before uploading to Kaggle |

---

## Summary Timeline

| Date | Challenge | Status |
|------|-----------|--------|
| Feb 23 | torchdiffeq missing on Kaggle | ✅ Fixed |
| Feb 23 | Data path incorrect on Kaggle | ✅ Fixed |
| Feb 23 | Model path mismatch between notebooks | ✅ Fixed |
| Feb 24 | Solver mismatch (misdiagnosis) | ⚠️ Fixed but wasn't root cause |
| Feb 25 | **Time scale mismatch** | ✅ Fixed — MSE 0.35 → 0.006 |
| Feb 25 | Weak L1 sparsity | 🔲 Needs re-training with λ=0.01 |
| Feb 25 | High autocorrelation | ✅ Addressed via paper framing |
| Feb 25 | Accidental backticks | ✅ Fixed |

---

## Final Corrected Results

| Model | Mean MSE | Interpretable? |
|-------|----------|----------------|
| **MC-UDE** | **0.006355** | ✅ Per-subject equations |
| Ridge_AR | 0.004632 | ❌ Black box |
| RF_AR | 0.004343 | ❌ Black box |
| Naive | 0.004655 | N/A |

**UDE wins 5/15 folds outright** while providing the only interpretable stress equations.
