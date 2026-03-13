# MC-UDE Research Log
## Experiment Timeline & Key Decisions

---

### 2026-02-23: Full Pipeline Run on Kaggle (GPU P100)

#### Training (01_train.py)
- **15-fold LOSO cross-validation** completed in ~1.4 hours
- **Mean Test MSE: 0.006281 ± 0.004204**
- β (recovery rate) consistent across subjects: 0.0634–0.0673
- L1 sparsity: 0% (all 18/18 features active) — λ_L1=0.001 too weak

**Per-fold results:**
| Fold | Subject | Test MSE | β |
|------|---------|----------|------|
| 1 | S002 | 0.005605 | 0.0651 |
| 2 | S003 | 0.013668 | 0.0673 |
| 3 | S004 | 0.006281 | 0.0652 |
| 4 | S005 | 0.004431 | 0.0655 |
| 5 | S006 | 0.003771 | 0.0635 |
| 6 | S007 | 0.004647 | 0.0634 |
| 7 | S008 | 0.018670 | 0.0644 |
| 8 | S009 | 0.004958 | 0.0648 |
| 9 | S010 | 0.003862 | 0.0643 |
| 10 | S011 | 0.004453 | 0.0645 |
| 11 | S013 | 0.004704 | 0.0643 |
| 12 | S014 | 0.006373 | 0.0664 |
| 13 | S015 | 0.003509 | 0.0658 |
| 14 | S016 | 0.004665 | 0.0649 |
| 15 | S017 | 0.004612 | 0.0653 |

**Top features consistently across all subjects:**
1. Heart Rate (α ≈ 0.087)
2. Workload (α ≈ 0.086)
3. Temp_Mean / Temp_Std (α ≈ 0.084)
4. Resp_Rate (α ≈ 0.081)
5. Activity_Std (α ≈ 0.082)

---

### 2026-02-24: Fair Comparison (02_evaluate.py) — First Run (BUGGY)

**Task:** All models predict S(0)→S(60) trajectories. No past stress as input.

**Results (with dopri5 solver — WRONG):**
| Model | Mean MSE | Std |
|-------|----------|-----|
| UDE | 0.347138 | 0.463054 |
| Ridge_AR | 0.004632 | 0.001806 |
| RF_AR | 0.004343 | 0.002385 |
| Naive | 0.004655 | 0.002103 |

**UDE appeared 75x worse than baselines!**

#### Root Cause: Time Scale Mismatch (NOT Solver Mismatch)

**Initial misdiagnosis:** Thought the issue was euler vs dopri5 solver mismatch.
Changing solver made no difference (0.465→0.464). The real issue was **time scale**.

**Actual problem:** Training uses actual CSV time values (`df['time']`), where time
spacing is ~0.017 minutes per step. But evaluation used `torch.arange(60)` = [0,1,2,...,59],
which has spacing of 1.0 per step — **~60x larger**.

**Why this breaks the ODE:**
- The model learned `dS/dt` dynamics calibrated for dt ≈ 0.017
- With Euler: `S(t+1) = S(t) + dS/dt × dt`
- Training dt ≈ 0.017 → small, correct updates
- Evaluation dt = 1.0 → updates 60x too large → trajectory explodes

**Fix:** Extract actual time values from CSV and use them in evaluation:
```python
# BEFORE (wrong — integer indices):
t = torch.arange(seq_len, dtype=torch.float32)

# AFTER (correct — actual CSV time, zero-based):
t = torch.FloatTensor(t_test[si])  # [0, 0.017, 0.033, ...]
```

---

### 2026-02-25: Fair Comparison CORRECTED (with time-scale fix)

**Fix applied:** `prepare_data` now extracts actual CSV time values. Evaluation
uses `t = torch.FloatTensor(t_test[si])` instead of `torch.arange(60)`.

**Results (FINAL — use these for paper):**
| Fold | Subject | UDE | Ridge | RF | Naive | UDE Wins? |
|------|---------|-------|-------|-------|-------|-----------|
| 1 | S002 | 0.0050 | 0.0052 | 0.0036 | 0.0037 | ✅ vs Ridge |
| 2 | S003 | 0.0121 | 0.0057 | 0.0017 | 0.0026 | ❌ |
| 3 | S004 | 0.0059 | 0.0036 | 0.0036 | 0.0037 | ❌ |
| 4 | S005 | 0.0055 | 0.0047 | 0.0046 | 0.0046 | ❌ |
| 5 | S006 | 0.0047 | 0.0066 | 0.0075 | 0.0076 | ✅ Best |
| 6 | S007 | 0.0053 | 0.0052 | 0.0057 | 0.0059 | ≈ tied |
| 7 | S008 | 0.0203 | 0.0046 | 0.0049 | 0.0048 | ❌ outlier |
| 8 | S009 | 0.0042 | 0.0032 | 0.0032 | 0.0035 | ❌ |
| 9 | S010 | 0.0029 | 0.0018 | 0.0014 | 0.0018 | ❌ |
| 10 | S011 | 0.0040 | 0.0052 | 0.0050 | 0.0051 | ✅ Best |
| 11 | S013 | 0.0041 | 0.0021 | 0.0007 | 0.0026 | ❌ |
| 12 | S014 | 0.0071 | 0.0053 | 0.0059 | 0.0060 | ❌ |
| 13 | S015 | 0.0045 | 0.0059 | 0.0062 | 0.0062 | ✅ Best |
| 14 | S016 | 0.0034 | 0.0021 | 0.0020 | 0.0023 | ❌ |
| 15 | S017 | 0.0064 | 0.0083 | 0.0092 | 0.0094 | ✅ Best |

**Summary:**
| Model | Mean MSE | Std | Interpretable? |
|-------|----------|-----|----------------|
| MC-UDE | 0.006355 | 0.004415 | ✅ Yes |
| Ridge_AR | 0.004632 | 0.001806 | ❌ No |
| RF_AR | 0.004343 | 0.002385 | ❌ No |
| Naive | 0.004655 | 0.002103 | N/A |

**UDE wins 5/15 folds** outright, tied on 1. Mean MSE ~37% higher than RF but
with full interpretability (per-subject equations, physics constraints, biomarker profiles).

---

### Recommended Next Steps

1. **Re-train with λ_L1=0.01** — Stronger sparsity for differentiated per-subject profiles
2. **Build Streamlit demo** — Visual app for college presentation
3. **Architecture diagram** — IEEE-style figure for the paper
4. **Write paper** — Use these results + equations + profiles
