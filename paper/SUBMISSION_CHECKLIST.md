# Research Paper Submission Checklist

## Paper Status: DRAFT

**Target Venues:**
- [ ] IEEE EMBC (Engineering in Medicine and Biology Conference)
- [ ] ACM ICMI (International Conference on Multimodal Interaction)
- [ ] IEEE JBHI (Journal of Biomedical and Health Informatics)
- [ ] Nature Scientific Reports

---

## Pre-Submission Checklist

### **Content Complete**
- [x] Abstract written
- [x] Introduction complete
- [x] Related work comprehensive
- [x] Methodology detailed
- [ ] Results section (waiting for Kaggle training)
- [x] Discussion written
- [x] Conclusion written
- [x] References cited (10 key papers)

### **Figures & Tables**
- [ ] Figure 1: Model architecture diagram
- [ ] Figure 2: Coefficient heatmap
- [ ] Figure 3: Performance comparison
- [ ] Figure 4: Subject-specific equations
- [ ] Figure 5: Prediction time series
- [ ] Table I: Performance metrics
- [ ] Table II: Learned coefficients
- [ ] Table III: Subject clustering

### **Experimental Results**
- [ ] Kaggle training complete (in progress)
- [ ] Results downloaded
- [ ] Metrics calculated
- [ ] Statistical tests performed
- [ ] Figures generated from results

### **Code & Reproducibility**
- [x] Code organized in repository
- [ ] README with instructions
- [ ] Requirements.txt
- [ ] Trained models saved
- [ ] Demo notebook
- [ ] GitHub repository public

---

## Writing Tasks (After Results)

### **1. Update Results Section**
- [ ] Fill in actual MSE values from Kaggle
- [ ] Add statistical significance tests
- [ ] Update Table I with real numbers
- [ ] Update Table II with learned coefficients
- [ ] Generate all figures with real data

### **2. Refine Discussion**
- [ ] Interpret learned coefficients
- [ ] Discuss subject clustering
- [ ] Compare with baselines
- [ ] Explain performance gains

### **3. Polish Writing**
- [ ] Proofread entire paper
- [ ] Check grammar and style
- [ ] Verify citations
- [ ] Ensure consistent terminology
- [ ] Check figure/table references

---

## Technical Validation

### **Experiments to Run**
- [ ] LOSO cross-validation (15 folds) - IN PROGRESS
- [ ] Baseline comparisons (Linear, RF, LSTM)
- [ ] Ablation study (remove NN, remove physics)
- [ ] Hyperparameter sensitivity
- [ ] Statistical significance tests

### **Metrics to Report**
- [ ] Mean ± Std MSE across folds
- [ ] MAE, R² scores
- [ ] Per-subject results
- [ ] Training time
- [ ] Inference time

### **Reproducibility**
- [ ] Random seeds fixed
- [ ] Hyperparameters documented
- [ ] Data splits saved
- [ ] Environment specified (Python 3.11, PyTorch 2.0)

---

## Submission Requirements

### **IEEE EMBC**
- **Deadline:** Check conference website
- **Format:** 4-6 pages, IEEE two-column
- **Requirements:**
  - [ ] PDF in IEEE format
  - [ ] Copyright form
  - [ ] Supplementary materials (optional)
- **Review:** Double-blind

### **ACM ICMI**
- **Deadline:** Check conference website
- **Format:** 8 pages, ACM format
- **Requirements:**
  - [ ] PDF in ACM format
  - [ ] Video demo (optional but recommended)
  - [ ] Code repository link
- **Review:** Single-blind

### **IEEE JBHI (Journal)**
- **Deadline:** Rolling submissions
- **Format:** No page limit, IEEE journal format
- **Requirements:**
  - [ ] Extended version with more experiments
  - [ ] Detailed methodology
  - [ ] Comprehensive related work
  - [ ] Clinical validation (if possible)
- **Review:** Double-blind

---

## Post-Submission Tasks

### **If Accepted**
- [ ] Prepare camera-ready version
- [ ] Address reviewer comments
- [ ] Update figures to publication quality
- [ ] Prepare presentation/poster
- [ ] Upload to arXiv
- [ ] Share on social media

### **If Rejected**
- [ ] Read reviewer feedback carefully
- [ ] Identify weaknesses
- [ ] Strengthen experiments
- [ ] Revise and resubmit to another venue

---

## Current Status Summary

**What's Done:**
✅ Complete paper draft (3,500 words)
✅ Methodology fully described
✅ Model architecture defined
✅ Code tested and working
✅ Kaggle training started

**What's Pending:**
⏳ Kaggle training completion (7-10 hours remaining)
⏳ Results analysis
⏳ Figure generation
⏳ Baseline comparisons
⏳ Final proofreading

**Estimated Time to Submission:**
- Results ready: +10 hours (Kaggle)
- Analysis & figures: +4 hours
- Baselines: +6 hours
- Writing polish: +3 hours
- **Total: ~1-2 days after Kaggle completes**

---

## Next Immediate Steps

1. **Wait for Kaggle** (currently running)
2. **Download results** when complete
3. **Generate figures** using scripts in FIGURES_GUIDE.md
4. **Update Results section** with actual numbers
5. **Run baseline comparisons** (Linear, RF, LSTM)
6. **Proofread** and polish
7. **Choose venue** and format accordingly
8. **Submit!**

---

## Contact for Collaboration

If you want co-authors or feedback:
- [ ] Reach out to advisor/collaborators
- [ ] Share draft for feedback
- [ ] Discuss authorship order
- [ ] Acknowledge funding sources

---

**Paper is 80% complete!**
**Main blocker: Waiting for Kaggle training results**
**ETA to submission: 2-3 days**
