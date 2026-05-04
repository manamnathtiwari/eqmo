"""
✅ TESTED & WORKING - Universal Demo

## Status: READY TO RUN

The models you trained use a SINGLE alpha coefficient (not 18 individual ones).

This is the SIMPLE version:
- dS/dt = -β·S + α·(sum of features) + NN(S, features)

NOT the multi-coefficient version.

## To Run:

```bash
streamlit run demo/phase-2/universal_demo.py
```

## What Works:
- ✅ Loads all 15 trained models
- ✅ Generates synthetic data
- ✅ Gets predictions from ALL models
- ✅ Shows ensemble with uncertainty
- ✅ Compares all models

## Model Architecture (from your trained files):
- Single β (recovery rate)
- Single α (overall feature sensitivity)  
- Neural network (19→64→64→1)

**This matches your actual trained models!**
