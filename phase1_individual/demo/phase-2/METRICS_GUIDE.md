# Understanding Equation Quality Metrics

## Quick Answer
**"How do I know the discovered equation is realistic?"**

Look at these 4 metrics in the app:

### 1. **Average Coefficient Error** ⭐ MOST IMPORTANT
- **< 0.1** = ✅ Excellent (almost perfect!)
- **0.1 - 0.3** = ✓ Good (reliable)
- **0.3 - 0.5** = ⚠ Moderate (approximate)
- **> 0.5** = ❌ Poor (not reliable)

### 2. **Coefficient Correlation**
- **> 0.9** = ✅ Excellent match
- **0.7 - 0.9** = ✓ Good match
- **< 0.7** = ⚠ Poor match

### 3. **Prediction R²**
- **> 0.95** = ✅ Excellent predictions
- **0.85 - 0.95** = ✓ Good predictions
- **< 0.85** = ⚠ Needs improvement

### 4. **Max Coefficient Error**
- Shows if one specific coefficient is problematic
- Should be close to average error

## Your Equation is Realistic if:
1. ✅ Avg Error < 0.2
2. ✅ Correlation > 0.8
3. ✅ R² > 0.90

## Example:
```
True equation:     dy/dt = 2.0·x - 3.0·y + 1.0·z
Learned equation:  dy/dt = 1.98·x - 2.85·y + 0.95·z

Avg Error: 0.073 ✅ Excellent!
Correlation: 0.98 ✅ Excellent!
R²: 0.99 ✅ Excellent!

Result: Equation is realistic and reliable!
```

## If Metrics Are Poor:
- Reduce noise level (try 0.01 instead of 0.05)
- Increase epochs (try 1000 instead of 500)
- Use more samples (try 24h instead of 6h)
- Simplify equation (use fewer variables)

**The app now shows all this automatically with detailed explanations!**
