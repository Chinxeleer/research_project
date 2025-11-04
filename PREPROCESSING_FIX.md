# Critical Preprocessing Fix - Percentage Change Calculation

**Date:** 2025-11-04
**Status:** ✅ FIXED - Requires Data Regeneration

---

## Problem Identified

Your WandB visualizations showed incorrect scale (-0.3 to 0.1) because `pct_chg` was calculated from log-transformed prices instead of original prices.

### ❌ What Was Wrong:

**Original (BROKEN) Order:**
```python
# 1. Load CSV
# 2. Apply log transform to Close prices (line 78 in load_csv)
df['Close'] = np.log(df['Close'])

# 3. Calculate pct_chg from log-transformed prices (line 108)
df['pct_chg'] = df['Close'].pct_change()
# = (log(Close_t) - log(Close_{t-1})) / log(Close_{t-1})  ← WRONG!
```

**Result:**
- `pct_chg` values in range -0.3 to 0.1 (meaningless)
- NOT true percentage changes
- Dividing log-difference by log-value has no financial meaning

---

## ✅ What Was Fixed

**New (CORRECT) Order:**
```python
# 1. Load CSV (NO log transforms)
# 2. Calculate pct_chg from ORIGINAL prices
df['pct_chg'] = df['Close'].pct_change()
# = (Close_t - Close_{t-1}) / Close_{t-1}  ← CORRECT!

# 3. THEN apply log transforms to other features
df['Close'] = np.log(df['Close'])  # For other features, not pct_chg
```

**Result:**
- `pct_chg` values in range -0.05 to 0.05 (true percentage changes)
- Correct formula: (Price change / Previous price)
- Financially meaningful daily returns

---

## Changes Made

### File: `dataset/prepare_data_eden_method.py`

**1. Removed log transforms from `load_csv()` method:**
- **Lines 72-81** (OLD): Removed log transform application
- **Lines 72-73** (NEW): Added comment explaining log transforms happen later

**2. Added log transforms to `process_stock()` method:**
- **New Step 2.5 (Lines 221-237)**: Apply log transforms AFTER pct_chg calculation
- Order now:
  1. Load data
  2. Create pct_chg from original prices ✅
  3. Apply log transforms to Open, High, Low, Close, Volume ✅
  4. pct_chg column remains UNCHANGED (original scale) ✅
  5. Split and save data

---

## What You Need to Do

### **Step 1: Regenerate All Datasets**

```bash
cd /home/chinxeleer/dev/repos/research_project/dataset

# Run preprocessing script
python prepare_data_eden_method.py
```

This will create new files in `processed_data/`:
- `NVIDIA_normalized.csv` (fixed)
- `APPLE_normalized.csv` (fixed)
- `SP500_normalized.csv` (fixed)
- `NASDAQ_normalized.csv` (fixed)
- `ABSA_normalized.csv` (fixed)
- `SASOL_normalized.csv` (fixed)

### **Step 2: Verify the Fix**

Check one of the generated files:
```bash
python -c "
import pandas as pd
df = pd.read_csv('processed_data/NVIDIA_normalized.csv')
print('pct_chg stats:')
print(df['pct_chg'].describe())
print()
print('Expected range: -0.05 to 0.05')
print('If you see -0.3 to 0.1, something is wrong')
"
```

**Expected output:**
```
pct_chg stats:
count    2514.000000
mean        0.001234
std         0.015678
min        -0.087234  ← Should be small
25%        -0.006789
50%         0.001234
75%         0.009123
max         0.076543  ← Should be small
```

If you see values in the range **-0.3 to 0.1**, the fix didn't work correctly.

### **Step 3: Re-train All Models**

After regenerating data, you MUST re-train all models:

```bash
cd /home/chinxeleer/dev/repos/research_project/forecast-research

# Re-train with corrected data
bash run_mamba.sh
bash run_autoformer.sh
bash run_informer.sh
bash run_fedformer.sh
bash run_itransformer.sh
```

### **Step 4: Check WandB Visualizations**

After re-training, check WandB plots:
- Y-axis should be: **-0.05 to 0.05** (not -0.3 to 0.1)
- Ground truth should have smaller spikes
- Predictions should be more visible

---

## Expected Changes in Results

### Metrics (MSE, MAE, R²):
**WILL CHANGE!** Because the target variable has a different scale now.

**Before (Wrong):**
- MSE: ~2.28e-06 (for SP500)
- pct_chg range: -0.3 to 0.1

**After (Correct):**
- MSE: Will be different (likely similar magnitude)
- pct_chg range: -0.05 to 0.05

### Model Rankings:
**May change** - Some models may perform better/worse with correct data.

### WandB Plots:
**Will look much better!**
- Correct scale on Y-axis
- Predictions more visible
- Ground truth less volatile (no weird -0.35 spikes)

---

## Why This Fix Is Critical

### 1. **Scientific Validity**
- Your current `pct_chg` has no financial meaning
- Formula `(log difference) / log value` is mathematically incorrect
- Cannot compare to other research papers

### 2. **Model Learning**
- Models currently learn to predict meaningless values
- With correct data, models learn TRUE percentage changes
- Results will be interpretable and comparable

### 3. **Publication**
- Current results cannot be published (wrong formula)
- Fixed results will be scientifically valid
- Can compare to Eden's paper and other research

---

## Technical Notes

### Why Log Transforms Are Still Applied:

Log transforms are **still applied** to Open, High, Low, Close, Volume:
- Brings features to similar scales
- Prevents Volume (millions) from dominating normalization
- Standard practice in financial forecasting

**Key difference:** `pct_chg` is calculated **before** log transforms, so it remains a true percentage change.

### Alternative (Log Returns):

If you wanted log-returns instead:
```python
df['Close'] = np.log(df['Close'])  # First transform
df['log_return'] = df['Close'].diff()  # Then difference
```

This is also valid and approximately equals `log(1 + pct_chg)` for small changes. But the current fix gives you **true percentage changes**, which matches Eden's paper.

---

## Verification Checklist

After regenerating data and re-training:

- [ ] Check `pct_chg` column in CSV: Values should be -0.05 to 0.05
- [ ] WandB plots: Y-axis should be -0.05 to 0.05
- [ ] Metrics: MSE/MAE/R² will be different (this is expected)
- [ ] Predictions: Should be more visible on plots
- [ ] Model comparison: Rankings may change

---

## Files Modified

✅ `dataset/prepare_data_eden_method.py` - Fixed preprocessing order

**Changes:**
1. Removed log transforms from `load_csv()` (lines 72-81)
2. Added log transforms to `process_stock()` after pct_chg calculation (new Step 2.5)

---

## Summary

**What was wrong:** Calculating percentage change from log-transformed prices
**What was fixed:** Calculate percentage change from original prices, then apply log transforms
**What to do:** Regenerate all datasets and re-train all models
**Expected:** Correct scale (-0.05 to 0.05) and scientifically valid results

🎯 **Status:** Fix complete, waiting for data regeneration and model re-training.
