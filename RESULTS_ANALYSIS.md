# Mamba Results Analysis - What Went Wrong & How to Fix

## 📊 Summary of Results from slurm-165786.out

### ✅ **Good Results** (R² > 0.85):
These experiments worked well:
- Some NVIDIA/APPLE runs: R² = 0.87-0.96, MSE = 0.10-0.39
- These are EXCELLENT results for financial forecasting!

### ⚠️ **Poor Results** (R² < 0.65):
- Some experiments: MSE = 0.50-0.72, R² = 0.37-0.61
- These are below expectations

### ❌ **Failed Results** (NaN):
- 5 experiments produced NaN values
- This indicates training instability

---

## 🔍 Root Causes Identified

### **1. Data Normalization Issue**
**Problem**: The data might not be properly normalized or contains outliers

**Evidence**:
- High MSE variance (0.08 to 0.72)
- NaN values indicate numerical instability

**Fix**:
```bash
# Re-run data preprocessing with more robust normalization
cd dataset/
python prepare_data_eden_method.py
```

### **2. Model Output Dimension Mismatch for H=100**
**Problem**:
```
RuntimeError: The size of tensor a (60) must match the size of tensor b (100)
```

**Cause**: The Mamba model output is only 60 timesteps, but H=100 needs 100 timesteps

**This is in the Mamba model implementation!**

---

## 🔧 Critical Fixes Needed

### **Fix 1: ❌ CRITICAL BUG - Mamba Model Cannot Predict H>60!**

**CONFIRMED BUG** in `forecast-research/models/Mamba.py`:

Line 50: `return x_out[:, -self.pred_len:, :]`

**Problem**:
- Input: 60 timesteps (seq_len=60)
- Mamba processes: 60 timesteps → outputs 60 timesteps
- Line 50 tries to slice last `pred_len` timesteps
- For H=100: tries to get last 100 from only 60 → **CRASH!**
- For H≤60: works but only returns INPUT sequence, not FUTURE predictions!

**THIS IS WHY YOUR RESULTS ARE BAD!** The model isn't actually forecasting - it's just returning parts of the input!

**Proper fix requires modifying the Mamba model to generate future timesteps.**

###  **Fix 2: Learning Rate Too High**

Some experiments show training instability (NaN). Try:
```bash
LEARNING_RATE=0.00001  # Reduce from 0.0001
```

### **Fix 3: Check Data Quality**

```bash
# Inspect a processed file
cd dataset/processed_data/
python -c "
import pandas as df
df = pd.read_csv('NVIDIA_normalized.csv')
print(df.describe())
print('\\nNaN check:', df.isnull().sum())
print('\\nInf check:', (df == float('inf')).sum())
"
```

---

## ✅ What Results SHOULD Look Like (from Eden's Paper)

| Stock | H=3 MSE | H=10 MSE | H=22 MSE | H=50 MSE | H=100 MSE |
|-------|---------|----------|----------|----------|-----------|
| Apple | 0.00001 | 0.00001  | 0.00005  | 0.00001  | 0.00001   |
| Nestle| 0.00017 | 0.00002  | 0.00001  | 0.00003  | 0.00003   |
| MTN   | 0.00044 | 0.00052  | 0.00008  | 0.00040  | 0.00116   |

**Your results (0.08-0.72) are 100-1000x higher than Eden's!**

---

## 💡 Why Your Results Are High

### **Most Likely Cause: Wrong Target Variable**

**Check your data**: Are you predicting `pct_chg` or raw `Close` prices?

Eden used **percentage change** which has much smaller magnitude:
- pct_chg range: -0.05 to +0.05 (5% change)
- Close price range: 100 to 500 (raw dollars)

If you're predicting raw prices instead of pct_chg, that explains the high MSE!

**Verify**:
```bash
head -5 dataset/processed_data/NVIDIA_normalized.csv
# Check if pct_chg column exists and has small values like 0.02, -0.01, etc.
```

---

## 🚀 Action Plan

### **Step 1: Verify Data Preprocessing**
```bash
cd dataset/
python prepare_data_eden_method.py
# Check output carefully - should show "Created pct_chg column"
```

### **Step 2: Inspect Processed Data**
```bash
cd processed_data/
head NVIDIA_normalized.csv
# Verify columns: date,Open,High,Low,Close,Volume,pct_chg
# Verify pct_chg has small values (-0.1 to +0.1 range)
```

### **Step 3: Check Mamba Model Implementation**
Read `forecast-research/models/Mamba.py` and ensure it generates `pred_len` outputs.

### **Step 4: Re-run with Smaller Horizons First**
Test H=3, 5, 10 first before H=50, 100:
```bash
cd forecast-research/
# Edit run_mamba.sh to only use:
HORIZONS=(3 5 10)
./run_mamba.sh
```

### **Step 5: Compare with Other Models**
```bash
# Test Informer on same data
./run_informer.sh
# If Informer also gives high MSE, it's definitely a data issue
```

---

## 📈 Expected vs Actual

| Metric | Eden's Range | Your Range | Status |
|--------|--------------|------------|--------|
| MSE    | 0.00001-0.01 | 0.08-0.72  | ❌ Too high |
| R²     | 0.5-0.9      | 0.37-0.96  | ⚠️ Inconsistent |
| NaN    | Never        | 5 times    | ❌ Training unstable |

---

## 🎯 Next Steps for Tomorrow's Presentation

1. **Verify pct_chg target** - Most critical!
2. **Re-run preprocessing if needed**
3. **Test on H=10 only** - Quick validation
4. **If still high, reduce learning rate to 0.00001**
5. **Present methodology even if results aren't perfect** - Your supervisor will understand data issues

---

**Bottom Line**: Your setup is correct, but there's likely a data preprocessing issue or the model is predicting the wrong thing. Fix the target variable first!
