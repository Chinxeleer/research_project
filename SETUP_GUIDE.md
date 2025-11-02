# Financial Forecasting Research - Setup Guide
## Complete Instructions for University Cluster Execution

---

## 🎯 Overview

This research project compares 5 state-of-the-art deep learning models for financial time series forecasting:
- **Mamba** (State Space Model)
- **Informer** (Efficient Transformer)
- **Autoformer** (Auto-correlation mechanism)
- **FEDformer** (Frequency Enhanced Decomposition)
- **iTransformer** (Inverted Transformer)

**Methodology**: Following Eden Modise's paper exactly
- **Metrics**: MSE, MAE, RMSE, R², MAPE
- **Horizons**: H=3, 5, 10, 22, 50, 100 days
- **Target**: Percentage change (pct_chg) - indirect modeling approach
- **Data Split**: 2215/200/100 days (train/val/test)

---

## ⚠️ CRITICAL FIXES APPLIED

### **1. Mamba Model Fix**
**Problem**: `RuntimeError: selective_scan only supports state dimension <= 256`

**Solution**: All scripts now use:
- `d_model=64` (reduced from 128)
- `expand=2`
- `d_ff=16`
- **Result**: `d_inner = 64 × 2 = 128` ✅ (within limit)

### **2. DOS Line Endings Fixed**
All shell scripts have been converted to UNIX format.

### **3. Metrics Updated**
R² metric added to all experiment outputs.

---

## 📋 Step-by-Step Execution Guide

### **STEP 1: Prepare Your Data (CRITICAL!)**

First, you need to preprocess all your CSV files using Eden's methodology.

```bash
# Navigate to dataset folder
cd dataset/

# Run the preprocessing script
python prepare_data_eden_method.py
```

**What this does:**
- Loads all CSV files from `dataset/` folder
- Cleans data (removes #N/A, invalid values)
- Creates percentage change target variable (pct_chg)
- Splits data: 2215 days train / 200 days val / 100 days test
- Normalizes using ONLY training statistics (prevents data leakage)
- Saves processed files to `dataset/processed_data/`

**Expected output:**
```
processed_data/
├── NVIDIA_normalized.csv
├── APPLE_normalized.csv
├── SP500_normalized.csv
├── NASDAQ_normalized.csv
├── ABSA_normalized.csv
├── SASOL_normalized.csv
├── DRD_GOLD_normalized.csv
├── ANGLO_AMERICAN_normalized.csv
└── *_norm_params.csv (normalization parameters for each)
```

**⚠️ IMPORTANT**: Do NOT proceed to Step 2 until this completes successfully!

---

### **STEP 2: Navigate to Forecast Research Folder**

```bash
cd ../forecast-research/
```

---

### **STEP 3: Choose Your Execution Strategy**

You have 3 options:

#### **Option A: Run ALL Experiments (Recommended for Complete Results)**

This runs all 5 models × 6-8 datasets × 6 horizons = ~180-240 experiments

```bash
./run_experiments.sh
```

**Estimated time**: 24-48 hours on GPU cluster (depends on hardware)

---

#### **Option B: Run Individual Models (Recommended for Testing/Debugging)**

Test one model first to ensure everything works:

```bash
# Test Mamba first (your primary focus)
./run_mamba.sh

# If successful, run others:
./run_informer.sh
./run_autoformer.sh
./run_fedformer.sh
./run_itransformer.sh
```

---

#### **Option C: Run Single Experiment (Quick Test)**

Test with ONE stock and ONE horizon to verify setup:

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../dataset/processed_data/ \
    --data_path NVIDIA_normalized.csv \
    --model_id Test_Mamba_NVIDIA_H10 \
    --model Mamba \
    --data custom \
    --features M \
    --target pct_chg \
    --seq_len 60 \
    --label_len 30 \
    --pred_len 10 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 6 \
    --d_model 64 \
    --d_ff 16 \
    --d_conv 4 \
    --expand 2 \
    --e_layers 2 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --patience 3 \
    --use_gpu 1 \
    --gpu 0 \
    --des 'Test' \
    --itr 1
```

This should complete in 10-20 minutes. Check the output for:
- ✅ Training loss decreasing
- ✅ Test metrics: MSE, MAE, RMSE, R²
- ✅ No errors

---

### **STEP 4: Monitor Progress**

During training, you'll see output like:

```
Epoch: 1 cost time: 23.4s
    train_loss: 0.0234
    val_loss: 0.0198

Epoch: 2 cost time: 22.1s
    train_loss: 0.0189
    val_loss: 0.0176
...

TEST - MSE: 0.0012, MAE: 0.0234, RMSE: 0.0346, R²: 0.7834, MAPE: 2.34%
```

**Good signs:**
- Train loss and val loss both decreasing
- R² > 0.5 (closer to 1.0 is better)
- MSE and MAE values decreasing over epochs

**Bad signs:**
- Loss = NaN or exploding (>1000)
- R² negative (worse than baseline)
- Val loss increasing while train loss decreases (overfitting)

---

### **STEP 5: Collect Results**

After experiments complete, results are saved in multiple locations:

```
forecast-research/
├── checkpoints/          # Model weights
│   └── {model_id}/
│       └── checkpoint.pth
│
├── results/              # Predictions and metrics
│   └── {model_id}/
│       ├── pred.npy      # Predictions
│       ├── true.npy      # Ground truth
│       └── metrics.npy   # [MAE, MSE, RMSE, MAPE, MSPE, R²]
│
├── test_results/         # Visualizations
│   └── {model_id}/
│       └── *.pdf
│
└── result_long_term_forecast.txt  # Summary of all experiments
```

---

## 📊 Analyzing Results

### **Extract Results for Your Presentation**

```bash
# View summary of all experiments
cat result_long_term_forecast.txt

# Example output:
# Mamba_NVIDIA_H10_Exp
# mse:0.0012, mae:0.0234, rmse:0.0346, r2:0.7834, mape:2.34%, mspe:0.0056
```

### **Load Predictions for Analysis**

```python
import numpy as np

# Load results for a specific experiment
model_id = "Mamba_NVIDIA_H10"
results_dir = f"results/{model_id}_Exp/"

predictions = np.load(f"{results_dir}/pred.npy")
ground_truth = np.load(f"{results_dir}/true.npy")
metrics = np.load(f"{results_dir}/metrics.npy")

mae, mse, rmse, mape, mspe, r2 = metrics

print(f"Model: {model_id}")
print(f"MAE: {mae:.6f}")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²: {r2:.6f}")
print(f"MAPE: {mape:.2f}%")
```

---

## 🔧 Common Issues and Solutions

### **Issue 1: "ModuleNotFoundError"**

**Solution**: Install requirements on cluster

```bash
pip install -r requirements.txt
```

---

### **Issue 2: High MSE/MAE Scores**

**Possible causes:**
1. ❌ Data not normalized → Run Step 1 again
2. ❌ Wrong target variable → Should be `pct_chg`, not `Close`
3. ❌ Learning rate too high → Try 0.00001 instead of 0.0001
4. ❌ Insufficient training → Increase epochs to 100

**Quick fix:**
```bash
# In the training script, modify:
LEARNING_RATE=0.00001
TRAIN_EPOCHS=100
```

---

### **Issue 3: GPU Out of Memory**

**Solution**: Reduce batch size

```bash
# Modify in script:
BATCH_SIZE=16  # or even 8
```

---

### **Issue 4: Data File Not Found**

**Error**: `FileNotFoundError: NVIDIA_normalized.csv`

**Solution**: Ensure Step 1 completed successfully
```bash
ls dataset/processed_data/
# Should show all *_normalized.csv files
```

---

## 🎓 For Your Supervisor Presentation

### **Key Results to Present**

Create a comparison table like Eden's paper:

| Model | Stock | H=3 MSE | H=10 MSE | H=22 MSE | H=50 MSE | H=100 MSE |
|-------|-------|---------|----------|----------|----------|-----------|
| Mamba | NVIDIA| 0.00044 | 0.00052  | 0.00008  | 0.00040  | 0.00116   |
| Informer | NVIDIA | 0.00007 | 0.00009  | 0.00015  | 0.00040  | 0.00131   |
| ... | ... | ... | ... | ... | ... | ... |

### **Key Points to Highlight**

1. ✅ **Methodology**: Replicated Eden's exact approach
   - Percentage change target (indirect modeling)
   - Proper data normalization (no leakage)
   - Same train/val/test split (2215/200/100)

2. ✅ **Metrics**: Comprehensive evaluation
   - MSE, MAE (standard forecasting metrics)
   - RMSE (interpretable in same units as data)
   - R² (explains variance, 0-1 scale)

3. ✅ **Models**: State-of-the-art comparison
   - Mamba (latest SSM architecture)
   - Transformer variants (Informer, Autoformer, FEDformer)
   - iTransformer (specialized for multivariate)

4. ✅ **Robustness**: Multiple horizons tested
   - Short-term: H=3, 5 (weekly forecasts)
   - Medium-term: H=10, 22 (monthly forecasts)
   - Long-term: H=50, 100 (quarterly forecasts)

---

## ⚡ Quick Reference Commands

```bash
# 1. Preprocess data
cd dataset && python prepare_data_eden_method.py

# 2. Run quick test
cd ../forecast-research
./run_mamba.sh  # or run single experiment

# 3. Check results
cat result_long_term_forecast.txt

# 4. View specific experiment
ls results/Mamba_NVIDIA_H10_Exp/
```

---

## 📞 Troubleshooting Checklist

Before running experiments, verify:

- [ ] All CSV files are in `dataset/` folder
- [ ] Preprocessing script ran successfully
- [ ] `dataset/processed_data/` contains *_normalized.csv files
- [ ] You're in `forecast-research/` directory
- [ ] Training scripts are executable (`chmod +x *.sh`)
- [ ] GPU is available (check with `nvidia-smi` if using CUDA)

---

## 🚀 Expected Timeline

| Task | Time Estimate |
|------|---------------|
| Data preprocessing | 5-10 minutes |
| Single model test | 10-20 minutes |
| Full model (one stock, all horizons) | 2-4 hours |
| All experiments | 24-48 hours |

**Recommendation for tomorrow's presentation:**
1. Run data preprocessing NOW (10 min)
2. Run Mamba on 2-3 stocks with H=10,22,50 (4-6 hours overnight)
3. Present those results + explain methodology

---

## 📚 References

This setup follows:
- **Eden Modise's Paper**: "Comparative Study of Deep Learning Methods in Stock Forecasting and Prediction"
- **Time-Series-Library**: Modified fork with 25+ models
- **Data Normalization**: StandardScaler with training statistics only

---

## ✅ Success Criteria

Your experiments are successful if:
1. MSE and MAE values are similar to Eden's paper (0.0001 - 0.01 range)
2. R² values are positive (>0.0) and ideally >0.5
3. Models show reasonable rankings across horizons
4. No NaN values or crashes during training

Good luck with your presentation! 🎓
