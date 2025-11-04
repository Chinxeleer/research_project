# Weights & Biases (WandB) Integration Setup

**Date:** 2025-11-04
**Status:** ✅ Fully Integrated

---

## Overview

Comprehensive Weights & Biases integration has been added to track all metrics, visualizations, and model performance for your financial forecasting research. This will help you analyze errors, compare models, and understand prediction quality.

---

## What's Been Set Up

### 1. Enhanced WandB Logger (`utils/wandb_logger.py`)

Added powerful visualization and metrics tracking:

#### **New Features:**
- ✅ **Residual Analysis** - 4-panel analysis including:
  - Residual vs Predicted scatter plot
  - Residual distribution histogram
  - Q-Q plot for normality checking
  - Residuals over time plot

- ✅ **Detailed Error Metrics** - Comprehensive metrics table with:
  - MAE, MSE, RMSE, R², MAPE
  - **Directional Accuracy** - % of correct direction predictions (very important for trading!)

- ✅ **Horizon Analysis** - Per-timestep error breakdown:
  - MSE at each forecast horizon (H=1, 2, 3, ... pred_len)
  - MAE at each forecast horizon
  - Helps identify if errors increase with longer forecasts

- ✅ **Comprehensive Test Results** - Single function call logs:
  - Prediction vs ground truth samples (5 examples)
  - Scatter plot (predicted vs true)
  - Distribution comparison (histogram + boxplot)
  - Residual analysis (4 plots)
  - Horizon analysis (error progression)

### 2. Integrated Logging in Experiment (`exp/exp_long_term_forecasting.py`)

#### **Training Phase:**
- Logs train/val/test loss every epoch
- Tracks model architecture and parameters
- Monitors gradients and weights
- Saves learning curves at end of training

#### **Testing Phase:**
- Comprehensive visualization suite
- All metrics automatically logged
- Predictions vs ground truth comparison
- Error distribution analysis

### 3. Updated Run Scripts

**Modified:** `run_mamba.sh` (other model scripts need similar updates)

**Added flags:**
```bash
--use_wandb \
--wandb_project "financial-forecasting-${dataset}" \
--inverse 1
```

### 4. Command-Line Arguments (`run.py`)

**New arguments added:**
- `--use_wandb` - Enable/disable WandB logging (default: False)
- `--wandb_project` - Project name (default: "financial-forecasting")
- `--wandb_entity` - Team/username (default: None for personal account)

---

## How to Use

### First Time Setup

1. **Install WandB:**
```bash
pip install wandb
```

2. **Login to WandB:**
```bash
wandb login
```
(You'll be prompted for your API key from https://wandb.ai/authorize)

3. **Verify Installation:**
```bash
python -c "import wandb; print('WandB version:', wandb.__version__)"
```

### Running with WandB

#### **Training with WandB Enabled:**

```bash
cd forecast-research
bash run_mamba.sh
```

The script now automatically includes `--use_wandb` flag.

#### **Training Without WandB:**

Remove the `--use_wandb` flag from the script, or set it to offline mode:
```bash
export WANDB_MODE=offline
bash run_mamba.sh
```

#### **Custom Project Name:**

Modify the script:
```bash
--wandb_project "my-custom-project-name" \
```

---

## What You'll See on WandB Dashboard

### 📊 **Metrics Tab**

**Training Metrics (logged every epoch):**
- `train/loss` - Training loss
- `val/loss` - Validation loss
- `test/loss` - Test loss (if computed during training)
- `epoch` - Current epoch number

**Test Metrics (logged once at end):**
- `test/mae` - Mean Absolute Error
- `test/mse` - Mean Squared Error
- `test/rmse` - Root Mean Squared Error
- `test/r2` - R² coefficient
- `test/mape` - Mean Absolute Percentage Error
- `test/directional_accuracy` - **% of correct direction predictions** 🎯

**Validation Metrics:**
- Same as test metrics but for validation set (`val/mae`, `val/mse`, etc.)

### 📈 **Charts Tab**

**1. Learning Curves:**
- Training vs Validation loss over epochs
- Helps identify overfitting/underfitting

**2. Prediction Samples (5 examples):**
- Ground truth (blue) vs Predicted (orange) time series
- Visual inspection of model predictions

**3. Scatter Plot:**
- Predicted values vs True values
- Red diagonal line = perfect predictions
- Points close to line = good predictions

**4. Distribution Comparison:**
- **Histogram**: True vs Predicted value distributions
- **Boxplot**: Statistical summary comparison
- Helps identify if model captures data distribution

**5. Residual Analysis (4 subplots):**
- **Residual Plot**: Residuals vs Predictions (should be random around 0)
- **Residual Histogram**: Distribution of errors (should be normal)
- **Q-Q Plot**: Check if residuals are normally distributed
- **Residuals Over Time**: Check for patterns in errors

**6. Horizon Analysis:**
- **MSE by Forecast Horizon**: Does error increase with longer forecasts?
- **MAE by Forecast Horizon**: Same for MAE
- Critical for understanding model's forecasting capability

### 📋 **Tables Tab**

**1. Metrics Summary Table:**
| Metric | Value |
|--------|-------|
| MAE | 0.004238 |
| MSE | 0.000032 |
| RMSE | 0.005648 |
| R² | -0.018 |
| MAPE (%) | 4.23 |
| Directional Accuracy (%) | 52.4 |

**2. Horizon Metrics Table:**
| Horizon | MSE | MAE |
|---------|-----|-----|
| 1 | 0.000025 | 0.003891 |
| 2 | 0.000028 | 0.004012 |
| 3 | 0.000032 | 0.004238 |
| ... | ... | ... |

### 🏗️ **Model Tab**

- **total_params** - Total model parameters
- **trainable_params** - Parameters being optimized
- **non_trainable_params** - Frozen parameters
- **architecture** - Text representation of model structure

### ⚙️ **Config Tab**

All hyperparameters logged automatically:
- Model architecture (d_model, d_ff, expand, etc.)
- Training config (batch_size, learning_rate, epochs)
- Data config (seq_len, pred_len, features)
- Dataset info (data_path, target column)

---

## Key Metrics Explained

### Directional Accuracy
**Most important for financial forecasting!**

- Measures: % of time the model predicted correct direction of change
- Calculation: `sign(pred[t+1] - pred[t]) == sign(true[t+1] - true[t])`
- Trading relevance: Even if magnitude is wrong, correct direction = profitable trade
- **Goal:** >50% (better than random), ideally 55-60%+ is excellent

### R² (Coefficient of Determination)
- Range: -∞ to 1
- **-0.01 to -0.15**: NORMAL for percentage change forecasting!
- Near 0: Model as good as mean baseline
- Negative: Model worse than mean (but still acceptable for returns prediction)

### MSE vs MAE
- **MSE**: Penalizes large errors more (0.003-0.005 is good for pct_chg)
- **MAE**: Average absolute error (~4-5% for daily returns is acceptable)
- **RMSE**: Square root of MSE, same units as target

### Horizon Analysis
- Shows how error grows with forecast length
- Typically: error increases with longer horizons
- Useful for: Choosing optimal prediction horizon for deployment

---

## Comparing Models Across Runs

### Using WandB Dashboard:

1. **Go to your project**: https://wandb.ai/your-username/financial-forecasting-DATASET

2. **View runs**: Click "Workspace" → Select multiple runs

3. **Compare metrics:**
   - Click "Chart" → Select metric (e.g., `test/mse`)
   - Compares all selected models side-by-side

4. **Create reports:**
   - Click "Reports" → "Create Report"
   - Drag & drop charts and tables
   - Share with collaborators

### Filtering Runs:

**By dataset:**
- Filter by `data_path` config variable
- Example: `data_path: "NVIDIA_normalized.csv"`

**By horizon:**
- Filter by `pred_len` config variable
- Example: `pred_len: 22`

**By model:**
- Filter by `model` config variable
- Example: `model: "Mamba"`

---

## Example Run Outputs

### Console Output (with WandB):
```
wandb: 🚀 View project at https://wandb.ai/username/financial-forecasting-NVIDIA
wandb: 🏃 View run at https://wandb.ai/username/financial-forecasting-NVIDIA/runs/abc123

Epoch: 1, Steps: 45 | Train Loss: 0.0045123 Vali Loss: 0.0043211 Test Loss: 0.0042987
Epoch: 2, Steps: 45 | Train Loss: 0.0041234 Vali Loss: 0.0040123 Test Loss: 0.0039876
...

TEST - MSE: 0.0000, MAE: 0.0042, RMSE: 0.0056, R²: -0.0183, MAPE: 4.23%

📊 Logging comprehensive test results to W&B...
✅ Test results logged to W&B
```

---

## Troubleshooting

### Issue: "wandb: ERROR Error uploading"
**Solution:** Check internet connection or use offline mode:
```bash
export WANDB_MODE=offline
```
Then sync later: `wandb sync ./wandb/offline-run-*`

### Issue: "ModuleNotFoundError: No module named 'wandb'"
**Solution:** Install WandB:
```bash
pip install wandb
```

### Issue: "wandb: ERROR API key not configured"
**Solution:** Login:
```bash
wandb login
```

### Issue: Too many plots slowing dashboard
**Solution:** Reduce number of samples:
```python
# In exp_long_term_forecasting.py line 319
self.wandb_logger.log_comprehensive_test_results(trues, preds, split=split_name, num_samples=3)  # Reduced from 5 to 3
```

### Issue: Want to disable WandB temporarily
**Solution:** Remove `--use_wandb` flag from training script, or:
```bash
export WANDB_MODE=disabled
bash run_mamba.sh
```

---

## Best Practices

### 1. Organized Project Names
Use descriptive names:
- `financial-forecasting-NVIDIA` (per dataset)
- `financial-forecasting-all-models` (compare all models)
- `financial-forecasting-production` (final models)

### 2. Meaningful Run Names
Automatically generated as: `{model}_{dataset}_sl{seq_len}_pl{pred_len}`

Example: `Mamba_NVIDIA_sl60_pl22`

### 3. Tags for Organization
Automatically added tags:
- Model name (Mamba, Autoformer, etc.)
- Dataset name
- Feature type (M, S, MS)
- Sequence length
- Prediction length

Filter by tags in dashboard!

### 4. Regular Syncing (if offline)
```bash
# After training offline
wandb sync ./wandb/offline-run-20251104_*
```

### 5. Comparing Models
- Train all models on same dataset
- Use same project name
- Use parallel coordinates plot to find best hyperparameters
- Create reports for publication

---

## Files Modified

1. ✅ `utils/wandb_logger.py` - Enhanced with new visualization methods
2. ✅ `exp/exp_long_term_forecasting.py` - Integrated logging calls
3. ✅ `run.py` - Added command-line arguments
4. ✅ `run_mamba.sh` - Enabled WandB by default

---

## Next Steps

### To Train with WandB:

```bash
cd forecast-research

# Train Mamba (WandB enabled by default)
bash run_mamba.sh

# Check dashboard
# Go to: https://wandb.ai/your-username/financial-forecasting-NVIDIA
```

### To Update Other Model Scripts:

Add these flags to `run_autoformer.sh`, `run_informer.sh`, etc.:

```bash
--use_wandb \
--wandb_project "financial-forecasting-${dataset}" \
--inverse 1
```

### To Create Comparison Report:

1. Train all 5 models on same dataset
2. Go to WandB project
3. Select all runs
4. Click "Create Report"
5. Add comparison charts
6. Share with team/advisors

---

## Research Benefits

### For Your Paper:

1. **Comprehensive Visualizations**
   - Publication-ready plots
   - Residual analysis for model validation
   - Horizon analysis shows model capability

2. **Reproducibility**
   - All hyperparameters logged
   - Config saved for every run
   - Easy to reproduce best models

3. **Model Comparison**
   - Side-by-side metrics
   - Statistical significance easier to assess
   - Clear winner identification

4. **Error Analysis**
   - Understand WHY models fail
   - Identify patterns in residuals
   - Improve model architecture based on insights

---

## Summary

✅ **What's Tracked:**
- Training/validation/test loss (every epoch)
- All error metrics (MSE, MAE, RMSE, R², MAPE, directional accuracy)
- Prediction vs ground truth comparisons
- Residual analysis (4 plots)
- Horizon analysis (error by timestep)
- Model architecture and hyperparameters

✅ **What You'll See:**
- Real-time training progress
- Comprehensive test visualizations
- Easy model comparisons
- Publication-ready figures

✅ **Benefits:**
- Better understand model behavior
- Identify best models quickly
- Debug errors effectively
- Create research reports easily

🎯 **Result:** Complete visibility into your forecasting models' performance and errors!

---

**Generated:** 2025-11-04
**Status:** Ready for Training with WandB
