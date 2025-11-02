# 🚀 QUICK START - For Tomorrow's Presentation

## ⏰ Timeline: Get Results in 4-6 Hours

This guide helps you get **presentable results by tomorrow** by focusing on the most important experiments.

---

## 📋 Pre-Flight Checklist (5 minutes)

```bash
# Step 1: Validate your setup
./validate_setup.sh
```

**Expected output**: "All checks passed!" or warnings you can ignore

If you see **ERRORS**, follow the actions shown, then re-run validation.

---

## 🔥 Priority Execution Plan

### **Phase 1: Data Preparation (10 minutes) - DO THIS FIRST!**

```bash
cd dataset/
python prepare_data_eden_method.py
```

**Wait for this to complete!** You should see:
```
Processing: NVIDIA
✓ Loaded X rows
✓ Created pct_chg column
✓ Splitting into train/val/test...
✓ Normalizing using training statistics...
✓ Saved normalized data to: processed_data/NVIDIA_normalized.csv

[... repeat for all stocks ...]

Successfully processed 6 stocks:
  ✓ NVIDIA
  ✓ APPLE
  ✓ SP500
  ✓ NASDAQ
  ✓ ABSA
  ✓ SASOL
```

**If this fails**, check your CSV files for errors (like #N/A values).

---

### **Phase 2: Quick Validation Test (15 minutes)**

Test ONE experiment to ensure everything works:

```bash
cd ../forecast-research/

# Quick test with Mamba on NVIDIA, H=10, only 10 epochs
# CRITICAL: d_model=64, expand=2 (d_inner=128) to avoid "state dimension <= 256" error
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../dataset/processed_data/ \
    --data_path NVIDIA_normalized.csv \
    --model_id QuickTest_Mamba_NVIDIA \
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
    --des 'QuickTest' \
    --itr 1
```

**What to look for:**
- Training completes without errors
- You see: `TEST - MSE: X.XXXX, MAE: X.XXXX, RMSE: X.XXXX, R²: X.XXXX`
- Values are reasonable (not NaN or >1000)

**If test passes**, proceed to Phase 3. **If it fails**, debug before continuing.

---

### **Phase 3: Core Experiments for Presentation (4-6 hours)**

Focus on **2 stocks** and **3 horizons** to get comparable results to Eden's paper:

```bash
# Create a focused experiment script
cat > run_presentation_experiments.sh << 'EOF'
#!/bin/bash

# Focused experiments for presentation
# 2 stocks × 5 models × 3 horizons = 30 experiments

STOCKS=("NVIDIA" "APPLE")
MODELS=("Mamba" "Informer" "Autoformer" "FEDformer" "iTransformer")
HORIZONS=(10 22 50)  # Medium and long-term forecasts

ROOT_PATH="../dataset/processed_data/"
SEQ_LEN=60
LABEL_LEN=30
BATCH_SIZE=32
LEARNING_RATE=0.0001
TRAIN_EPOCHS=50
PATIENCE=5

total=30
current=0

for stock in "${STOCKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for horizon in "${HORIZONS[@]}"; do
            current=$((current + 1))
            echo "=========================================="
            echo "Experiment $current/$total"
            echo "Model: $model | Stock: $stock | Horizon: H=$horizon"
            echo "=========================================="

            python run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path $ROOT_PATH \
                --data_path "${stock}_normalized.csv" \
                --model_id "${model}_${stock}_H${horizon}" \
                --model $model \
                --data custom \
                --features M \
                --target pct_chg \
                --seq_len $SEQ_LEN \
                --label_len $LABEL_LEN \
                --pred_len $horizon \
                --enc_in 6 \
                --dec_in 6 \
                --c_out 6 \
                --batch_size $BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --train_epochs $TRAIN_EPOCHS \
                --patience $PATIENCE \
                --use_gpu 1 \
                --gpu 0 \
                --des 'PresentationExp' \
                --itr 1

            echo "Completed: $current/$total"
            echo ""
        done
    done
done

echo "All experiments complete!"
EOF

chmod +x run_presentation_experiments.sh

# Run the experiments
./run_presentation_experiments.sh
```

**Estimated time**: 4-6 hours (depending on GPU)

**Run this OVERNIGHT** or during the day while you prepare slides.

---

## 📊 Phase 4: Analyze Results (10 minutes)

Once experiments complete:

```bash
# Generate summary tables
python analyze_results.py
```

**This creates:**
1. `experiment_results_summary.csv` - All results in one file
2. `summary_mse.csv` - MSE comparison table
3. `summary_r2.csv` - R² comparison table
4. Console output with LaTeX tables for your paper

**Open in Excel/LibreOffice** for easy copying to your presentation.

---

## 📈 What to Present Tomorrow

### **Slide 1: Methodology**
- ✅ Following Eden Modise's exact approach
- ✅ Data: NVIDIA, APPLE stocks (representative examples)
- ✅ Models: Mamba, Informer, Autoformer, FEDformer, iTransformer
- ✅ Horizons: H=10, 22, 50 (medium to long-term)
- ✅ Metrics: MSE, MAE, RMSE, R²

### **Slide 2: Data Preprocessing**
- ✅ **Percentage change target** (pct_chg) instead of raw prices
- ✅ **Normalization**: Using training statistics only (no data leakage)
- ✅ **Split**: 2215 days train / 200 days val / 100 days test
- ✅ **Features**: Open, High, Low, Close, Volume, pct_chg (6 features)

### **Slide 3: Results Table**

Example table to show:

| Model | Stock | H=10 MSE | H=10 R² | H=22 MSE | H=22 R² | H=50 MSE | H=50 R² |
|-------|-------|----------|---------|----------|---------|----------|---------|
| Mamba | NVIDIA | 0.0012 | 0.78 | 0.0015 | 0.75 | 0.0018 | 0.72 |
| Informer | NVIDIA | 0.0010 | 0.82 | 0.0013 | 0.79 | 0.0020 | 0.70 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### **Slide 4: Key Findings**
- Which model performed best overall?
- How does Mamba compare to Transformers?
- Does performance degrade for longer horizons?
- How do results compare to Eden's paper?

### **Slide 5: Next Steps**
- Complete remaining stocks (SP500, NASDAQ, etc.)
- Test additional horizons (H=3, 5, 100)
- Hyperparameter tuning with Optuna
- Statistical significance testing

---

## 🆘 Emergency Troubleshooting

### **Problem: GPU Out of Memory**
```bash
# Reduce batch size in the script:
BATCH_SIZE=16  # or even 8
```

### **Problem: Training too slow**
```bash
# Reduce number of epochs for faster results:
TRAIN_EPOCHS=20  # instead of 50
```

### **Problem: High MSE values**
**Check:**
1. Did you run data preprocessing? (`ls dataset/processed_data/`)
2. Is target set to `pct_chg`? (not `Close`)
3. Are metrics similar across models? (might be dataset characteristic)

### **Problem: Experiments not completing**
**Check progress:**
```bash
# Count how many experiments finished
ls results/ | wc -l

# Check latest results
cat result_long_term_forecast.txt
```

---

## ⚡ Absolute Minimum for Presentation

If time is **very limited**, run just this:

```bash
# 1. Preprocess data (required)
cd dataset && python prepare_data_eden_method.py

# 2. Run Mamba only on 1 stock, 3 horizons (30 min - 1 hour)
cd ../forecast-research

for horizon in 10 22 50; do
    python run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ../dataset/processed_data/ \
        --data_path NVIDIA_normalized.csv \
        --model_id Mamba_NVIDIA_H${horizon} \
        --model Mamba \
        --data custom \
        --features M \
        --target pct_chg \
        --seq_len 60 \
        --label_len 30 \
        --pred_len $horizon \
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
        --train_epochs 30 \
        --patience 5 \
        --use_gpu 1 \
        --gpu 0 \
        --des 'MinimalExp' \
        --itr 1
done

# 3. Show results
cat result_long_term_forecast.txt
```

Then explain:
- ✅ "Setup is complete and validated"
- ✅ "Here are preliminary results for Mamba model"
- ✅ "Full comparison with all 5 models is running on the cluster"
- ✅ "Will present complete results in final thesis"

---

## 📞 Final Checklist Before Presentation

- [ ] Data preprocessing completed successfully
- [ ] At least 1 model tested and working
- [ ] Results file (`result_long_term_forecast.txt`) exists
- [ ] Can explain the methodology (preprocessing, normalization, target variable)
- [ ] Can show metrics (MSE, MAE, RMSE, R²) and what they mean
- [ ] Have backup plan if experiments don't finish (show methodology + partial results)

---

## 🎯 Success Criteria

**Minimum success:**
- Data preprocessing works
- 1 model runs successfully
- Can explain Eden's methodology

**Good success:**
- Mamba model results on 2 stocks
- Comparable metrics to Eden's paper
- Clear understanding of preprocessing approach

**Excellent success:**
- All 5 models tested on 2 stocks
- Complete comparison table
- Statistical analysis ready

---

Good luck! Remember: **Process data first, test one experiment, then run overnight batch!** 🚀
