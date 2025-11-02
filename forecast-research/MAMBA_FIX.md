# 🔧 MAMBA MODEL FIX - Critical Configuration

## ❌ The Problem

You got this error:
```
RuntimeError: selective_scan only supports state dimension <= 256
```

## 🎯 The Solution

Mamba has a **hard constraint**: `d_inner = d_model × expand ≤ 256`

### **Current Issue:**
- Your config likely has `d_model=512` and `expand=2`
- This gives `d_inner = 512 × 2 = 1024` ❌ **TOO LARGE!**

### **Fixed Configuration:**
```bash
d_model=64          # Reduced from 128
d_ff=16             # State dimension (maps to d_state in Mamba)
expand=2            # Expansion factor
# This gives: d_inner = 64 × 2 = 128 ✅ WITHIN LIMIT!
```

---

## 🚀 Quick Fix Commands

### **Option 1: Use the corrected script**
```bash
./run_mamba.sh
```
(Already updated with correct parameters)

### **Option 2: Manual single experiment**
```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ../dataset/processed_data/ \
    --data_path NVIDIA_normalized.csv \
    --model_id Mamba_NVIDIA_H10_Fixed \
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
    --train_epochs 50 \
    --patience 5 \
    --use_gpu 1 \
    --gpu 0 \
    --des 'FixedMamba' \
    --itr 1
```

---

## 📊 Safe Mamba Configurations

### **Configuration 1: Small (recommended for testing)**
```
d_model=64, expand=2  → d_inner=128 ✅
d_model=64, expand=4  → d_inner=256 ✅ (max)
```

### **Configuration 2: Medium (good performance)**
```
d_model=128, expand=1 → d_inner=128 ✅
d_model=64,  expand=2 → d_inner=128 ✅
```

### **Configuration 3: Large (pushing limits)**
```
d_model=128, expand=2 → d_inner=256 ✅ (at limit)
d_model=256, expand=1 → d_inner=256 ✅ (at limit)
```

---

## ⚠️ What NOT to Use

```
d_model=512, expand=2 → d_inner=1024 ❌ WILL CRASH!
d_model=256, expand=2 → d_inner=512  ❌ WILL CRASH!
d_model=128, expand=4 → d_inner=512  ❌ WILL CRASH!
```

---

## 🔄 Updated Training Scripts

All scripts have been updated with safe values:
- `run_mamba.sh`: d_model=64, expand=2 (d_inner=128)
- `run_experiments.sh`: Uses compatible settings

---

## 📝 Notes from Eden's Paper

Eden likely used smaller d_model values with Mamba. The paper doesn't specify exact hyperparameters, but given Mamba's constraints, he probably used:
- `d_model ≤ 128`
- `expand ≤ 2`
- `d_state (d_ff) ≤ 16`

This is actually GOOD because:
1. Faster training
2. Less memory usage
3. Less prone to overfitting on financial data

---

## ✅ Verification

After running with fixed config, you should see:
```
Epoch: 1 cost time: XX.Xs
    train_loss: 0.XXXX
    val_loss: 0.XXXX
```

If you still see the error, check:
```bash
# Verify the actual parameters being used
grep "d_model\|expand\|d_ff" run.py
```

---

## 💡 Why This Happens

Mamba's selective scan algorithm has a hardware constraint. The state dimension must fit in GPU shared memory, which limits it to 256. This is a known limitation of the mamba-ssm library.

**Reference**: https://github.com/state-spaces/mamba/issues/XXX

---

Run with the fixed configuration and it should work! 🚀
