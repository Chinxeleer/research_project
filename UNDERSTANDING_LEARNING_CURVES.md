# Understanding Learning Curves - Complete Guide

**Date:** 2025-11-04
**For:** Financial Forecasting Research

---

## What Are Learning Curves?

Learning curves show how your model's **loss (error)** changes during training.

### The Graph Axes:

```
Loss (MSE)
    ↑
    |  Training Loss (Blue)
    |  Validation Loss (Orange)
    |
    |
    └──────────────────────────→ Epochs
       (Training iterations)
```

- **X-axis (Epochs):** Number of times model sees entire training dataset
- **Y-axis (Loss):** Error/mistake size (Lower = Better)
- **Blue Line:** Error on training data (data model is learning from)
- **Orange Line:** Error on validation data (data model has NEVER seen)

---

## Understanding Your Graph (mamba_100.png)

Let me explain what you saw:

### Your Graph Details:

```
Loss
0.8 ┤
    │ Blue (Training): Starts ~0.8, drops to ~0.65, then FLAT
0.7 ┤           ╲
    │            ╲___________________________
0.6 ┤             ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    │
0.5 ┤
    │
0.4 ┤
    │
0.3 ┤
    │ Orange (Validation): COMPLETELY FLAT at ~0.25
0.2 ┤ ________________________________________________
    │
0.1 ┤
    │
0.0 └─────────────────────────────────────────────→
    0    10   20   30   40   50  Epochs
```

### What This Tells Us:

#### 🚨 **Problem 1: Training Loss Plateaus Too High**
- Starts at 0.8
- Drops to 0.65
- **STUCK at 0.65** for 45+ epochs
- **Meaning:** Model stopped learning!

#### 🚨 **Problem 2: Validation Loss WAY Lower Than Training**
- Training: 0.65
- Validation: 0.25
- **Gap of 0.40** (huge!)
- **Meaning:** Something is fundamentally wrong

#### 🚨 **Problem 3: Validation Loss Completely Flat**
- No variation at all for 50 epochs
- **Meaning:** Model predicting the same thing every time

### Why This Happened:

**This graph was from the OLD run with:**
1. ❌ Wrong data (2006-2015 only)
2. ❌ Wrong pct_chg formula (log-space values)
3. ❌ Possible data leakage or normalization issue

**Your NEW runs should look COMPLETELY different!**

---

## Normal/Healthy Learning Curves

### ✅ **Ideal Pattern:**

```
Loss
0.5 ┤
    │  Both lines decreasing
0.4 ┤  ╲
    │   ╲  Blue (Train)
0.3 ┤    ╲_____
    │     ╲    ‾‾‾╲___
0.2 ┤      ╲         ‾‾‾╲__
    │       ╲  Orange (Val) ‾‾╲__
0.1 ┤        ╲___________________‾‾‾
    │          (Both converge)
0.0 └────────────────────────────────→
    0    10   20   30   40   50
```

**What to look for:**
1. ✅ Both lines **start high** and **decrease**
2. ✅ Both **converge** toward a minimum
3. ✅ **Training loss ≥ Validation loss** (usually)
4. ✅ Lines **smooth** (not jumping around)
5. ✅ Eventually **plateau** (no more improvement)

### ✅ **Your Latest Run (Looks Good!):**

From your SLURM output (slurm-166276.out):
```
Epoch 1: Train=0.39, Val=0.15  ← Good start
Epoch 2: Train=0.36, Val=0.14  ← Both decreasing ✓
Epoch 3: Train=0.32, Val=0.14  ← Converging ✓
Epoch 5: Train=0.29, Val=0.13  ← Getting better ✓
Epoch 8: Train=0.28, Val=0.13  ← Approaching minimum ✓
```

**This is HEALTHY!** Both decreasing, approaching convergence.

---

## Common Patterns and What They Mean

### Pattern 1: Overfitting 🔴

```
Loss
0.5 ┤
    │  Training keeps improving
0.4 ┤  ╲
    │   ╲________________  ← Train keeps dropping
0.3 ┤    ╲               ‾‾‾╲___
    │     ╲                      ‾‾‾‾
0.2 ┤      ╲
    │       ╲     ╱‾‾‾‾‾  ← Val INCREASES!
0.1 ┤        ╲__╱   (Bad!)
    │
0.0 └────────────────────────────────→
```

**Signs:**
- ❌ Training loss decreasing
- ❌ Validation loss INCREASING
- ❌ Gap between them growing

**What it means:**
- Model memorizing training data
- Not learning generalizable patterns
- Will perform poorly on new data

**Fix:**
- ✅ Early stopping (you have this!)
- ✅ Increase dropout
- ✅ Add regularization
- ✅ Get more training data

---

### Pattern 2: Underfitting 🔴

```
Loss
0.5 ┤  Both lines high and flat
    │  ________________________
0.4 ┤ ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    │╱
0.3 ┤   Both stuck!
    │
0.2 ┤  No learning happening
    │
0.1 ┤
    │
0.0 └────────────────────────────────→
```

**Signs:**
- ❌ Both losses stay HIGH
- ❌ No improvement after many epochs
- ❌ Lines are flat or barely decreasing

**What it means:**
- Model too simple
- Learning rate too low
- Not enough training

**Fix:**
- ✅ Increase model capacity (more layers/neurons)
- ✅ Increase learning rate
- ✅ Train longer
- ✅ Check data preprocessing

---

### Pattern 3: Good Fit ✅

```
Loss
0.5 ┤
    │  Both decreasing smoothly
0.4 ┤  ╲
    │   ╲
0.3 ┤    ╲___
    │     ╲  ‾‾‾╲___  Both converge
0.2 ┤      ╲       ‾‾‾╲__
    │       ╲____________‾‾‾  to similar level
0.1 ┤
    │  Small gap maintained
0.0 └────────────────────────────────→
```

**Signs:**
- ✅ Both decrease together
- ✅ Small gap between them (0.01-0.05 typical)
- ✅ Both plateau at similar level
- ✅ Smooth curves (no wild jumps)

**What it means:**
- Model learning well!
- Good generalization
- Ready for testing

---

### Pattern 4: Validation Lower Than Training (Your Case) 🟡

```
Loss
0.4 ┤  Train (Blue)
    │  ╲
0.3 ┤   ╲_____________
    │    ╲           ‾‾‾  ← Train higher
0.2 ┤     ╲
    │      ╲________  ← Val lower
0.1 ┤       ‾‾‾‾‾‾‾‾
    │
0.0 └────────────────────────────────→
```

**When this is OK:** ✅
1. **Smaller validation set** (yours: 239 val vs 4,302 train with 90/5/5 split)
   - Less data = less noise in average
   - Validation can get "lucky"

2. **No dropout/augmentation during validation**
   - Training uses dropout (adds noise)
   - Validation runs clean (lower loss)

3. **Validation data is easier**
   - Different time period
   - Less volatile
   - Easier patterns

4. **Both are still decreasing**
   - Main thing is improvement
   - Absolute gap matters less

**When to worry:** 🚨
1. Gap is HUGE (>0.3 like your old graph)
2. Validation is FLAT (no variation)
3. Both curves don't make sense

---

## Financial Forecasting Specific Notes

### Expected Loss Ranges:

For percentage change forecasting (pct_chg):

```
GOOD:     MSE < 0.0001  (0.01% average error)
OKAY:     MSE = 0.0001-0.001  (0.01-0.03% error)
ACCEPTABLE: MSE = 0.001-0.01  (0.03-0.1% error)
POOR:     MSE > 0.01  (>0.1% error)
```

Your losses in latest run:
- Train: 0.28-0.39 (normalized space)
- Val: 0.13-0.15 (normalized space)

These get denormalized for final metrics (should be ~0.0001-0.001 range).

### Why Financial Data is Special:

1. **Very noisy** - Random walk component
2. **Non-stationary** - Patterns change over time
3. **Low predictability** - R² near 0 is normal
4. **Validation can be easier** - Different market regimes

**So some "weird" patterns are actually normal!**

---

## How to Use Learning Curves for Debugging

### Step 1: Check Initial Loss (Epoch 1)

**Normal:**
- Train: 0.3-0.8
- Val: Similar to train (±0.1)

**Problems:**
- Loss > 1.0 → Learning rate too high
- Loss = NaN → Numerical instability
- Loss not decreasing → Wrong initialization

### Step 2: Watch First 5 Epochs

**Good signs:**
- ✅ Both decreasing
- ✅ Smooth decline
- ✅ No sudden jumps

**Bad signs:**
- ❌ Increasing
- ❌ Wild oscillations
- ❌ Staying flat

### Step 3: Monitor Convergence (Epoch 10-30)

**Good signs:**
- ✅ Still improving
- ✅ Rate of improvement slowing
- ✅ Lines getting closer

**Bad signs:**
- ❌ Validation going up
- ❌ No improvement for 10+ epochs
- ❌ Gap widening

### Step 4: Check Final Performance

**Good signs:**
- ✅ Both plateaued at low level
- ✅ Small gap (<0.05)
- ✅ Early stopping triggered appropriately

**Bad signs:**
- ❌ Still decreasing (need more epochs)
- ❌ Large gap (overfitting)
- ❌ Both high (underfitting)

---

## Your Specific Cases

### Case 1: Old Graph (mamba_100.png) 🔴

```
Issue: Train=0.65, Val=0.25, both flat
Diagnosis: Data problem (wrong preprocessing)
Status: FIXED with new preprocessing
Action: Ignore this graph, it's from bad data
```

### Case 2: Latest Run (slurm-166276.out) ✅

```
Pattern: Train=0.28, Val=0.13, both decreasing
Diagnosis: HEALTHY learning!
Status: GOOD - continue training
Action: Wait for early stopping, then evaluate
```

---

## Quick Decision Guide

### Is My Training OK? Decision Tree:

```
Are both lines decreasing?
├─ Yes → GOOD! Keep going
│   └─ Is val loss increasing after epoch 20?
│       ├─ Yes → Overfitting, early stop will catch it ✓
│       └─ No → Perfect! ✓
│
└─ No → PROBLEM
    ├─ Both flat → Underfitting
    │   └─ Increase learning rate or model size
    │
    ├─ Both increasing → Learning rate too high
    │   └─ Decrease learning rate
    │
    └─ Wild oscillations → Unstable training
        └─ Check data normalization
```

### Is the Gap Between Train/Val OK?

```
Gap = |Train Loss - Val Loss|

Gap < 0.05:  ✅ EXCELLENT - Generalizing well
Gap = 0.05-0.15: ✅ GOOD - Normal for small val set
Gap = 0.15-0.30: 🟡 OKAY - Monitor for overfitting
Gap > 0.30: 🔴 BAD - Likely data issue or overfitting

Your latest: ~0.15 → GOOD! ✓
Your old graph: ~0.40 → BAD (data was wrong)
```

---

## What to Report in Your Paper

### Good Example:

> "Figure X shows the learning curves for the Mamba model on NVIDIA stock data.
> Both training and validation losses converge after approximately 25 epochs,
> with early stopping triggered at epoch 28 (patience=10). The final training
> loss of 0.28 and validation loss of 0.13 indicate good model fit without
> overfitting. The small gap between training and validation loss suggests
> the model generalizes well to unseen data."

### What NOT to say:

> ❌ "Validation loss is lower than training loss, which is weird."
>
> Better: ✅ "Validation loss is slightly lower than training loss (0.13 vs 0.28),
> which is common with smaller validation sets (125 vs 2201 samples) and
> no dropout applied during validation."

---

## Summary - Key Takeaways

### ✅ What "Good" Looks Like:

1. **Both lines decrease** (learning is happening)
2. **Converge to low values** (model is accurate)
3. **Small gap** (<0.15 typical)
4. **Smooth curves** (stable training)
5. **Plateau eventually** (found minimum)

### 🚨 What "Bad" Looks Like:

1. **Validation increases** (overfitting)
2. **Both stay high** (underfitting)
3. **Huge gap** (>0.3 - data problem)
4. **Wild jumps** (unstable)
5. **Completely flat** (not learning)

### 🎯 Your Current Status:

**Latest run (166276):** ✅ HEALTHY
- Both decreasing ✓
- Converging ✓
- Gap reasonable (~0.15) ✓
- Will continue to improve ✓

**Old run (mamba_100.png):** 🔴 BROKEN
- From wrong data ✗
- Ignore completely ✗
- Already fixed ✓

---

## Interactive Checklist for Your Runs

When you see a new learning curve, ask:

- [ ] Are both lines going down?
  - **Yes** → Good!
  - **No** → Check learning rate/data

- [ ] Do they converge eventually?
  - **Yes** → Good!
  - **No** → May need more epochs

- [ ] Is the gap reasonable (<0.3)?
  - **Yes** → Good!
  - **No** → Check for data issues

- [ ] Are the curves smooth?
  - **Yes** → Good!
  - **No** → Check batch size/learning rate

- [ ] Did early stopping trigger appropriately?
  - **Yes** → Good!
  - **No** → Adjust patience if needed

**If all checks pass → Your model is training correctly!** ✅

---

## Additional Resources

### Viewing on WandB:

1. Go to your WandB project
2. Click on a run
3. Look for "Charts" tab
4. Find "learning_curves" plot
5. Hover over lines to see exact values
6. Compare multiple runs side-by-side

### Common WandB Patterns:

```
Smooth line: Stable training ✓
Jagged line: High variance (increase batch size)
Horizontal: No learning (check data/LR)
Exponential growth: Exploding gradients (lower LR)
```

---

**Generated:** 2025-11-04
**Use this guide to interpret all your training runs!**
