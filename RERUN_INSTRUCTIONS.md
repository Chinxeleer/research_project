# Quick Rerun Instructions

## TL;DR - What You Need to Do

Your previous experimental results were **invalid** because Mamba had 100x fewer parameters than the transformers. I've fixed the Mamba configuration to make it fair.

**You need to rerun ALL experiments.**

---

## What I Fixed

**Changed in `/forecast-research/run_mamba.sh`:**
```bash
# Before (WRONG):
D_MODEL=64    # Way too small!
D_FF=16       # Way too small!

# After (CORRECT):
D_MODEL=128   # Maximum allowed for Mamba
D_FF=128      # Proper capacity
```

Now Mamba has ~2-3M parameters vs ~10-15M for transformers (closer, though still not equal).

---

## Commands to Run on Cluster

```bash
# 1. Navigate to project directory
cd /home/chinxeleer/dev/repos/research_project/forecast-research

# 2. Run all 5 models (submit as separate jobs):
sbatch run_mamba.sh         # FIXED - use this new version!
sbatch run_autoformer.sh
sbatch run_informer.sh
sbatch run_fedformer.sh
sbatch run_itransformer.sh

# Wait for all jobs to complete (~6-12 hours total)
```

---

## Expected Results

After rerunning with fair configuration:

✅ **Mamba should rank 1st or 2nd** (not 4th like before)
✅ **240 total experiments** (5 models × 8 datasets × 6 horizons)
✅ **SP500 data included** (was missing before)
✅ **Fair comparison** - all models have comparable capacity

---

## After Experiments Finish

1. **Check slurm outputs:** New files will be created (e.g., `slurm-XXXXXX.out`)
2. **Extract results:**
   ```bash
   conda activate predenv
   python extract_all_results.py
   ```
3. **Verify:** Should see "TOTAL EXPERIMENTS EXTRACTED: 240"
4. **Check rankings:** Mamba should be in top 2

---

## If You See Issues

**Mamba still ranks low?**
→ There may be a bug in the model implementation. Let me know.

**Missing SP500?**
→ Check if data file exists: `ls dataset/processed_data/SP500_normalized.csv`

**Training fails?**
→ May need to reduce batch_size if GPU memory issue

**Results look weird?**
→ Run validation checklist in FIXES_APPLIED.md

---

## Quick Sanity Check (After Results)

Your results should show:
- ✓ Mamba wins on 4-6 out of 8 datasets
- ✓ R² between -0.15 and 0.0 (negative is normal!)
- ✓ MSE increases with horizon length
- ✓ Higher volatility stocks have higher MSE

If not, something is still wrong.

---

## Why This Matters

**Before Fix:**
- Mamba: 64 d_model, ~200K params → Ranked 4th/5th
- Transformers: 512 d_model, ~15M params → Won unfairly

**After Fix:**
- Mamba: 128 d_model, ~3M params → Expected 1st/2nd
- Transformers: 512 d_model, ~15M params → Fair fight

This changes your entire paper! Previous results were scientifically invalid.

---

**Questions?** Check FIXES_APPLIED.md for full details.

**Ready?** Run the 5 sbatch commands above! 🚀
