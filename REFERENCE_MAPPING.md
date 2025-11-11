# Reference Number Corrections

## Current Problems Identified:

### Major Mismatches:
- [5] Currently: Zou (survey) → Should be: Zhou (Informer)
- [6] Currently: Fama → Should be: Wu (Autoformer)
- [7] Currently: Wu (Price graphs) → Should be: Zhou (FEDformer)
- [8] Currently: Soni → Should be: Liu (iTransformer)
- [9] Currently: Jiang → Should be: Zeng (Are transformers effective)
- [11] Currently: Bengio → Should be: Gu & Dao (Mamba) [cited for Mamba in intro]
- [12] Currently: Orvieto → Should be: Shi (MambaStock) [cited for MambaStock]

### Correct Papers at Wrong Numbers:
- Zhou (Informer) is at [30] but should be [5]
- Wu (Autoformer) is at [31] but should be [6]
- Zhou (FEDformer) is at [32] but should be [7]
- Liu (iTransformer) is at [33] but should be [8]
- Zeng is at [34] but should be [9]
- Gu & Dao (Mamba) is at [38] but should be [11]
- Shi (MambaStock) is at [39] but should be [12]

### Duplicates to Remove:
- [12] and [13] both Orvieto → Keep one at appropriate number
- [14] and [19] both Hochreiter LSTM → Keep one at [14]
- [8] and [20] both Soni → Keep one at [20]
- [21] and [23] both Vaswani Attention → Keep one at [21]

### Missing References:
- [19] should be for Bidirectional RNNs (Schuster & Paliwal OR just cite Hochreiter again for context)
- [29] cited for Diebold-Mariano test but references Taloma (wrong paper)
- [30] should be Gu S4 paper
- [32] cited for "SSMs time-series forecasting" - need to identify correct paper

## Corrected Reference Order:

[1] Alamu (2024) - Stock price prediction
[2] Li (2020) - Generating realistic stock market order streams
[3] Latif (2024) - Enhanced prediction PLSTM
[4] Khan (2023) - Performance comparison of ML models
[5] Zhou (2021) - **Informer** ← FIX
[6] Wu (2021) - **Autoformer** ← FIX
[7] Zhou (2022) - **FEDformer** ← FIX
[8] Liu (2024) - **iTransformer** ← FIX
[9] Zeng (2023) - **Are transformers effective** ← FIX
[10] Elman (1990) - Finding structure in time (RNNs)
[11] Gu & Dao (2023) - **Mamba** ← FIX
[12] Shi (2024) - **MambaStock** ← FIX
[13] Orvieto (2023) - Resurrecting RNNs
[14] Hochreiter (1997) - LSTM
[15] Zaremba (2014) - RNN regularization
[16] Cho (2014) - GRU
[17] Chung (2014) - Empirical evaluation of GRUs
[18] Koutnik (2014) - Clockwork RNN
[19] Hochreiter (1997) - LSTM (can cite again for bidirectional context)
[20] Soni (2022) - ML approaches in stock price prediction
[21] Vaswani (2017) - Attention is all you need
[22] Vaswani (2018) - Tensor2tensor
[23] Liu (2024) - Genbench (genomics)
[24] See (2017) - Summarization with pointer-generator
[25] Devlin (2019) - BERT
[26] Radford (2018) - GPT
[27] Peng (2024) - Limitations of transformer
[28] (Need to check what's cited here)
[29] (Diebold-Mariano - need proper reference OR remove if not in .bib)
[30] Gu (2022) - S4 (Efficiently modeling long sequences)
[31] Alaa (2019) - Attentive state-space modeling
[32] (Need to identify SSM time-series forecasting paper)
[33] Huang (2024) - Decision mamba

## Additional Papers in .bib Not Properly Used:
- Zou (2022) - Survey (currently misplaced at [5])
- Fama (1970) - Efficient markets (currently misplaced at [6])
- Wu (2022) - Price graphs (currently misplaced at [7])
- Jiang (2021) - Applications of DL (currently misplaced at [9])
- Bengio (1994) - Vanishing gradient (currently at [11], needs new number)
- Taloma (2024) - Concrete dense network (currently at [29], wrong paper)

## Citations That Need Renumbering:

### Introduction (lines 19-29):
- [1] ✓ correct
- [2, 3] ✓ correct
- [7, 8, 9] → These should cite ML feature engineering papers. Currently Wu Price graphs, Soni, Jiang. May be acceptable but check context.
- [19, 20] → Should be [14, 16] for LSTM and GRU
- [21] ✓ correct
- [5, 6, 7, 8, 9] → Need complete renumbering as above
- [28] → Check if correct
- [30] ✓ will be correct after renumbering
- [11] → Will be correct after renumbering
- [12] → Will be correct after renumbering

### Literature Review (Section 2):
Extensive renumbering needed throughout based on above corrections.
