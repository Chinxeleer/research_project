# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a time-series forecasting research project for financial data prediction using deep learning models. The repository contains:
- **forecast-research/**: A modified fork of the Time-Series-Library with 25+ state-of-the-art forecasting models
- **dataset/**: Financial dataset processing and normalization utilities

## Repository Structure

```
research_project/
├── forecast-research/          # Main forecasting library
│   ├── models/                 # 25+ forecasting models (TimesNet, iTransformer, Mamba, etc.)
│   ├── exp/                    # Experiment classes for different tasks
│   ├── data_provider/          # Data loaders and factories
│   ├── layers/                 # Custom neural network layers
│   ├── utils/                  # Metrics, tools, augmentation, wandb logger
│   ├── scripts/                # Shell scripts for training different models
│   ├── run.py                  # Main training script
│   ├── run_wandb.py            # Training with Weights & Biases logging
│   ├── optuna_tune.py          # Hyperparameter tuning with Optuna
│   ├── optuna_wandb_run.py     # Combined Optuna + WandB tuning
│   └── train_best_model.py     # Train final model with best hyperparameters
└── dataset/
    ├── normalize.py            # Data preprocessing script
    └── processed_data/         # Processed datasets
```

## Common Commands

### Training Models

**Basic training:**
```bash
cd forecast-research
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id my_model \
  --model TimesNet \
  --data custom \
  --root_path ../dataset/processed_data/ \
  --data_path your_data.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001
```

**Training with WandB logging:**
```bash
python run_wandb.py \
  --model TimesNet \
  --data custom \
  --root_path ../dataset/ \
  --data_path your_data.csv \
  --use_wandb \
  --wandb_project "financial-forecasting"
```

**Hyperparameter tuning with Optuna:**
```bash
python optuna_tune.py \
  --model TimesNet \
  --n_trials 50 \
  --study_name my_study
```

**Optuna tuning with WandB:**
```bash
python optuna_wandb_run.py \
  --model TimesNet \
  --n_trials 50
```

**Train final model with best parameters:**
```bash
python train_best_model.py \
  --model TimesNet \
  --study_name my_study \
  --train_epochs 100
```

**Testing only (no training):**
```bash
python run.py \
  --is_training 0 \
  --task_name long_term_forecast \
  --model_id test_model \
  --model TimesNet \
  [... other parameters ...]
```

### Data Preprocessing

**Normalize financial datasets:**
```bash
cd dataset
python normalize.py
```

This script:
- Cleans data (removes zero/negative values, outliers, missing values)
- Applies log transformations to prices (log) and volume (log1p)
- Applies StandardScaler normalization
- Saves processed datasets to `processed_data/`

### Using Training Scripts

Shell scripts for common benchmarks are provided in `forecast-research/scripts/`:
```bash
cd forecast-research/scripts/long_term_forecast/ECL_script
bash TimesNet.sh
```

## Architecture Guide

### Task Types

The codebase supports 5 different time-series tasks:
1. **long_term_forecast** - Multi-step ahead forecasting (most common)
2. **short_term_forecast** - Single-step forecasting
3. **imputation** - Fill missing values
4. **classification** - Time-series classification
5. **anomaly_detection** - Detect anomalies in sequences

Each task has a corresponding experiment class in `exp/`:
- `exp_long_term_forecasting.py`
- `exp_short_term_forecasting.py`
- `exp_imputation.py`
- `exp_classification.py`
- `exp_anomaly_detection.py`

### Model Architecture

All models inherit from `exp/exp_basic.py` and follow this pattern:
1. Models are defined in `models/` directory (one file per model)
2. Each model implements a `Model` class with standard PyTorch `forward()` method
3. Model selection happens in `exp_basic.py` via the `model_dict`

### Data Flow

1. **Data Loading**: `data_provider/data_factory.py` routes to appropriate dataset class
2. **Dataset Classes**: `data_provider/data_loader.py` contains dataset implementations
   - `Dataset_ETT_hour`, `Dataset_Custom`, etc.
   - Each handles train/val/test splitting, scaling, and windowing
3. **Windowing**: Uses sliding window approach with:
   - `seq_len`: Input sequence length
   - `label_len`: Overlap between input and prediction (for decoder-based models)
   - `pred_len`: Prediction horizon

### Key Configuration Parameters

**Data parameters:**
- `--data`: Dataset type (custom, ETTh1, ETTm1, etc.)
- `--root_path`: Root directory of data files
- `--data_path`: Specific CSV file name
- `--features`: M (multivariate), S (univariate), MS (multivariate predict univariate)
- `--target`: Target column name for S/MS tasks
- `--freq`: Time frequency (h=hourly, d=daily, w=weekly, etc.)

**Model dimensions:**
- `--enc_in`: Number of input features (must match CSV columns minus date)
- `--dec_in`: Decoder input size (usually same as enc_in)
- `--c_out`: Number of output features
- `--d_model`: Model hidden dimension
- `--d_ff`: Feed-forward dimension
- `--n_heads`: Number of attention heads
- `--e_layers`: Number of encoder layers
- `--d_layers`: Number of decoder layers

**Sequence parameters:**
- `--seq_len`: Input sequence length
- `--label_len`: Decoder start token length (usually seq_len // 2)
- `--pred_len`: Prediction horizon

**Model-specific:**
- `--top_k`, `--num_kernels`: For TimesNet
- `--expand`, `--d_conv`: For Mamba
- `--patch_len`: For PatchTST, iTransformer

### Experiment Workflow

1. **Initialize Experiment**: Creates appropriate `Exp_*` class based on task
2. **Build Model**: Loads model architecture from `models/`
3. **Get Data**: Loads and preprocesses data using `data_provider`
4. **Training Loop** (`exp.train(setting)`):
   - Trains model with early stopping
   - Saves checkpoints to `checkpoints/{setting}/`
   - Logs metrics (loss, learning rate)
5. **Testing** (`exp.test(setting, test=1)`):
   - `test=0`: Validation set
   - `test=1`: Test set
   - Returns MSE and MAE metrics
   - Saves predictions to `results/{setting}/`

### Optuna Integration

The hyperparameter tuning workflow:
1. **Define search space** in `optuna_tune.py::objective()`
2. **Run trials**: Each trial trains a model and evaluates on validation set
3. **Pruning**: MedianPruner stops unpromising trials early
4. **Best params**: Saved to `optuna_results/{study_name}_best_params.json`
5. **Visualizations**: HTML plots saved to `optuna_results/`
6. **Final training**: Use `train_best_model.py` with discovered parameters

### WandB Integration

Logging is handled by `utils/wandb_logger.py`:
- Logs hyperparameters at start
- Tracks train/val loss per epoch
- Logs final test metrics
- Saves learning curves and predictions
- Enable with `--use_wandb` and `--wandb_project` flags

## Dataset Format

CSV files should have this structure:
```csv
Date,feature1,feature2,feature3,...
2020-01-01,1.2,3.4,5.6,...
2020-01-02,1.3,3.5,5.7,...
...
```

**Important:**
- First column should be named `date` or `Date`
- Remaining columns are features
- No missing dates (or handle with imputation task)
- For financial data, run through `dataset/normalize.py` first

## Available Models

Key models implemented (see `models/` directory):
- **TimesNet**: Multi-period time-series analysis
- **iTransformer**: Inverted transformer (channel-wise attention)
- **PatchTST**: Patching and channel-independent attention
- **Mamba**: State-space model with selective mechanism
- **DLinear**: Simple linear model baseline
- **Autoformer**: Auto-correlation mechanism
- **FEDformer**: Frequency-enhanced decomposition
- **Transformer**: Standard transformer
- **Informer**: Efficient transformer for long sequences
- Plus 15+ more variants

## GPU Configuration

**Single GPU:**
```bash
--use_gpu --gpu 0
```

**Multiple GPUs:**
```bash
--use_multi_gpu --devices 0,1,2,3
```

**CPU/MPS (Apple Silicon):**
```bash
--gpu_type mps  # or 'cuda' for NVIDIA
```

The code auto-detects available hardware and falls back appropriately.

## Results and Checkpoints

**Checkpoints:** Saved to `forecast-research/checkpoints/{setting}/`
- Contains model weights (`checkpoint.pth`)
- Used for resuming training or testing

**Results:** Saved to `forecast-research/results/{setting}/`
- `pred.npy`: Predictions
- `true.npy`: Ground truth
- `metrics.txt`: MSE, MAE scores

**Optuna results:** Saved to `forecast-research/optuna_results/`
- Best parameters JSON
- Visualization HTML files
- Study database (`optuna_study.db`)

## Contributing to TSlib

This project is based on Time-Series-Library. To add new models:
1. Create model file in `models/` directory
2. Implement `Model` class with standard forward pass
3. Add to `model_dict` in `exp/exp_basic.py`
4. Test with all 5 task types
5. Provide training scripts in `scripts/`

See `CONTRIBUTING.md` for full guidelines.

## Notes

- **Random seed**: Fixed to 2021 in `run.py` for reproducibility
- **Early stopping**: Default patience is 3 epochs (increase for final training)
- **Data split**: Default is 80/10/10 train/val/test (defined in dataset classes)
- **Augmentation**: Supports 14 types (jitter, scaling, warping, etc.) via `--augmentation_ratio` flag
- **Memory**: For Mamba models, ensure d_inner (d_model * expand) < 256 to avoid OOM
