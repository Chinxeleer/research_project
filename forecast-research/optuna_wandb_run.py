"""
Optuna Hyperparameter Tuning - Corrected for Time-Series-Library
Following Eden Modise's Research Paper with Time-Series-Library parameter names

IMPORTANT: Time-Series-Library uses different parameter names than the paper.
This script maps paper's parameters to Time-Series-Library's actual parameters:

Paper's Table VIII          →    Time-Series-Library Equivalent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
lr (learning rate)          →    --learning_rate
hidden_dim                  →    --d_model (model dimension)
num_layers                  →    --e_layers (encoder layers)

Additional Time-Series-Library parameters we CAN tune:
- d_ff (feedforward dimension, usually 4x d_model)
- n_heads (number of attention heads)
- dropout
"""

import optuna
from optuna.trial import Trial
import argparse
import torch
import numpy as np
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import os
import json
import warnings
warnings.filterwarnings('ignore')


def get_base_args():
    """Base arguments for Time-Series-Library"""
    parser = argparse.ArgumentParser(description='Optuna Tuning - Paper Replication')
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='optuna_tune')
    parser.add_argument('--model', type=str, default='Mamba',
                        help='Model: Mamba, TimesNet, Transformer, Autoformer, iTransformer, DLinear')
    
    # Data config - UPDATE THESE
    parser.add_argument('--data', type=str, default='financial')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='your_financial_data.csv')  # ⚠️ UPDATE
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='pct_chg')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)
    
    # Model dimensions - UPDATE based on your data
    parser.add_argument('--enc_in', type=int, default=7)  # ⚠️ UPDATE: number of features
    parser.add_argument('--dec_in', type=int, default=7)  # ⚠️ UPDATE: same as enc_in
    parser.add_argument('--c_out', type=int, default=7)   # ⚠️ UPDATE: same as enc_in
    
    # Model architecture (will be tuned)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true', default=False)
    
    # Training config
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='optuna_paper')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1')
    
    # Additional
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    
    # Optuna config
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--study_name', type=str, default='paper_replication')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_paper.db')
    
    # W&B (optional)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='financial-optuna-paper')
    parser.add_argument('--wandb_entity', type=str, default=None)
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    return args


def objective(trial: Trial, base_args):
    """
    Objective function mapping paper's hyperparameters to Time-Series-Library
    
    Paper's Table VIII        →    Time-Series-Library
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lr: 1e-5 to 1e-2          →    --learning_rate
    hidden_dim: 8 to 64       →    --d_model
    num_layers: 1 to 4        →    --e_layers
    """
    
    # Copy base args
    args = argparse.Namespace(**vars(base_args))
    
    # ========== MAP PAPER'S PARAMETERS TO TIME-SERIES-LIBRARY ==========
    
    # Paper's "lr" → Time-Series-Library's "learning_rate"
    args.learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    
    # Paper's "hidden_dim" → Time-Series-Library's "d_model"
    args.d_model = trial.suggest_int('hidden_dim', 8, 64)
    
    # Paper's "num_layers" → Time-Series-Library's "e_layers"
    args.e_layers = trial.suggest_int('num_layers', 1, 4)
    
    # Set d_ff based on d_model (standard practice: d_ff = 4 * d_model)
    args.d_ff = args.d_model * 4
    
    # Keep decoder layers fixed
    args.d_layers = 1
    
    # Adjust n_heads based on d_model (d_model must be divisible by n_heads)
    # Common head sizes: 32, 64
    if args.d_model >= 64:
        args.n_heads = 8
    elif args.d_model >= 32:
        args.n_heads = 4
    elif args.d_model >= 16:
        args.n_heads = 2
    else:
        args.n_heads = 1
    
    
    # ========== END OF PARAMETER MAPPING ==========
    
    # Update identifiers
    args.model_id = f'trial_{trial.number}'
    args.des = f'trial_{trial.number}'
    
    # Print trial info
    print("\n" + "="*70)
    print(f"Trial {trial.number}")
    print("="*70)
    print("Paper Parameters          →    Time-Series-Library Parameters")
    print("-"*70)
    print(f"lr: {args.learning_rate:.6f}       →    learning_rate: {args.learning_rate:.6f}")
    print(f"hidden_dim: {args.d_model}           →    d_model: {args.d_model}")
    print(f"num_layers: {args.e_layers}           →    e_layers: {args.e_layers}")
    print("-"*70)
    print(f"Auto-adjusted: n_heads={args.n_heads}, d_ff={args.d_ff}")
    print("="*70 + "\n")
    
    # Initialize W&B if enabled
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"trial_{trial.number}",
            config={
                'trial': trial.number,
                'lr': args.learning_rate,
                'hidden_dim': args.d_model,
                'num_layers': args.e_layers,
                'd_model': args.d_model,
                'e_layers': args.e_layers,
                'n_heads': args.n_heads,
                'd_ff': args.d_ff
            },
            reinit=True,
            tags=['optuna', 'paper_replication', args.model]
        )
    
    try:
        # Create experiment
        exp = Exp_Long_Term_Forecast(args)
        
        # This is typically done in exp_long_term_forecasting.py in _select_optimizer()
        # See note below on how to handle this
        
        print(f'>>>>>> Training trial {trial.number} >>>>>>')
        exp.train(setting=f'trial_{trial.number}')
        
        print(f'>>>>>> Validating trial {trial.number} >>>>>>')
        mse, mae = exp.test(setting=f'trial_{trial.number}', test=0)
        
        # Log to W&B
        if args.use_wandb:
            wandb.log({
                'final/val_mse': mse,
                'final/val_mae': mae
            })
            wandb.finish()
        
        print(f"\n✅ Trial {trial.number} completed - MSE: {mse:.6f}, MAE: {mae:.6f}\n")
        
        return mse
        
    except Exception as e:
        print(f"\n❌ Trial {trial.number} failed: {e}\n")
        if args.use_wandb:
            wandb.finish()
        return float('inf')


def run_optimization(args):
    """Run Optuna optimization"""
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5
        )
    )
    
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER TUNING")
    print("Paper Parameters → Time-Series-Library Parameters")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data_path}")
    print(f"Trials: {args.n_trials}")
    print("\nParameter Mapping:")
    print("  Paper's 'lr'         → Time-Series-Library's 'learning_rate'")
    print("  Paper's 'hidden_dim' → Time-Series-Library's 'd_model'")
    print("  Paper's 'num_layers' → Time-Series-Library's 'e_layers'")
    print("="*70 + "\n")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=None,
        n_jobs=1,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best MSE: {study.best_trial.value:.6f}")
    print("\nBest Hyperparameters:")
    print("-"*70)
    
    # Map back to paper's parameter names for comparison
    best_params = study.best_trial.params
    print(f"lr (learning_rate):  {best_params['lr']:.10f}")
    print(f"hidden_dim (d_model): {best_params['hidden_dim']}")
    print(f"num_layers (e_layers): {best_params['num_layers']}")
    print("="*70 + "\n")
    
    # Save results
    results_dir = './optuna_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save in both paper format and Time-Series-Library format
    results = {
        'paper_format': {
            'lr': best_params['lr'],
            'hidden_dim': best_params['hidden_dim'],
            'num_layers': best_params['num_layers']
        },
        'time_series_library_format': {
            'learning_rate': best_params['lr'],
            'd_model': best_params['hidden_dim'],
            'e_layers': best_params['num_layers'],
            'd_ff': best_params['hidden_dim'] * 4
        },
        'best_mse': float(study.best_trial.value),
        'trial_number': study.best_trial.number
    }
    
    with open(f'{results_dir}/{args.study_name}_best_params.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Results saved: {results_dir}/{args.study_name}_best_params.json")
    
    # Create visualization
    try:
        import plotly
        
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f'{results_dir}/{args.study_name}_history.html')
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f'{results_dir}/{args.study_name}_importances.html')
        
        print(f"✅ Visualizations saved: {results_dir}/")
    except ImportError:
        print("⚠️  Install plotly: pip install plotly")
    
    return study


if __name__ == '__main__':
    print("\n" + "="*70)
    print("="*70 + "\n")
    
    args = get_base_args()
    
    # Validate
    print("📋 Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data_path}")
    print(f"  Target: {args.target}")
    print(f"  Trials: {args.n_trials}")
    
    print("\n" + "="*70)
    input("Press ENTER to start (Ctrl+C to cancel)...")
    
    study = run_optimization(args)
    
