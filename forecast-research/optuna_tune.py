"""
Optuna Hyperparameter Tuning for Time-Series-Library
Save this entire file as: optuna_tune.py (in Time-Series-Library root folder)

Usage:
    python optuna_tune.py --model TimesNet --n_trials 50
"""

import optuna
from optuna.trial import Trial
import argparse
import torch
import gc
import numpy as np
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import os
import warnings
warnings.filterwarnings('ignore')


def get_base_args():
    """Base arguments that don't change during tuning"""
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Tuning')
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='optuna_tune')
    parser.add_argument('--model', type=str, default='TimesNet',
                        help='model name: TimesNet, Transformer, Autoformer, iTransformer, Mamba, DLinear')
    
    # Data loader - UPDATE THESE FOR YOUR DATA
    parser.add_argument('--data', type=str, default='financial')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='your_financial_data.csv')  # ⚠️ CHANGE THIS
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task: M=multivariate, S=univariate')
    parser.add_argument('--target', type=str, default='close')  # ⚠️ CHANGE THIS
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Forecasting task - these will be tuned
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)
    
    # Model parameters - UPDATE THESE TO MATCH YOUR FEATURE COUNT
    parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')  # ⚠️ CHANGE THIS
    parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')  # ⚠️ CHANGE THIS
    parser.add_argument('--c_out', type=int, default=5, help='output size')  # ⚠️ CHANGE THIS
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_conv', type=int, default=4)
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
    
    # Optimization - these will be tuned
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='optuna_tuning')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)
    
    # GPU
    #parser.add_argument('--use_gpu', type=bool, default=True)
    #parser.add_argument('--gpu', type=int, default=0)
    #parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    #parser.add_argument('--devices', type=str, default='0')



    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    
    # Additional args
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)
    parser.add_argument('--augmentation_ratio', type=int, default=0)

    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Optuna specific
    parser.add_argument('--n_trials', type=int, default=50,
                        help='number of optuna trials')
    parser.add_argument('--study_name', type=str, default='financial_forecasting',
                        help='optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db',
                        help='optuna storage database')
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    return args


def objective(trial: Trial, base_args):
    """
    Objective function for Optuna to optimize
    
    Define which hyperparameters to tune here
    """
    
    # Copy base args
    args = argparse.Namespace(**vars(base_args))
    
    # ========== HYPERPARAMETERS TO TUNE ==========

    args.use_gpu = True
    args.gpu = 0
    args.devices = '0'
    args.use_multi_gpu = False

    # Sequence lengths
    args.seq_len = trial.suggest_categorical('seq_len', [48, 96])
    args.label_len = args.seq_len // 2
    args.pred_len = trial.suggest_categorical('pred_len', [1,7,14,30])
    args.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    args.dropout = trial.suggest_float('dropout', 0.0, 0.3)

     #Model-specifi ccitly check for Mamba, as the constraint is model-specific
    if args.model == 'Mamba':
        d_inner = int(args.d_model * args.expand)

        # Prune if the Inner Dimension is 256 or higher, based on previous failures
        if d_inner >= 256: 
            print(f"Pruning Trial {trial.number}: Mamba d_inner ({d_inner}) >= 256 limit.")
            raise optuna.TrialPruned()


    if args.model == 'TimesNet':
        args.top_k = trial.suggest_int('top_k', 3, 5)
        args.num_kernels = trial.suggest_int('num_kernels', 4, 6)
    
    # Update model_id with trial number
    args.model_id = f'optuna_trial_{trial.number}'
    args.des = f'trial_{trial.number}'
    
    # ========== RUN EXPERIMENT ==========
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print(f"seq_len: {args.seq_len}, pred_len: {args.pred_len}")
    print(f"d_model: {args.d_model}, n_heads: {args.n_heads}")
    print(f"e_layers: {args.e_layers}, d_layers: {args.d_layers}")
    print(f"batch_size: {args.batch_size}, lr: {args.learning_rate:.6f}")
    print(f"dropout: {args.dropout}")
    print(f"{'='*60}\n")
    
    # Initialize experiment
    exp = Exp_Long_Term_Forecast(args)
    
    # Train model
    print(f'>>>>>> Start training: trial_{trial.number} >>>>>>')
    exp.train(setting=f'trial_{trial.number}')
    
    # Validate model
    print(f'>>>>>> Testing on validation set: trial_{trial.number} >>>>>>')
    #mse, mae = exp.test(setting=f'trial_{trial.number}', test=0)  # test=0 for validation
    
    # Clean up to save disk space (optional)
    checkpoint_path = os.path.join(
        args.checkpoints,
        f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{args.itr}'
    )
    if os.path.exists(checkpoint_path):
        # Keep only the best checkpoint, delete training checkpoints
        for file in os.listdir(checkpoint_path):
            if file.startswith('checkpoint') and file != 'checkpoint.pth':
                try:
                    os.remove(os.path.join(checkpoint_path, file))
                except:
                    pass
    
    # Return validation MSE (lower is better)
    #return mse
    try:
        #exp = Exp_Long_Term_Forecast(args)
        #exp.train(setting=f"trial_{trial.number}")
        mse, mae = exp.test(setting=f"trial_{trial.number}", test=0)
        return mse
    finally:
        # clean up CUDA memory after each trial
        del exp
        gc.collect()
        torch.cuda.empty_cache()


def run_optimization(args):
    """
    Run Optuna optimization
    """
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction='minimize',  # Minimize validation MSE
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    print(f"\n{'='*70}")
    print(f"Starting Optuna Hyperparameter Optimization")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"{'='*70}\n")
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=None,
        n_jobs=1,  # Run trials sequentially (parallel not recommended for GPU)
        show_progress_bar=True
    )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Optimization Complete!")
    print(f"{'='*70}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Value (MSE): {trial.value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}\n")
    
    # Save results
    results_dir = './optuna_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save best parameters
    import json
    with open(f'{results_dir}/{args.study_name}_best_params.json', 'w') as f:
        json.dump(trial.params, f, indent=4)
    
    print(f"✅ Best parameters saved to: {results_dir}/{args.study_name}_best_params.json")
    
    # Create visualization (if plotly available)
    try:
        import plotly
        
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f'{results_dir}/{args.study_name}_history.html')
        
        # Parameter importances
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f'{results_dir}/{args.study_name}_importances.html')
        
        # Parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f'{results_dir}/{args.study_name}_parallel.html')
        
        print(f"✅ Visualizations saved to {results_dir}/")
    except ImportError:
        print("⚠️  Install plotly for visualizations: pip install plotly")
    
    return study


if __name__ == '__main__':
    # Parse arguments
    args = get_base_args()
    
    # Run optimization
    study = run_optimization(args)
    
    print("\n" + "="*70)
    print("✅ Hyperparameter tuning complete!")
    print(f"✅ Best parameters saved to: ./optuna_results/{args.study_name}_best_params.json")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review visualizations in optuna_results/ folder")
    print("  2. Train final model:")
    print(f"     python train_best_model.py --model {args.model} --train_epochs 100")
    print("="*70)
