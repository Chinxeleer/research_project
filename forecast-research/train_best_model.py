"""
Train final model with best parameters from Optuna tuning
Save as: train_best_model.py

Usage:
    python train_best_model.py --model TimesNet --data financial
"""

import argparse
import json
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def load_best_params(study_name='financial_forecasting'):
    """Load best parameters from Optuna study"""
    params_file = f'./optuna_results/{study_name}_best_params.json'
    
    try:
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        print(f"✅ Loaded best parameters from {params_file}")
        return best_params
    except FileNotFoundError:
        print(f"❌ Parameters file not found: {params_file}")
        print("Run optuna_tune.py first!")
        return None


def get_args():
    """Get arguments and merge with best parameters"""
    parser = argparse.ArgumentParser(description='Train with best Optuna parameters')
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='final_model')
    parser.add_argument('--model', type=str, default='TimesNet')
    
    # Data loader
    parser.add_argument('--data', type=str, default='financial')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='your_financial_data.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='close')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Training params (will be overridden by best params)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
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
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=50, 
                        help='Increase for final training')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10,
                        help='Increase for final training')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='final_best_model')
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
    
    # Study name
    parser.add_argument('--study_name', type=str, default='financial_forecasting')
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    return args


def main():
    # Get base arguments
    args = get_args()
    
    # Load best parameters from Optuna
    best_params = load_best_params(args.study_name)
    
    if best_params is None:
        print("Using default parameters (no Optuna results found)")
    else:
        # Override arguments with best parameters
        print("\n" + "="*70)
        print("Applying Best Parameters from Optuna:")
        print("="*70)
        
        for key, value in best_params.items():
            if hasattr(args, key):
                old_value = getattr(args, key)
                setattr(args, key, value)
                print(f"  {key}: {old_value} → {value}")
        
        # Update dependent parameters
        args.label_len = args.seq_len // 2
        args.d_ff = args.d_model * 4
        
        print("="*70 + "\n")
    
    # Initialize experiment
    print(f"{'='*70}")
    print(f"Final Model Training with Best Parameters")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Prediction length: {args.pred_len}")
    print(f"Training epochs: {args.train_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*70}\n")
    
    # Train final model
    setting = f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{args.itr}'
    
    exp = Exp_Long_Term_Forecast(args)
    
    # Train
    print('>>>>>> Start training final model >>>>>>')
    exp.train(setting)
    
    # Test on validation set
    print('>>>>>> Testing on validation set >>>>>>')
    val_mse, val_mae = exp.test(setting, test=0)
    
    # Test on test set
    print('>>>>>> Testing on test set >>>>>>')
    test_mse, test_mae = exp.test(setting, test=1)
    
    # Print final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Validation MSE: {val_mse:.6f}")
    print(f"Validation MAE: {val_mae:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print("="*70)
    print(f"\n✅ Model saved to: ./checkpoints/{setting}/")
    
    # Save final metrics
    import json
    results = {
        'model': args.model,
        'validation': {'mse': float(val_mse), 'mae': float(val_mae)},
        'test': {'mse': float(test_mse), 'mae': float(test_mae)},
        'best_params': best_params if best_params else 'default'
    }
    
    with open(f'./optuna_results/final_model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: ./optuna_results/final_model_results.json")


if __name__ == '__main__':
    main()
