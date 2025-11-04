"""
Weights & Biases Integration Wrapper
Save as: utils/wandb_logger.py

This provides a clean interface to log experiments to wandb
"""

import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class WandbLogger:
    """
    Weights & Biases logger for Time-Series forecasting experiments
    """
    
    def __init__(self, args, project_name="financial-forecasting", entity=None, enabled=True):
        """
        Initialize wandb logger
        
        Args:
            args: Experiment arguments
            project_name: W&B project name
            entity: W&B team/username (None for personal)
            enabled: Whether to enable wandb logging
        """
        self.enabled = enabled
        
        if not self.enabled:
            print("W&B logging disabled")
            return
        
        # Initialize wandb
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            name=self._get_run_name(args),
            config=vars(args),
            reinit=True,
            tags=self._get_tags(args)
        )
        
        print(f"✅ W&B initialized: {self.run.name}")
        print(f"   Dashboard: {self.run.url}")
    
    def _get_run_name(self, args):
        """Generate descriptive run name"""
        return f"{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}"
    
    def _get_tags(self, args):
        """Generate tags for the run"""
        tags = [
            args.model,
            args.data,
            args.features,
            f"seq_{args.seq_len}",
            f"pred_{args.pred_len}"
        ]
        return tags
    
    def log_metrics(self, metrics, step=None, prefix=""):
        """
        Log metrics to wandb
        
        Args:
            metrics: Dictionary of metrics to log
            step: Training step/epoch
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        if not self.enabled:
            return
        
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        wandb.log(metrics, step=step)
    
    def log_epoch_metrics(self, epoch, train_loss, vali_loss=None, test_loss=None, 
                         train_metrics=None, vali_metrics=None, test_metrics=None):
        """
        Log metrics for all splits at end of epoch
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            vali_loss: Validation loss
            test_loss: Test loss
            train_metrics: Dict of training metrics (mse, mae, etc.)
            vali_metrics: Dict of validation metrics
            test_metrics: Dict of test metrics
        """
        if not self.enabled:
            return
        
        log_dict = {"epoch": epoch}
        
        # Training metrics
        log_dict["train/loss"] = train_loss
        if train_metrics:
            for key, val in train_metrics.items():
                log_dict[f"train/{key}"] = val
        
        # Validation metrics
        if vali_loss is not None:
            log_dict["val/loss"] = vali_loss
        if vali_metrics:
            for key, val in vali_metrics.items():
                log_dict[f"val/{key}"] = val
        
        # Test metrics
        if test_loss is not None:
            log_dict["test/loss"] = test_loss
        if test_metrics:
            for key, val in test_metrics.items():
                log_dict[f"test/{key}"] = val
        
        wandb.log(log_dict, step=epoch)
    
    def log_predictions(self, true, pred, split="test", num_samples=5):
        """
        Log prediction visualizations
        
        Args:
            true: Ground truth values (numpy array)
            pred: Predicted values (numpy array)
            split: Dataset split name
            num_samples: Number of samples to plot
        """
        if not self.enabled:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(min(num_samples, len(true))):
            axes[i].plot(true[i], label='Ground Truth', alpha=0.7)
            axes[i].plot(pred[i], label='Prediction', alpha=0.7)
            axes[i].set_title(f'{split.capitalize()} Sample {i+1}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({f"{split}/predictions": wandb.Image(fig)})
        plt.close(fig)
    
    def log_confusion_metrics(self, true, pred, split="test"):
        """
        Log regression metrics and scatter plot
        
        Args:
            true: Ground truth values (numpy array)
            pred: Predicted values (numpy array)
            split: Dataset split name
        """
        if not self.enabled:
            return
        
        # Flatten arrays
        true_flat = true.flatten()
        pred_flat = pred.flatten()
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(true_flat, pred_flat, alpha=0.3, s=1)
        
        # Add diagonal line
        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{split.capitalize()} Set: Predictions vs True')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log to wandb
        wandb.log({f"{split}/scatter": wandb.Image(fig)})
        plt.close(fig)
    
    def log_distribution_comparison(self, true, pred, split="test"):
        """
        Log distribution comparison between true and predicted values

        Args:
            true: Ground truth values (numpy array)
            pred: Predicted values (numpy array)
            split: Dataset split name
        """
        if not self.enabled:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(true.flatten(), bins=50, alpha=0.5, label='True', density=True)
        axes[0].hist(pred.flatten(), bins=50, alpha=0.5, label='Predicted', density=True)
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Distribution Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot([true.flatten(), pred.flatten()], labels=['True', 'Predicted'])
        axes[1].set_ylabel('Value')
        axes[1].set_title('Box Plot Comparison')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        wandb.log({f"{split}/distributions": wandb.Image(fig)})
        plt.close(fig)

    def log_residual_analysis(self, true, pred, split="test"):
        """
        Log residual analysis plots

        Args:
            true: Ground truth values (numpy array)
            pred: Predicted values (numpy array)
            split: Dataset split name
        """
        if not self.enabled:
            return

        residuals = pred.flatten() - true.flatten()

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Residual vs Predicted
        axes[0, 0].scatter(pred.flatten(), residuals, alpha=0.3, s=1)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals (Pred - True)')
        axes[0, 0].set_title('Residual Plot')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residual Histogram
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Residual Distribution (Mean: {residuals.mean():.6f})')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Residuals over time (first 200 points)
        n_points = min(200, len(residuals))
        axes[1, 1].plot(residuals[:n_points], alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Over Time (First 200 Steps)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        wandb.log({f"{split}/residual_analysis": wandb.Image(fig)})
        plt.close(fig)

    def log_error_metrics_detailed(self, true, pred, split="test"):
        """
        Log detailed error metrics including directional accuracy

        Args:
            true: Ground truth values (numpy array) [batch, horizon, features]
            pred: Predicted values (numpy array) [batch, horizon, features]
            split: Dataset split name
        """
        if not self.enabled:
            return

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import numpy as np

        # Flatten for overall metrics
        true_flat = true.flatten()
        pred_flat = pred.flatten()

        # Calculate metrics
        mae = mean_absolute_error(true_flat, pred_flat)
        mse = mean_squared_error(true_flat, pred_flat)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_flat, pred_flat)

        # MAPE (avoiding division by zero)
        mask = true_flat != 0
        mape = np.mean(np.abs((true_flat[mask] - pred_flat[mask]) / true_flat[mask])) * 100 if mask.sum() > 0 else 0

        # Directional accuracy (percentage of correct direction predictions)
        if true.shape[1] > 1:  # If we have multiple timesteps
            # Calculate changes from t to t+1
            true_changes = np.diff(true, axis=1)  # [batch, horizon-1, features]
            pred_changes = np.diff(pred, axis=1)

            # Sign agreement
            direction_correct = (np.sign(true_changes) == np.sign(pred_changes))
            directional_accuracy = direction_correct.mean() * 100
        else:
            directional_accuracy = 0.0

        # Log metrics
        metrics = {
            f"{split}/mae": mae,
            f"{split}/mse": mse,
            f"{split}/rmse": rmse,
            f"{split}/r2": r2,
            f"{split}/mape": mape,
            f"{split}/directional_accuracy": directional_accuracy
        }

        wandb.log(metrics)

        # Create metrics summary table
        metrics_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["MAE", f"{mae:.6f}"],
                ["MSE", f"{mse:.6f}"],
                ["RMSE", f"{rmse:.6f}"],
                ["R²", f"{r2:.6f}"],
                ["MAPE (%)", f"{mape:.2f}"],
                ["Directional Accuracy (%)", f"{directional_accuracy:.2f}"]
            ]
        )
        wandb.log({f"{split}/metrics_summary": metrics_table})

    def log_horizon_analysis(self, true, pred, split="test"):
        """
        Log per-horizon error analysis

        Args:
            true: Ground truth values (numpy array) [batch, horizon, features]
            pred: Predicted values (numpy array) [batch, horizon, features]
            split: Dataset split name
        """
        if not self.enabled:
            return

        if len(true.shape) < 2:
            return  # Need at least 2D array

        horizon_length = true.shape[1]

        # Calculate MSE and MAE per horizon step
        mse_per_horizon = []
        mae_per_horizon = []

        for h in range(horizon_length):
            true_h = true[:, h, :].flatten()
            pred_h = pred[:, h, :].flatten()

            mse_h = np.mean((pred_h - true_h) ** 2)
            mae_h = np.mean(np.abs(pred_h - true_h))

            mse_per_horizon.append(mse_h)
            mae_per_horizon.append(mae_h)

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        horizons = list(range(1, horizon_length + 1))

        # MSE per horizon
        axes[0].plot(horizons, mse_per_horizon, marker='o', linewidth=2)
        axes[0].set_xlabel('Forecast Horizon')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('MSE by Forecast Horizon')
        axes[0].grid(True, alpha=0.3)

        # MAE per horizon
        axes[1].plot(horizons, mae_per_horizon, marker='o', linewidth=2, color='orange')
        axes[1].set_xlabel('Forecast Horizon')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('MAE by Forecast Horizon')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        wandb.log({f"{split}/horizon_analysis": wandb.Image(fig)})
        plt.close(fig)

        # Log numerical values
        horizon_table = wandb.Table(
            columns=["Horizon", "MSE", "MAE"],
            data=[[h+1, mse_per_horizon[h], mae_per_horizon[h]] for h in range(horizon_length)]
        )
        wandb.log({f"{split}/horizon_metrics": horizon_table})

    def log_comprehensive_test_results(self, true, pred, split="test", num_samples=5):
        """
        Log comprehensive test results with all visualizations

        Args:
            true: Ground truth values (numpy array)
            pred: Predicted values (numpy array)
            split: Dataset split name
            num_samples: Number of prediction samples to visualize
        """
        if not self.enabled:
            return

        print(f"\n📊 Logging comprehensive {split} results to W&B...")

        # 1. Detailed metrics
        self.log_error_metrics_detailed(true, pred, split)

        # 2. Prediction samples
        self.log_predictions(true, pred, split, num_samples)

        # 3. Scatter plot (pred vs true)
        self.log_confusion_metrics(true, pred, split)

        # 4. Distribution comparison
        self.log_distribution_comparison(true, pred, split)

        # 5. Residual analysis
        self.log_residual_analysis(true, pred, split)

        # 6. Horizon analysis
        if len(true.shape) >= 2 and true.shape[1] > 1:
            self.log_horizon_analysis(true, pred, split)

        print(f"✅ {split.capitalize()} results logged to W&B")
    
    def log_learning_curve(self, train_losses, val_losses):
        """
        Log learning curves
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
        """
        if not self.enabled:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label='Training Loss', alpha=0.8)
        ax.plot(val_losses, label='Validation Loss', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        wandb.log({"learning_curves": wandb.Image(fig)})
        plt.close(fig)
    
    def log_model_architecture(self, model):
        """
        Log model architecture summary
        
        Args:
            model: PyTorch model
        """
        if not self.enabled:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "model/non_trainable_params": total_params - trainable_params
        })
        
        # Log model architecture as text
        wandb.log({"model/architecture": str(model)})
    
    def log_hyperparameters(self, hparams, metrics):
        """
        Log hyperparameters with final metrics for comparison
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        if not self.enabled:
            return
        
        wandb.log({
            "hparams": hparams,
            "final_metrics": metrics
        })
    
    def watch_model(self, model, log_freq=100):
        """
        Watch model gradients and parameters
        
        Args:
            model: PyTorch model
            log_freq: Logging frequency
        """
        if not self.enabled:
            return
        
        wandb.watch(model, log="all", log_freq=log_freq)
    
    def finish(self):
        """Finish the wandb run"""
        if not self.enabled:
            return
        
        wandb.finish()
        print("✅ W&B run finished")


def init_wandb(args, project_name="financial-forecasting", enabled=True):
    """
    Convenience function to initialize wandb logger
    
    Args:
        args: Experiment arguments
        project_name: W&B project name
        enabled: Whether to enable wandb
    
    Returns:
        WandbLogger instance
    """
    return WandbLogger(args, project_name=project_name, enabled=enabled)
