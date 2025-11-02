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
