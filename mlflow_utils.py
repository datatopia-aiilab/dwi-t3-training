"""
MLflow Utilities for DWI Segmentation Experiment Tracking
Provides wrapper functions for logging experiments, metrics, and artifacts
"""

import mlflow
import mlflow.pytorch
from datetime import datetime
from pathlib import Path
import torch
import json


def setup_mlflow(cfg, model_params=None):
    """
    Initialize MLflow tracking
    
    Args:
        cfg: Configuration module
        model_params: Dictionary with model parameter counts (optional)
    
    Returns:
        run: MLflow active run object (or None if disabled)
    """
    if not cfg.MLFLOW_ENABLED:
        print("ℹ️  MLflow tracking is disabled")
        return None
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(cfg.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(cfg.MLFLOW_EXPERIMENT_NAME)
            print(f"✅ Created MLflow experiment: {cfg.MLFLOW_EXPERIMENT_NAME}")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing MLflow experiment: {cfg.MLFLOW_EXPERIMENT_NAME}")
        
        mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)
        
        # Generate run name if not specified
        run_name = cfg.MLFLOW_RUN_NAME
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            arch = cfg.MODEL_ARCHITECTURE
            
            # Add encoder info for SMP models
            if arch != 'attention_unet':
                encoder = cfg.ENCODER_NAME
                pretrained = 'img' if cfg.ENCODER_WEIGHTS == 'imagenet' else 'rand'
                run_name = f"{arch}_{encoder}_{pretrained}_{timestamp}"
            else:
                run_name = f"{arch}_{timestamp}"
        
        # Start run
        run = mlflow.start_run(run_name=run_name)
        
        print(f"\n{'='*70}")
        print(f"🔬 MLFLOW TRACKING INITIALIZED")
        print(f"{'='*70}")
        print(f"   Experiment: {cfg.MLFLOW_EXPERIMENT_NAME}")
        print(f"   Run Name: {run_name}")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Tracking URI: {cfg.MLFLOW_TRACKING_URI}")
        print(f"{'='*70}\n")
        
        # Log basic tags
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag("architecture", cfg.MODEL_ARCHITECTURE)
        mlflow.set_tag("augmentation", "enabled" if cfg.AUGMENTATION_ENABLED else "disabled")
        mlflow.set_tag("loss_type", cfg.LOSS_TYPE)
        
        # Add encoder tags for SMP models
        if cfg.MODEL_ARCHITECTURE != 'attention_unet':
            mlflow.set_tag("encoder", cfg.ENCODER_NAME)
            mlflow.set_tag("pretrained", "yes" if cfg.ENCODER_WEIGHTS else "no")
            if cfg.ENCODER_WEIGHTS:
                mlflow.set_tag("pretrained_weights", cfg.ENCODER_WEIGHTS)
        
        # Log model parameters if provided
        if model_params:
            mlflow.log_param("total_parameters", model_params.get('total', 0))
            mlflow.log_param("trainable_parameters", model_params.get('trainable', 0))
            mlflow.log_param("model_size_mb", round(model_params.get('total', 0) * 4 / (1024**2), 2))
        
        return run
        
    except Exception as e:
        print(f"⚠️  Failed to initialize MLflow: {e}")
        print(f"   Continuing without MLflow tracking...")
        return None


def log_config_params(cfg):
    """
    Log all configuration parameters to MLflow
    
    Args:
        cfg: Configuration module
    """
    if not cfg.MLFLOW_ENABLED or not mlflow.active_run():
        return
    
    try:
        # Model architecture parameters
        mlflow.log_param("model_architecture", cfg.MODEL_ARCHITECTURE)
        mlflow.log_param("in_channels", cfg.IN_CHANNELS)
        mlflow.log_param("out_channels", cfg.OUT_CHANNELS)
        
        # Architecture-specific parameters
        if cfg.MODEL_ARCHITECTURE == 'attention_unet':
            mlflow.log_param("encoder_channels", str(cfg.ENCODER_CHANNELS))
            mlflow.log_param("decoder_channels", str(cfg.DECODER_CHANNELS))
            mlflow.log_param("bottleneck_channels", cfg.BOTTLENECK_CHANNELS)
            mlflow.log_param("use_attention", cfg.USE_ATTENTION)
        else:
            mlflow.log_param("encoder_name", cfg.ENCODER_NAME)
            mlflow.log_param("encoder_weights", cfg.ENCODER_WEIGHTS or "None")
        
        # Training parameters
        mlflow.log_param("num_epochs", cfg.NUM_EPOCHS)
        mlflow.log_param("batch_size", cfg.BATCH_SIZE)
        mlflow.log_param("learning_rate", cfg.LEARNING_RATE)
        mlflow.log_param("optimizer", cfg.OPTIMIZER)
        mlflow.log_param("weight_decay", cfg.WEIGHT_DECAY)
        mlflow.log_param("gradient_clip_value", cfg.GRADIENT_CLIP_VALUE)
        
        # Loss function
        mlflow.log_param("loss_type", cfg.LOSS_TYPE)
        if cfg.LOSS_TYPE == 'focal':
            mlflow.log_param("focal_alpha", cfg.FOCAL_ALPHA)
            mlflow.log_param("focal_gamma", cfg.FOCAL_GAMMA)
        elif cfg.LOSS_TYPE == 'combo':
            mlflow.log_param("combo_focal_weight", cfg.COMBO_FOCAL_WEIGHT)
            mlflow.log_param("combo_dice_weight", cfg.COMBO_DICE_WEIGHT)
            mlflow.log_param("focal_alpha", cfg.FOCAL_ALPHA)
            mlflow.log_param("focal_gamma", cfg.FOCAL_GAMMA)
        mlflow.log_param("dice_smooth", cfg.DICE_SMOOTH)
        
        # Scheduler
        mlflow.log_param("scheduler", cfg.SCHEDULER)
        mlflow.log_param("scheduler_patience", cfg.SCHEDULER_PATIENCE)
        mlflow.log_param("scheduler_factor", cfg.SCHEDULER_FACTOR)
        mlflow.log_param("scheduler_min_lr", cfg.SCHEDULER_MIN_LR)
        
        # Early stopping
        mlflow.log_param("early_stopping_patience", cfg.EARLY_STOPPING_PATIENCE)
        mlflow.log_param("early_stopping_min_delta", cfg.EARLY_STOPPING_MIN_DELTA)
        
        # Data parameters
        mlflow.log_param("image_size", str(cfg.IMAGE_SIZE))
        mlflow.log_param("train_ratio", cfg.TRAIN_RATIO)
        mlflow.log_param("val_ratio", cfg.VAL_RATIO)
        mlflow.log_param("test_ratio", cfg.TEST_RATIO)
        mlflow.log_param("normalize_method", cfg.NORMALIZE_METHOD)
        mlflow.log_param("clahe_enabled", cfg.CLAHE_ENABLED)
        
        # Augmentation parameters
        mlflow.log_param("augmentation_enabled", cfg.AUGMENTATION_ENABLED)
        if cfg.AUGMENTATION_ENABLED:
            mlflow.log_param("aug_horizontal_flip", cfg.AUG_HORIZONTAL_FLIP_PROB)
            mlflow.log_param("aug_vertical_flip", cfg.AUG_VERTICAL_FLIP_PROB)
            mlflow.log_param("aug_rotate_prob", cfg.AUG_ROTATE_PROB)
            mlflow.log_param("aug_rotate_limit", cfg.AUG_ROTATE_LIMIT)
            mlflow.log_param("aug_elastic_prob", cfg.AUG_ELASTIC_TRANSFORM_PROB)
            mlflow.log_param("aug_brightness_contrast", cfg.AUG_BRIGHTNESS_CONTRAST_PROB)
            mlflow.log_param("aug_gaussian_noise", cfg.AUG_GAUSSIAN_NOISE_PROB)
        
        # Hardware
        mlflow.log_param("device", str(cfg.DEVICE))
        mlflow.log_param("num_workers", cfg.NUM_WORKERS)
        mlflow.log_param("use_mixed_precision", cfg.USE_MIXED_PRECISION)
        if cfg.USE_CUDA:
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param("num_gpus", cfg.NUM_GPUS)
        
        print("✅ Logged configuration parameters to MLflow")
        
    except Exception as e:
        print(f"⚠️  Failed to log config parameters: {e}")


def log_epoch_metrics(epoch, train_metrics, val_metrics, learning_rate, epoch_time):
    """
    Log metrics for one epoch to MLflow
    
    Args:
        epoch: Current epoch number
        train_metrics: Dictionary with training metrics {'loss': float, 'dice': float}
        val_metrics: Dictionary with validation metrics {'loss': float, 'dice': float}
        learning_rate: Current learning rate
        epoch_time: Time taken for this epoch (seconds)
    """
    if not mlflow.active_run():
        return
    
    try:
        # Log training metrics
        mlflow.log_metric("train_loss", train_metrics['loss'], step=epoch)
        mlflow.log_metric("train_dice", train_metrics['dice'], step=epoch)
        
        # Log validation metrics
        mlflow.log_metric("val_loss", val_metrics['loss'], step=epoch)
        mlflow.log_metric("val_dice", val_metrics['dice'], step=epoch)
        
        # Log learning rate
        mlflow.log_metric("learning_rate", learning_rate, step=epoch)
        
        # Log epoch time
        mlflow.log_metric("epoch_time_seconds", epoch_time, step=epoch)
        
        # Log overfitting gap (Val Dice - Train Dice)
        overfitting_gap = train_metrics['dice'] - val_metrics['dice']
        mlflow.log_metric("overfitting_gap", overfitting_gap, step=epoch)
        
    except Exception as e:
        print(f"⚠️  Failed to log epoch metrics: {e}")


def log_best_metrics(best_val_dice, best_epoch, total_epochs, final_train_metrics, final_val_metrics, training_time):
    """
    Log best metrics and summary statistics at the end of training
    
    Args:
        best_val_dice: Best validation Dice score achieved
        best_epoch: Epoch where best validation Dice was achieved
        total_epochs: Total number of epochs trained
        final_train_metrics: Final training metrics
        final_val_metrics: Final validation metrics
        training_time: Total training time in seconds
    """
    if not mlflow.active_run():
        return
    
    try:
        # Log best metrics (these will show in the main experiments table)
        mlflow.log_metric("best_val_dice", best_val_dice)
        mlflow.log_metric("best_epoch", best_epoch)
        
        # Log final metrics
        mlflow.log_metric("final_train_loss", final_train_metrics['loss'])
        mlflow.log_metric("final_train_dice", final_train_metrics['dice'])
        mlflow.log_metric("final_val_loss", final_val_metrics['loss'])
        mlflow.log_metric("final_val_dice", final_val_metrics['dice'])
        
        # Log training statistics
        mlflow.log_metric("total_epochs_trained", total_epochs)
        mlflow.log_metric("training_time_minutes", training_time / 60)
        mlflow.log_metric("training_time_hours", training_time / 3600)
        mlflow.log_metric("time_per_epoch_seconds", training_time / total_epochs)
        
        # Log final overfitting gap
        final_gap = final_train_metrics['dice'] - final_val_metrics['dice']
        mlflow.log_metric("final_overfitting_gap", final_gap)
        
        print(f"✅ Logged best metrics: Val Dice {best_val_dice:.4f} at epoch {best_epoch}")
        
    except Exception as e:
        print(f"⚠️  Failed to log best metrics: {e}")


def log_test_metrics(test_metrics):
    """
    Log test set evaluation metrics
    
    Args:
        test_metrics: Dictionary with test metrics (dice, iou, precision, recall, etc.)
    """
    if not mlflow.active_run():
        return
    
    try:
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        print(f"✅ Logged test metrics: {list(test_metrics.keys())}")
        
    except Exception as e:
        print(f"⚠️  Failed to log test metrics: {e}")


def log_model_artifact(model_path, artifact_name="best_model"):
    """
    Log model checkpoint as MLflow artifact
    
    Args:
        model_path: Path to model checkpoint file
        artifact_name: Name for the artifact (default: "best_model")
    """
    if not mlflow.active_run():
        return
    
    try:
        model_path = Path(model_path)
        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="models")
            print(f"✅ Logged model artifact: {model_path.name}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
    
    except Exception as e:
        print(f"⚠️  Failed to log model artifact: {e}")


def log_training_history(history_path):
    """
    Log training history JSON file as artifact
    
    Args:
        history_path: Path to training history JSON file
    """
    if not mlflow.active_run():
        return
    
    try:
        history_path = Path(history_path)
        if history_path.exists():
            mlflow.log_artifact(str(history_path), artifact_path="history")
            print(f"✅ Logged training history: {history_path.name}")
        else:
            print(f"⚠️  History file not found: {history_path}")
    
    except Exception as e:
        print(f"⚠️  Failed to log training history: {e}")


def log_config_file(config_path=None):
    """
    Log config.py file as artifact for reproducibility
    
    Args:
        config_path: Path to config file (default: auto-detect)
    """
    if not mlflow.active_run():
        return
    
    try:
        if config_path is None:
            config_path = Path(__file__).parent / "config.py"
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="config")
            print(f"✅ Logged config file: {config_path.name}")
        else:
            print(f"⚠️  Config file not found: {config_path}")
    
    except Exception as e:
        print(f"⚠️  Failed to log config file: {e}")


def log_plot(plot_path, artifact_path="plots"):
    """
    Log plot/visualization as artifact
    
    Args:
        plot_path: Path to plot image file
        artifact_path: Subdirectory in artifacts (default: "plots")
    """
    if not mlflow.active_run():
        return
    
    try:
        plot_path = Path(plot_path)
        if plot_path.exists():
            mlflow.log_artifact(str(plot_path), artifact_path=artifact_path)
            print(f"✅ Logged plot: {plot_path.name}")
        else:
            print(f"⚠️  Plot file not found: {plot_path}")
    
    except Exception as e:
        print(f"⚠️  Failed to log plot: {e}")


def end_run(status="FINISHED"):
    """
    End MLflow run gracefully
    
    Args:
        status: Run status - "FINISHED", "FAILED", or "KILLED"
    """
    if not mlflow.active_run():
        return
    
    try:
        mlflow.end_run(status=status)
        print(f"\n{'='*70}")
        print(f"✅ MLflow run ended with status: {status}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"⚠️  Failed to end MLflow run: {e}")


def get_run_url():
    """
    Get URL to view current run in MLflow UI
    
    Returns:
        str: URL to run (or None if no active run)
    """
    active_run = mlflow.active_run()
    if not active_run:
        return None
    
    run_id = active_run.info.run_id
    experiment_id = active_run.info.experiment_id
    
    # Assuming MLflow UI is running on default port
    return f"http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}"


def print_run_info():
    """Print information about current MLflow run"""
    if not mlflow.active_run():
        print("ℹ️  No active MLflow run")
        return
    
    run = mlflow.active_run()
    print(f"\n{'='*70}")
    print(f"📊 CURRENT MLFLOW RUN")
    print(f"{'='*70}")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"   Experiment ID: {run.info.experiment_id}")
    print(f"   Status: {run.info.status}")
    
    url = get_run_url()
    if url:
        print(f"   View in UI: {url}")
    
    print(f"{'='*70}\n")


# Convenience function to log everything at the end of training
def log_training_complete(cfg, best_val_dice, best_epoch, total_epochs, 
                         final_train_metrics, final_val_metrics, training_time,
                         best_model_path, history_path):
    """
    Log all final artifacts and metrics when training completes
    
    Args:
        cfg: Configuration module
        best_val_dice: Best validation Dice
        best_epoch: Epoch with best validation Dice
        total_epochs: Total epochs trained
        final_train_metrics: Final training metrics
        final_val_metrics: Final validation metrics
        training_time: Total training time (seconds)
        best_model_path: Path to best model checkpoint
        history_path: Path to training history JSON
    """
    if not cfg.MLFLOW_ENABLED or not mlflow.active_run():
        return
    
    print(f"\n{'='*70}")
    print(f"📦 LOGGING TRAINING ARTIFACTS TO MLFLOW")
    print(f"{'='*70}\n")
    
    # Log best metrics
    log_best_metrics(best_val_dice, best_epoch, total_epochs, 
                    final_train_metrics, final_val_metrics, training_time)
    
    # Log artifacts
    log_model_artifact(best_model_path, "best_model")
    log_training_history(history_path)
    log_config_file()
    
    # Log any plots if they exist
    plots_dir = cfg.PLOTS_DIR
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*.png"):
            log_plot(plot_file)
    
    print(f"\n{'='*70}")
    print(f"✅ ALL ARTIFACTS LOGGED SUCCESSFULLY")
    print_run_info()
    print(f"{'='*70}\n")
