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
import numpy as np


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
                         best_model_path, history_path, 
                         curves_combined_path=None, curves_separated_path=None):
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
        curves_combined_path: Path to combined training curves plot (optional)
        curves_separated_path: Path to separated training curves plot (optional)
    """
    if not cfg.MLFLOW_ENABLED or not mlflow.active_run():
        return
    
    print(f"\n{'='*70}")
    print(f"📦 LOGGING TRAINING ARTIFACTS TO MLFLOW")
    print(f"{'='*70}\n")
    
    # Log best metrics
    print("   📊 Logging best metrics...")
    log_best_metrics(best_val_dice, best_epoch, total_epochs, 
                    final_train_metrics, final_val_metrics, training_time)
    print(f"      ✅ Best val dice: {best_val_dice:.4f} at epoch {best_epoch}")
    
    # Log artifacts
    print(f"   💾 Logging model checkpoint...")
    log_model_artifact(best_model_path, "best_model")
    
    print(f"   📈 Logging training history...")
    log_training_history(history_path)
    
    print(f"   ⚙️  Logging config file...")
    log_config_file()
    
    # ⭐ Log training curves (both versions) if provided
    print(f"   📊 Logging training curves...")
    logged_curves = []
    
    if curves_combined_path and Path(curves_combined_path).exists():
        mlflow.log_artifact(str(curves_combined_path), artifact_path="plots")
        print(f"      ✅ Combined: {Path(curves_combined_path).name}")
        logged_curves.append(Path(curves_combined_path))
    
    if curves_separated_path and Path(curves_separated_path).exists():
        mlflow.log_artifact(str(curves_separated_path), artifact_path="plots")
        print(f"      ✅ Separated: {Path(curves_separated_path).name}")
        logged_curves.append(Path(curves_separated_path))
    
    if not logged_curves:
        print(f"      ⚠️  No training curves found")
    
    # Log any other plots if they exist
    plots_dir = cfg.PLOTS_DIR
    if plots_dir.exists():
        other_plots = 0
        for plot_file in plots_dir.glob("*.png"):
            # Skip if already logged
            if any(plot_file.samefile(logged) for logged in logged_curves):
                continue
            log_plot(plot_file)
            other_plots += 1
        if other_plots > 0:
            print(f"   📊 Logged {other_plots} additional plot(s)")
    
    print(f"\n{'='*70}")
    print(f"✅ ALL ARTIFACTS LOGGED SUCCESSFULLY")
    
    # Show where to view results
    run = mlflow.active_run()
    if run:
        print(f"\n📊 MLflow Run Information:")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
        
        url = get_run_url()
        if url:
            print(f"\n   🌐 View results in MLflow UI:")
            print(f"   {url}")
            print(f"\n   💡 To open MLflow UI, run:")
            print(f"   mlflow ui --port 5000")
            print(f"   Then open: http://localhost:5000")
    
    print(f"{'='*70}\n")


# ==================== Test Evaluation Logging Functions ====================

def log_test_evaluation(test_metrics_aggregated, per_sample_metrics=None):
    """
    Log test set evaluation metrics to MLflow
    
    Args:
        test_metrics_aggregated: Dict with aggregated metrics
            Format: {'dice': {'mean': 0.65, 'std': 0.12, 'min': 0.42, 'max': 0.85, ...}, ...}
        per_sample_metrics: List of per-sample metric dicts (optional)
    """
    if not mlflow.active_run():
        return
    
    try:
        print("\n📊 Logging test evaluation metrics to MLflow...")
        
        # Log aggregated metrics (these show in main experiments table)
        for metric_name, values in test_metrics_aggregated.items():
            # Main metric (mean)
            mlflow.log_metric(f"test_{metric_name}", values['mean'])
            
            # Additional statistics
            mlflow.log_metric(f"test_{metric_name}_std", values['std'])
            mlflow.log_metric(f"test_{metric_name}_min", values['min'])
            mlflow.log_metric(f"test_{metric_name}_max", values['max'])
            mlflow.log_metric(f"test_{metric_name}_median", values['median'])
        
        # Log summary statistics
        if per_sample_metrics:
            mlflow.log_metric("test_num_samples", len(per_sample_metrics))
            
            # Find best and worst samples
            dice_scores = [m['dice'] for m in per_sample_metrics]
            best_dice = max(dice_scores)
            worst_dice = min(dice_scores)
            dice_range = best_dice - worst_dice
            
            mlflow.log_metric("test_dice_range", dice_range)
            mlflow.log_metric("test_dice_best", best_dice)
            mlflow.log_metric("test_dice_worst", worst_dice)
            
            # Log volume metrics if available
            if 'gt_volume_ml' in per_sample_metrics[0]:
                gt_volumes = [m['gt_volume_ml'] for m in per_sample_metrics]
                pred_volumes = [m['pred_volume_ml'] for m in per_sample_metrics]
                volume_errors = [m['volume_error_percent'] for m in per_sample_metrics]
                
                mlflow.log_metric("test_mean_gt_volume_ml", np.mean(gt_volumes))
                mlflow.log_metric("test_mean_pred_volume_ml", np.mean(pred_volumes))
                mlflow.log_metric("test_mean_volume_error_percent", np.mean(volume_errors))
                mlflow.log_metric("test_median_gt_volume_ml", np.median(gt_volumes))
                mlflow.log_metric("test_median_pred_volume_ml", np.median(pred_volumes))
                
                print("   ✅ Logged volume metrics")
        
        print("   ✅ Logged aggregated test metrics")
        
    except Exception as e:
        print(f"   ⚠️  Failed to log test metrics: {e}")


def log_per_sample_results_csv(csv_path, artifact_name="per_sample_results"):
    """
    Log per-sample results CSV as MLflow artifact
    
    Args:
        csv_path: Path to CSV file
        artifact_name: Name for artifact folder
    """
    if not mlflow.active_run():
        return
    
    try:
        csv_path = Path(csv_path)
        if csv_path.exists():
            mlflow.log_artifact(str(csv_path), artifact_path="evaluation")
            print(f"   ✅ Logged per-sample CSV: {csv_path.name}")
        else:
            print(f"   ⚠️  CSV file not found: {csv_path}")
    
    except Exception as e:
        print(f"   ⚠️  Failed to log per-sample CSV: {e}")


def log_qualitative_images(images_dir, max_images=None):
    """
    Log qualitative prediction images to MLflow
    
    Args:
        images_dir: Directory containing prediction images
        max_images: Maximum number of images to log (None = all)
    """
    if not mlflow.active_run():
        return
    
    try:
        images_dir = Path(images_dir)
        if not images_dir.exists():
            print(f"   ⚠️  Images directory not found: {images_dir}")
            return
        
        # Find all PNG images
        image_files = sorted(images_dir.glob("*.png"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            print(f"   ⚠️  No PNG images found in: {images_dir}")
            return
        
        # Log each image
        for img_file in image_files:
            mlflow.log_artifact(str(img_file), artifact_path="predictions")
        
        print(f"   ✅ Logged {len(image_files)} prediction images")
        
    except Exception as e:
        print(f"   ⚠️  Failed to log prediction images: {e}")


def log_test_plots(plots_dir):
    """
    Log test evaluation plots (distribution, etc.) to MLflow
    
    Args:
        plots_dir: Directory containing plots
    """
    if not mlflow.active_run():
        return
    
    try:
        plots_dir = Path(plots_dir)
        if not plots_dir.exists():
            return
        
        # Log test-specific plots
        test_plots = [
            'test_metrics_distribution.png',
            'metrics_distribution.png',
        ]
        
        logged_count = 0
        for plot_name in test_plots:
            plot_path = plots_dir / plot_name
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), artifact_path="plots")
                logged_count += 1
        
        if logged_count > 0:
            print(f"   ✅ Logged {logged_count} test plots")
        
    except Exception as e:
        print(f"   ⚠️  Failed to log test plots: {e}")


def log_complete_evaluation(results, csv_path, images_dir, plots_dir, config):
    """
    Convenience function to log complete evaluation results
    
    Args:
        results: Results dict from evaluation_module.run_evaluation()
        csv_path: Path to per-sample CSV
        images_dir: Directory with prediction images
        plots_dir: Directory with plots
        config: Configuration module
    """
    if not config.MLFLOW_ENABLED or not mlflow.active_run():
        return
    
    print(f"\n{'='*70}")
    print(f"📦 LOGGING TEST EVALUATION TO MLFLOW")
    print(f"{'='*70}\n")
    
    # Log metrics
    print(f"   📊 Logging aggregated test metrics...")
    log_test_evaluation(
        results['aggregated'],
        results['per_sample']
    )
    
    # Log CSV
    print(f"   💾 Logging per-sample CSV...")
    log_per_sample_results_csv(csv_path)
    
    # Log ALL images (no limit for complete analysis)
    print(f"   🖼️  Logging prediction images...")
    log_qualitative_images(images_dir, max_images=None)  # ⭐ None = log ALL images
    
    # Log plots
    print(f"   📈 Logging test plots...")
    log_test_plots(plots_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ TEST EVALUATION LOGGED TO MLFLOW")
    print(f"   - Aggregated metrics: {len(results['aggregated'])} metrics")
    print(f"   - Per-sample results: {len(results['per_sample'])} samples")
    
    # Show volume metrics if available
    if results['per_sample'] and 'gt_volume_ml' in results['per_sample'][0]:
        print(f"   - Volume metrics: ✅ Included (gt_volume_ml, pred_volume_ml, volume_error_percent)")
    else:
        print(f"   - Volume metrics: ❌ Not found")
    
    print(f"   - Prediction images: logged to artifacts/predictions/")
    print(f"   - Test plots: logged to artifacts/plots/")
    print(f"{'='*70}\n")
