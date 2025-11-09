"""
Training Script for DWI Ischemic Stroke Segmentation
Complete training pipeline with validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import json

# Import our modules
import config
from models import get_model  # ⬆️ Updated to support multiple architectures
from loss import get_loss_function
from dataset import create_dataloaders
from utils import (
    calculate_dice_score, 
    save_training_history, 
    print_metrics_table,
    build_slice_mapping,
    plot_training_curves_advanced
)
from mlflow_utils import (
    setup_mlflow,
    log_config_params,
    log_epoch_metrics,
    log_training_complete,
    log_complete_evaluation,
    end_run
)
from evaluation_module import (
    run_evaluation,
    generate_qualitative_results,
    print_evaluation_summary,
    save_per_sample_results_csv,
    plot_metrics_distribution
)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like Dice
    """
    
    def __init__(self, patience=15, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, cfg, scaler=None):
    """
    Train for one epoch
    
    Args:
        model: Neural network model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epoch: Current epoch number
        cfg: Configuration module
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        dict: {'loss': float, 'dice': float}
    """
    model.train()
    
    running_loss = 0.0
    running_dice = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch (handle both 2-tuple and 3-tuple for backward compatibility)
        if len(batch_data) == 3:
            images, masks, _ = batch_data  # Ignore filename during training
        else:
            images, masks = batch_data
        
        # Move to device
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping (ป้องกัน exploding gradients)
            if hasattr(cfg, 'GRADIENT_CLIP_VALUE'):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_VALUE)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Normal training
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping (ป้องกัน exploding gradients)
            if hasattr(cfg, 'GRADIENT_CLIP_VALUE'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_VALUE)
            
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Convert to binary predictions
            preds = (outputs > 0.5).float()
            batch_dice = calculate_dice_score(
                preds.cpu().numpy(), 
                masks.cpu().numpy()
            )
        
        # Update running metrics
        running_loss += loss.item()
        running_dice += batch_dice
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{batch_dice:.4f}'
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / num_batches
    epoch_dice = running_dice / num_batches
    
    return {
        'loss': epoch_loss,
        'dice': epoch_dice
    }


def validate_one_epoch(model, dataloader, criterion, device, epoch):
    """
    Validate for one epoch
    
    Args:
        model: Neural network model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device (cuda/cpu)
        epoch: Current epoch number
    
    Returns:
        dict: {'loss': float, 'dice': float}
    """
    model.eval()
    
    running_loss = 0.0
    running_dice = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    with torch.no_grad():
        for batch_data in pbar:
            # Unpack batch (handle both 2-tuple and 3-tuple for backward compatibility)
            if len(batch_data) == 3:
                images, masks, _ = batch_data  # Ignore filename during validation
            else:
                images, masks = batch_data
            
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            preds = (outputs > 0.5).float()
            batch_dice = calculate_dice_score(
                preds.cpu().numpy(), 
                masks.cpu().numpy()
            )
            
            # Update running metrics
            running_loss += loss.item()
            running_dice += batch_dice
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{batch_dice:.4f}'
            })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / num_batches
    epoch_dice = running_dice / num_batches
    
    return {
        'loss': epoch_loss,
        'dice': epoch_dice
    }


def train_model(cfg):
    """
    Main training function
    
    Args:
        cfg: Configuration module
    """
    print("\n" + "="*70)
    print("🎓 TRAINING DWI SEGMENTATION MODEL")
    print("="*70)
    
    # Print configuration
    cfg.print_config()
    
    # Create directories
    cfg.create_directories()
    
    # Set device
    device = cfg.DEVICE
    print(f"\n💻 Using device: {device}")
    if cfg.USE_CUDA:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create dataloaders
    print("\n📦 Creating dataloaders...")
    dataloaders = create_dataloaders(cfg)
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Batch size: {cfg.BATCH_SIZE}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = get_model(cfg)  # ⬆️ Updated to support multiple architectures
    model = model.to(device)
    
    # Count parameters (if method exists)
    model_params = None
    if hasattr(model, 'count_parameters'):
        params = model.count_parameters()
        model_params = params
        print(f"   Total parameters: {params['total']:,}")
        print(f"   Trainable parameters: {params['trainable']:,}")
        print(f"   Model size: ~{params['total'] * 4 / (1024**2):.2f} MB")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_params = {'total': total_params, 'trainable': trainable_params}
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / (1024**2):.2f} MB")
    
    # Initialize MLflow tracking
    mlflow_run = setup_mlflow(cfg, model_params)
    
    # Create loss function
    print("\n📉 Creating loss function...")
    criterion = get_loss_function(
        loss_type=cfg.LOSS_TYPE,
        focal_weight=cfg.COMBO_FOCAL_WEIGHT,
        dice_weight=cfg.COMBO_DICE_WEIGHT,
        focal_alpha=cfg.FOCAL_ALPHA,
        focal_gamma=cfg.FOCAL_GAMMA,
        dice_smooth=cfg.DICE_SMOOTH
    )
    print(f"   Loss type: {cfg.LOSS_TYPE.upper()}")
    
    # Create optimizer
    print("\n⚙️  Creating optimizer...")
    if cfg.OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )
    elif cfg.OPTIMIZER.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.OPTIMIZER}")
    
    print(f"   Optimizer: {cfg.OPTIMIZER.upper()}")
    print(f"   Learning rate: {cfg.LEARNING_RATE}")
    print(f"   Weight decay: {cfg.WEIGHT_DECAY}")
    
    # Create learning rate scheduler
    print("\n📊 Creating LR scheduler...")
    if cfg.SCHEDULER.lower() == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize validation Dice
            factor=cfg.SCHEDULER_FACTOR,
            patience=cfg.SCHEDULER_PATIENCE,
            min_lr=cfg.SCHEDULER_MIN_LR
        )
        print(f"   Scheduler: ReduceLROnPlateau")
        print(f"   Patience: {cfg.SCHEDULER_PATIENCE} epochs")
        print(f"   Factor: {cfg.SCHEDULER_FACTOR}")
    elif cfg.SCHEDULER.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.NUM_EPOCHS,
            eta_min=cfg.SCHEDULER_MIN_LR
        )
        print(f"   Scheduler: CosineAnnealingLR")
    else:
        scheduler = None
        print(f"   Scheduler: None")
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE,
        min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
        mode='max'  # Maximize validation Dice
    )
    print(f"\n⏹️  Early stopping patience: {cfg.EARLY_STOPPING_PATIENCE} epochs")
    
    # Mixed precision training
    scaler = None
    if cfg.USE_MIXED_PRECISION and cfg.USE_CUDA:
        scaler = torch.amp.GradScaler('cuda')
        print(f"   Mixed precision: Enabled")
    
    # Log configuration parameters to MLflow
    log_config_params(cfg)
    
    # Training history
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'learning_rates': []
    }
    
    best_val_dice = 0.0
    best_epoch = 0
    
    # Start training
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg, scaler
        )
        
        # Validate
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        if scheduler is not None:
            if cfg.SCHEDULER.lower() == 'reduce_on_plateau':
                scheduler.step(val_metrics['dice'])
            else:
                scheduler.step()
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_dice'].append(val_metrics['dice'])
        history['learning_rates'].append(current_lr)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Log metrics to MLflow
        log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr, epoch_time)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{cfg.NUM_EPOCHS} - {epoch_time:.1f}s - LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Dice: {train_metrics['dice']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Dice:   {val_metrics['dice']:.4f}")
        
        # Check if best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            best_epoch = epoch
            
            # Save best model
            best_model_path = cfg.get_model_save_path('best_model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_metrics['dice'],
                'val_loss': val_metrics['loss']
            }, best_model_path)
            
            print(f"  ✅ New best model! Val Dice: {val_metrics['dice']:.4f} (saved)")
        
        # Check early stopping
        if early_stopping(val_metrics['dice']):
            print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
            print(f"   Best Val Dice: {best_val_dice:.4f} at epoch {best_epoch}")
            break
        
        # Save checkpoint every N epochs
        if epoch % 10 == 0:
            checkpoint_path = cfg.get_checkpoint_path(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_metrics['dice'],
                'val_loss': val_metrics['loss'],
                'history': history
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: epoch_{epoch:03d}.pth")
        
        print("-" * 70)
    
    # Training completed
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"\nTotal training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Best validation Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {cfg.get_model_save_path('best_model')}")
    
    # Save training history
    history_path = cfg.RESULTS_DIR / "training_history.json"
    save_training_history(history, history_path)
    
    # Generate advanced training curves
    curves_path = cfg.PLOTS_DIR / 'training_curves_advanced.png'
    plot_training_curves_advanced(
        history, 
        best_epoch=best_epoch, 
        save_path=curves_path
    )
    print(f"\n📊 Advanced training curves saved to: {curves_path}")
    
    # Log all training artifacts to MLflow
    best_model_path = cfg.get_model_save_path('best_model')
    log_training_complete(
        cfg, best_val_dice, best_epoch, epoch,
        train_metrics, val_metrics, total_time,
        best_model_path, history_path
    )
    
    # Save final model
    final_model_path = cfg.get_model_save_path('final_model')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_metrics['dice'],
        'val_loss': val_metrics['loss']
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    
    summary = {
        'Best Val Dice': f"{best_val_dice:.4f}",
        'Best Epoch': best_epoch,
        'Total Epochs': epoch,
        'Final Train Loss': f"{train_metrics['loss']:.4f}",
        'Final Train Dice': f"{train_metrics['dice']:.4f}",
        'Final Val Loss': f"{val_metrics['loss']:.4f}",
        'Final Val Dice': f"{val_metrics['dice']:.4f}",
        'Training Time': f"{total_time/60:.1f} min"
    }
    
    print_metrics_table(summary, "Training Summary")
    
    # ======================================================================
    # AUTO-EVALUATE ON TEST SET
    # ======================================================================
    print("\n" + "="*70)
    print("🧪 RUNNING AUTOMATIC TEST EVALUATION")
    print("="*70)
    
    try:
        # Load best model for evaluation
        print(f"\n📂 Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Run evaluation on test set
        test_loader = dataloaders['test']
        print(f"📊 Evaluating on {len(test_loader.dataset)} test samples...")
        
        results = run_evaluation(
            model=model,
            test_loader=test_loader,
            device=device,
            config=cfg,
            show_progress=True
        )
        
        # Display formatted results
        print_evaluation_summary(results, show_per_sample_top=5)
        
        # Save per-sample results to CSV
        csv_path = cfg.RESULTS_DIR / "test_per_sample_results.csv"
        save_per_sample_results_csv(results['per_sample'], csv_path)
        print(f"\n💾 Saved per-sample results to: {csv_path}")
        print(f"   Total samples: {len(results['per_sample'])}")
        print(f"   Columns: filename, dice, iou, precision, recall, f1")
        
        # Generate qualitative results (prediction images with filenames)
        print(f"\n🖼️  Generating prediction visualizations...")
        saved_images = generate_qualitative_results(
            sample_results=results['sample_results'],
            save_dir=cfg.PREDICTIONS_DIR,
            config=cfg,
            num_samples=10
        )
        print(f"   ✅ Generated {len(saved_images)} prediction images in: {cfg.PREDICTIONS_DIR}")
        
        # Plot metrics distribution
        print(f"\n📊 Creating metrics distribution plots...")
        plot_metrics_distribution(results['per_sample'], cfg.PLOTS_DIR)
        print(f"   ✅ Saved distribution plot to: {cfg.PLOTS_DIR / 'test_metrics_distribution.png'}")
        
        # Log complete evaluation to MLflow
        if cfg.MLFLOW_ENABLED:
            print(f"\n📦 Logging test evaluation to MLflow...")
            log_complete_evaluation(
                results=results,
                csv_path=csv_path,
                images_dir=cfg.PREDICTIONS_DIR,
                plots_dir=cfg.PLOTS_DIR,
                config=cfg
            )
            print("   ✅ Test evaluation logged successfully")
        
        print("\n" + "="*70)
        print("✅ TEST EVALUATION COMPLETED!")
        print("="*70)
        
    except Exception as e:
        print(f"\n⚠️  Error during test evaluation: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback:")
        traceback.print_exc()
        print("   Training completed successfully, but test evaluation failed.")
        print("   You can run evaluation manually: python evaluate.py")
    
    # End MLflow run
    end_run(status="FINISHED")
    
    print("\n🎉 Training and evaluation complete!")
    print(f"   ✓ Training time: {total_time/60:.1f} minutes")
    print(f"   ✓ Best validation Dice: {best_val_dice:.4f}")
    print(f"   ✓ Test evaluation: Completed")
    print(f"   ✓ Results saved to: {cfg.RESULTS_DIR}")
    if cfg.MLFLOW_ENABLED:
        print(f"   ✓ View MLflow UI: mlflow ui --backend-store-uri {cfg.MLFLOW_TRACKING_URI}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Check if processed data exists
    if not config.PROCESSED_TRAIN_IMG.exists():
        print("❌ Processed data not found!")
        print(f"   Expected location: {config.PROCESSED_TRAIN_IMG}")
        print(f"\n   Please run preprocessing first:")
        print(f"   python 01_preprocess.py")
        exit(1)
    
    # Start training with error handling for MLflow
    try:
        train_model(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        end_run(status="KILLED")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        end_run(status="FAILED")
        exit(1)

