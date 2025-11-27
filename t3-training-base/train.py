"""
DWI Baseline Training - All-in-One Script
Simple and straightforward - everything in one file!

Usage:
    python train.py

This script does:
1. Load data from raw .nii.gz files (in-memory preprocessing)
2. Train Attention U-Net model
3. Evaluate on test set
4. Log everything to MLflow
"""

import os
import sys
import glob
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import mlflow
import mlflow.pytorch

# Import config and model
import config
from model import AttentionUNet, count_parameters


# ==================== Dataset ====================

class DWIDataset(Dataset):
    """Simple DWI Dataset with 2.5D slices"""
    
    def __init__(self, images, masks, transform=None):
        """
        Args:
            images: numpy array of shape (N, H, W) - normalized images
            masks: numpy array of shape (N, H, W) - binary masks
            transform: albumentations transform
        """
        self.images = images
        self.masks = masks
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]  # (3, H, W) - already 2.5D
        mask = self.masks[idx]    # (H, W)
        
        # Apply augmentation
        if self.transform:
            # Convert to HWC for albumentations
            image = np.transpose(image, (1, 2, 0))  # (H, W, 3)
            
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']  # (3, H, W) tensor
            mask = augmented['mask']    # (H, W) tensor
            
            # Add channel dimension to mask
            mask = mask.unsqueeze(0).float()  # (1, H, W)
        else:
            # No augmentation - just convert to tensor
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask


# ==================== Data Loading ====================

def load_and_preprocess_data():
    """
    Load all .nii.gz files, preprocess, and split into train/val/test
    All done in-memory - no saved files!
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "="*60)
    print("📂 Loading and Preprocessing Data")
    print("="*60)
    
    # Ensure RAW_DATA_PATH is a Path object
    from pathlib import Path
    raw_data_path = Path(config.RAW_DATA_PATH) if isinstance(config.RAW_DATA_PATH, str) else config.RAW_DATA_PATH
    
    # Find all image files
    images_dir = raw_data_path / "images"
    masks_dir = raw_data_path / "masks"
    
    image_files = sorted(glob.glob(str(images_dir / "*.nii.gz")))
    print(f"Found {len(image_files)} image files")
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No .nii.gz files found in {images_dir}")
    
    # Load all slices
    all_images = []
    all_masks = []
    
    print("\nLoading files...")
    for img_path in tqdm(image_files, desc="Loading"):
        # Get corresponding mask
        img_name = Path(img_path).name
        mask_path = masks_dir / img_name
        
        if not mask_path.exists():
            print(f"Warning: Mask not found for {img_name}, skipping...")
            continue
        
        # Load NIfTI files
        img_nii = nib.load(img_path)
        mask_nii = nib.load(str(mask_path))
        
        img_data = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()
        
        # Process each slice
        for slice_idx in range(img_data.shape[2]):
            # Get current slice
            img_slice = img_data[:, :, slice_idx]
            mask_slice = mask_data[:, :, slice_idx]
            
            # Skip empty masks
            if mask_slice.sum() == 0:
                continue
            
            # Resize
            img_slice = cv2.resize(img_slice, config.IMAGE_SIZE)
            mask_slice = cv2.resize(mask_slice, config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
            
            # Normalize image (z-score)
            img_slice = (img_slice - img_slice.mean()) / (img_slice.std() + 1e-8)
            
            # Binarize mask
            mask_slice = (mask_slice > 0.5).astype(np.float32)
            
            # Create 2.5D (3 channels: previous, current, next)
            # For edge cases, duplicate the slice
            if slice_idx == 0:
                prev_slice = img_slice
            else:
                prev_slice = img_data[:, :, slice_idx - 1]
                prev_slice = cv2.resize(prev_slice, config.IMAGE_SIZE)
                prev_slice = (prev_slice - prev_slice.mean()) / (prev_slice.std() + 1e-8)
            
            if slice_idx == img_data.shape[2] - 1:
                next_slice = img_slice
            else:
                next_slice = img_data[:, :, slice_idx + 1]
                next_slice = cv2.resize(next_slice, config.IMAGE_SIZE)
                next_slice = (next_slice - next_slice.mean()) / (next_slice.std() + 1e-8)
            
            # Stack to 3 channels (C, H, W)
            img_3d = np.stack([prev_slice, img_slice, next_slice], axis=0)
            
            all_images.append(img_3d)
            all_masks.append(mask_slice)
    
    # Convert to numpy arrays
    all_images = np.array(all_images, dtype=np.float32)
    all_masks = np.array(all_masks, dtype=np.float32)
    
    print(f"\nTotal slices loaded: {len(all_images)}")
    print(f"Image shape: {all_images.shape}")
    print(f"Mask shape: {all_masks.shape}")
    
    # Split into train/val/test
    np.random.seed(config.RANDOM_SEED)
    indices = np.random.permutation(len(all_images))
    
    n_train = int(len(indices) * config.TRAIN_RATIO)
    n_val = int(len(indices) * config.VAL_RATIO)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_images = all_images[train_idx]
    train_masks = all_masks[train_idx]
    val_images = all_images[val_idx]
    val_masks = all_masks[val_idx]
    test_images = all_images[test_idx]
    test_masks = all_masks[test_idx]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_images)} slices")
    print(f"  Val: {len(val_images)} slices")
    print(f"  Test: {len(test_images)} slices")
    
    # Create augmentation transforms
    if config.USE_AUGMENTATION:
        print("\n✓ Augmentation ENABLED for training")
        train_transform = A.Compose([
            A.HorizontalFlip(p=config.AUG_HFLIP_PROB),
            A.Rotate(limit=config.AUG_ROTATE_LIMIT, p=config.AUG_ROTATE_PROB),
            A.RandomBrightnessContrast(
                brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
                contrast_limit=config.AUG_BRIGHTNESS_LIMIT,
                p=config.AUG_BRIGHTNESS_PROB
            ),
            ToTensorV2()
        ])
    else:
        print("\n✗ Augmentation DISABLED (no augmentation for training)")
        train_transform = A.Compose([
            ToTensorV2()
        ])
    
    val_transform = A.Compose([
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = DWIDataset(train_images, train_masks, transform=train_transform)
    val_dataset = DWIDataset(val_images, val_masks, transform=val_transform)
    test_dataset = DWIDataset(test_images, test_masks, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader


# ==================== Loss Function ====================

class DiceLoss(nn.Module):
    """Simple Dice Loss"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


def calculate_dice_score(pred, target):
    """Calculate Dice score (metric)"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    
    return dice.item()


def calculate_iou(pred, target):
    """Calculate IoU (Jaccard Index)"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.item()


# ==================== Training ====================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    running_dice = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.USE_AMP:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            preds = (outputs > 0.5).float()
            dice = calculate_dice_score(preds, masks)
        
        running_loss += loss.item()
        running_dice += dice
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })
    
    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    
    return avg_loss, avg_dice


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            preds = (outputs > 0.5).float()
            dice = calculate_dice_score(preds, masks)
            
            running_loss += loss.item()
            running_dice += dice
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
    
    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    
    return avg_loss, avg_dice


# ==================== Evaluation ====================

def evaluate_test_set(model, test_loader, device):
    """
    Evaluate on test set and save predictions
    Returns: test_dice, test_iou, prediction images
    """
    print("\n" + "="*60)
    print("📊 Evaluating on Test Set")
    print("="*60)
    
    model.eval()
    
    all_dice = []
    all_iou = []
    predictions = []
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            # Calculate metrics
            dice = calculate_dice_score(preds, masks)
            iou = calculate_iou(preds, masks)
            
            all_dice.append(dice)
            all_iou.append(iou)
            
            # Save first 10 predictions for visualization
            if idx < 10:
                img_np = images[0, 1].cpu().numpy()  # Middle slice
                mask_np = masks[0, 0].cpu().numpy()
                pred_np = preds[0, 0].cpu().numpy()
                
                predictions.append({
                    'image': img_np,
                    'mask': mask_np,
                    'pred': pred_np,
                    'dice': dice,
                    'iou': iou
                })
    
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    mean_iou = np.mean(all_iou)
    std_iou = np.std(all_iou)
    
    print(f"\nTest Results:")
    print(f"  Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"  IoU Score: {mean_iou:.4f} ± {std_iou:.4f}")
    print("="*60 + "\n")
    
    return mean_dice, std_dice, mean_iou, std_iou, predictions


def save_test_predictions(predictions, save_dir):
    """Save test prediction visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, pred_data in enumerate(predictions):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image
        axes[0].imshow(pred_data['image'], cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Ground Truth
        axes[1].imshow(pred_data['image'], cmap='gray')
        axes[1].imshow(pred_data['mask'], cmap='Reds', alpha=0.5)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_data['image'], cmap='gray')
        axes[2].imshow(pred_data['pred'], cmap='Greens', alpha=0.5)
        axes[2].set_title(f"Prediction\nDice: {pred_data['dice']:.4f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sample_{idx+1:03d}.png", dpi=100, bbox_inches='tight')
        plt.close()


def plot_training_curves(history, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice
    axes[1].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[1].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Training and Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==================== Main Training Loop ====================

def main():
    """Main training function"""
    
    # Print configuration
    config.print_config()
    config.create_directories()
    
    # Set random seeds
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Load data
    train_loader, val_loader, test_loader = load_and_preprocess_data()
    
    # Create model
    print("="*60)
    print("🏗️  Creating Model")
    print("="*60)
    
    model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS
    ).to(config.DEVICE)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print("="*60 + "\n")
    
    # Loss, optimizer, scheduler
    criterion = DiceLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if config.USE_AMP else None
    
    # Setup MLflow
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    run_name = f"baseline_{datetime.now():%Y%m%d_%H%M%S}"
    
    with mlflow.start_run(run_name=run_name):
        print("="*60)
        print("🔬 MLflow Tracking Initialized")
        print("="*60)
        print(f"Experiment: {config.MLFLOW_EXPERIMENT_NAME}")
        print(f"Run Name: {run_name}")
        print("="*60 + "\n")
        
        # Log parameters
        mlflow.log_param("image_size", config.IMAGE_SIZE)
        mlflow.log_param("batch_size", config.BATCH_SIZE)
        mlflow.log_param("learning_rate", config.LEARNING_RATE)
        mlflow.log_param("weight_decay", config.WEIGHT_DECAY)
        mlflow.log_param("epochs", config.EPOCHS)
        mlflow.log_param("base_channels", config.BASE_CHANNELS)
        mlflow.log_param("early_stop_patience", config.EARLY_STOP_PATIENCE)
        mlflow.log_param("train_samples", len(train_loader.dataset))
        mlflow.log_param("val_samples", len(val_loader.dataset))
        mlflow.log_param("test_samples", len(test_loader.dataset))
        mlflow.log_param("model_params", total_params)
        mlflow.log_param("use_augmentation", config.USE_AUGMENTATION)
        if config.USE_AUGMENTATION:
            mlflow.log_param("augmentation", "hflip+rotate+brightness")
            mlflow.log_param("aug_hflip_prob", config.AUG_HFLIP_PROB)
            mlflow.log_param("aug_rotate_limit", config.AUG_ROTATE_LIMIT)
            mlflow.log_param("aug_rotate_prob", config.AUG_ROTATE_PROB)
            mlflow.log_param("aug_brightness_prob", config.AUG_BRIGHTNESS_PROB)
        else:
            mlflow.log_param("augmentation", "none")
        mlflow.log_param("random_seed", config.RANDOM_SEED)
        
        # Training loop
        print("="*60)
        print("🚀 Starting Training")
        print("="*60 + "\n")
        
        best_dice = 0.0
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_dice': [],
            'val_loss': [],
            'val_dice': []
        }
        
        for epoch in range(config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
            print("-" * 60)
            
            # Train
            train_loss, train_dice = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, config.DEVICE
            )
            
            # Validate
            val_loss, val_dice = validate(model, val_loader, criterion, config.DEVICE)
            
            # Update scheduler
            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_dice'].append(train_dice)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_dice", train_dice, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            # Print epoch summary
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                torch.save(model.state_dict(), config.MODELS_DIR / "best_model.pth")
                print(f"  ✅ New best model saved! (Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{config.EARLY_STOP_PATIENCE})")
            
            # Early stopping
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "="*60)
        print("✅ Training Completed!")
        print("="*60)
        print(f"Best Val Dice: {best_dice:.4f}")
        print("="*60 + "\n")
        
        # Plot training curves
        with tempfile.TemporaryDirectory() as tmpdir:
            curve_path = os.path.join(tmpdir, "training_curve.png")
            plot_training_curves(history, curve_path)
            mlflow.log_artifact(curve_path)
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(config.MODELS_DIR / "best_model.pth"))
        
        # Evaluate on test set
        test_dice, test_dice_std, test_iou, test_iou_std, predictions = evaluate_test_set(
            model, test_loader, config.DEVICE
        )
        
        # Log test metrics
        mlflow.log_metric("test_dice_mean", test_dice)
        mlflow.log_metric("test_dice_std", test_dice_std)
        mlflow.log_metric("test_iou_mean", test_iou)
        mlflow.log_metric("test_iou_std", test_iou_std)
        
        # Save test predictions
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = os.path.join(tmpdir, "test_predictions")
            save_test_predictions(predictions, pred_dir)
            mlflow.log_artifacts(pred_dir)
        
        # Save test metrics JSON
        test_metrics = {
            'dice_mean': float(test_dice),
            'dice_std': float(test_dice_std),
            'iou_mean': float(test_iou),
            'iou_std': float(test_iou_std),
            'best_val_dice': float(best_dice)
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = os.path.join(tmpdir, "test_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            mlflow.log_artifact(metrics_path)
        
        # Log best model
        mlflow.log_artifact(str(config.MODELS_DIR / "best_model.pth"))
        
        print("\n" + "="*60)
        print("📦 All Artifacts Logged to MLflow")
        print("="*60)
        print("\nView results:")
        print("  mlflow ui --port 5000")
        print("  Then open: http://localhost:5000")
        print("\n" + "="*60)
        print("🎉 Done!")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
