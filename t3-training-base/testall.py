"""
Test All Data Script for DWI Baseline Training
Load all data from 1_data_raw, run inference with trained model, and save results
Auto-creates timestamped output folders for each run

Usage:
    python testall.py
"""

import os
import sys
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# Import config and model
import config
from model import AttentionUNet


# ============================================================================
# CONFIGURATION - กำหนดค่าที่นี่
# ============================================================================
CONFIG = {
    'model_path': 'models/best_model.pth',  # path to trained model
    'batch_size': 4,                         # batch size for inference
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,                        # number of data loading workers
    'output_base': 'test_results',           # base folder for results
    'dpi': 300,                              # PNG resolution (high quality)
    'image_format': 'png',                   # output format
}


# ==================== Dataset ====================

class TestDataset(Dataset):
    """Simple dataset for testing - no augmentation"""
    
    def __init__(self, images, masks, filenames):
        """
        Args:
            images: numpy array of shape (N, 3, H, W) - 2.5D images
            masks: numpy array of shape (N, H, W) - binary masks
            filenames: list of filenames for saving results
        """
        self.images = images
        self.masks = masks
        self.filenames = filenames
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()  # (3, H, W)
        mask = torch.from_numpy(self.masks[idx]).unsqueeze(0).float()  # (1, H, W)
        filename = self.filenames[idx]
        
        return image, mask, filename


# ==================== Data Loading ====================

def load_all_data():
    """
    Load all data from 1_data_raw folder
    Returns: images, masks, filenames
    """
    print("\n" + "="*60)
    print("📂 Loading All Data for Testing")
    print("="*60)
    
    from pathlib import Path
    raw_data_path = Path(config.RAW_DATA_PATH) if isinstance(config.RAW_DATA_PATH, str) else config.RAW_DATA_PATH
    
    # Find all image files
    images_dir = raw_data_path / "images"
    masks_dir = raw_data_path / "masks"
    
    # Try .npy files first, then .nii.gz
    image_files = sorted(glob.glob(str(images_dir / "*.npy")))
    if len(image_files) == 0:
        image_files = sorted(glob.glob(str(images_dir / "*.nii.gz")))
        file_type = "nii.gz"
    else:
        file_type = "npy"
    
    print(f"Found {len(image_files)} {file_type} image files")
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No .npy or .nii.gz files found in {images_dir}")
    
    # Load all slices
    all_images = []
    all_masks = []
    all_filenames = []
    
    print(f"\nLoading {file_type} files...")
    for img_path in tqdm(image_files, desc="Loading"):
        # Get corresponding mask
        img_name = Path(img_path).name
        
        if file_type == "npy":
            mask_path = masks_dir / img_name
        else:
            mask_path = masks_dir / img_name
        
        if not mask_path.exists():
            print(f"Warning: Mask not found for {img_name}, skipping...")
            continue
        
        # Load data based on file type
        if file_type == "npy":
            img_data = np.load(img_path)  # Can be (3, H, W) or (H, W, D)
            mask_data = np.load(str(mask_path))  # Can be (H, W) or (H, W, D)
        else:
            import nibabel as nib
            img_nii = nib.load(img_path)
            mask_nii = nib.load(str(mask_path))
            img_data = img_nii.get_fdata()
            mask_data = mask_nii.get_fdata()
        
        # Check if already preprocessed (2.5D format: 3, H, W)
        if img_data.ndim == 3 and img_data.shape[0] == 3:
            # Already preprocessed - use directly
            if mask_data.ndim == 2:
                # Single mask
                base_name = Path(img_name).stem
                filename = f"{base_name}"
                
                # Resize if needed
                if img_data.shape[1:] != tuple(config.IMAGE_SIZE):
                    img_3d = np.stack([
                        cv2.resize(img_data[0], config.IMAGE_SIZE),
                        cv2.resize(img_data[1], config.IMAGE_SIZE),
                        cv2.resize(img_data[2], config.IMAGE_SIZE)
                    ], axis=0)
                    mask_resized = cv2.resize(mask_data, config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
                else:
                    img_3d = img_data
                    mask_resized = mask_data
                
                # Binarize mask
                mask_binary = (mask_resized > 0.5).astype(np.float32)
                
                all_images.append(img_3d)
                all_masks.append(mask_binary)
                all_filenames.append(filename)
            else:
                print(f"Warning: Unexpected mask shape for {img_name}: {mask_data.shape}, skipping...")
                
        else:
            # Raw 3D data - process each slice
            # Handle different data shapes
            if img_data.ndim == 2:
                img_data = img_data[:, :, np.newaxis]
                mask_data = mask_data[:, :, np.newaxis]
            
            # Process each slice
            for slice_idx in range(img_data.shape[2]):
                # Get current slice
                img_slice = img_data[:, :, slice_idx]
                mask_slice = mask_data[:, :, slice_idx]
                
                # Resize
                img_slice_resized = cv2.resize(img_slice, config.IMAGE_SIZE)
                mask_slice_resized = cv2.resize(mask_slice, config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
                
                # Normalize image (z-score)
                img_slice_norm = (img_slice_resized - img_slice_resized.mean()) / (img_slice_resized.std() + 1e-8)
                
                # Binarize mask
                mask_slice_binary = (mask_slice_resized > 0.5).astype(np.float32)
                
                # Create 2.5D (3 channels: previous, current, next)
                if slice_idx == 0:
                    prev_slice = img_slice_norm
                else:
                    prev_slice = img_data[:, :, slice_idx - 1]
                    prev_slice = cv2.resize(prev_slice, config.IMAGE_SIZE)
                    prev_slice = (prev_slice - prev_slice.mean()) / (prev_slice.std() + 1e-8)
                
                if slice_idx == img_data.shape[2] - 1:
                    next_slice = img_slice_norm
                else:
                    next_slice = img_data[:, :, slice_idx + 1]
                    next_slice = cv2.resize(next_slice, config.IMAGE_SIZE)
                    next_slice = (next_slice - next_slice.mean()) / (next_slice.std() + 1e-8)
                
                # Stack to 3 channels (C, H, W)
                img_3d = np.stack([prev_slice, img_slice_norm, next_slice], axis=0)
                
                # Create filename
                base_name = Path(img_name).stem
                filename = f"{base_name}_slice_{slice_idx:03d}"
                
                all_images.append(img_3d)
                all_masks.append(mask_slice_binary)
                all_filenames.append(filename)
    
    # Convert to numpy arrays
    all_images = np.array(all_images, dtype=np.float32)
    all_masks = np.array(all_masks, dtype=np.float32)
    
    print(f"\nTotal slices loaded: {len(all_images)}")
    print(f"Image shape: {all_images.shape}")
    print(f"Mask shape: {all_masks.shape}")
    print("="*60 + "\n")
    
    # Create and return dataset
    dataset = TestDataset(all_images, all_masks, all_filenames)
    return dataset


# ==================== Testing ====================

def test_all(model_path, dataloader, output_dir, device):
    """
    Run inference on all data and save results
    
    Args:
        model_path: Path to trained model checkpoint
        dataloader: DataLoader with test data
        output_dir: Directory to save results
        device: Device to run on (cuda/cpu)
    """
    print("\n" + "="*60)
    print("🔬 Running Inference on All Data")
    print("="*60)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Store metrics for summary
    all_dice_scores = []
    all_iou_scores = []
    all_filenames_list = []
    
    with torch.no_grad():
        for idx, (images, masks, filenames) in enumerate(tqdm(dataloader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            # Process each item in batch
            for i in range(images.shape[0]):
                # Get data
                img_slice = images[i, 1].cpu().numpy()  # Middle slice (normalized)
                mask_gt = masks[i, 0].cpu().numpy()
                pred_mask = preds[i, 0].cpu().numpy()
                filename = filenames[i]
                
                # Calculate metrics
                dice = calculate_dice(mask_gt, pred_mask)
                iou = calculate_iou(mask_gt, pred_mask)
                
                all_dice_scores.append(dice)
                all_iou_scores.append(iou)
                all_filenames_list.append(filename)
                
                # Save visualization with metrics
                save_prediction_image(
                    img_slice, 
                    mask_gt, 
                    pred_mask, 
                    output_dir / f"{filename}.png",
                    dice_score=dice,
                    iou_score=iou
                )
    
    print(f"\n✅ All results saved to: {output_dir}")
    
    # Generate and save summary
    print("\n" + "="*60)
    print("📊 Generating Summary Report")
    print("="*60)
    
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    generate_summary_report(
        all_filenames_list,
        all_dice_scores,
        all_iou_scores,
        summary_dir
    )
    
    print("="*60 + "\n")


# ==================== Metrics Calculation ====================

def calculate_dice(mask_gt, pred_mask, smooth=1e-6):
    """
    Calculate Dice coefficient
    
    Args:
        mask_gt: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice score (0 to 1)
    """
    mask_gt_flat = mask_gt.flatten()
    pred_mask_flat = pred_mask.flatten()
    
    intersection = np.sum(mask_gt_flat * pred_mask_flat)
    union = np.sum(mask_gt_flat) + np.sum(pred_mask_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def calculate_iou(mask_gt, pred_mask, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        mask_gt: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score (0 to 1)
    """
    mask_gt_flat = mask_gt.flatten()
    pred_mask_flat = pred_mask.flatten()
    
    intersection = np.sum(mask_gt_flat * pred_mask_flat)
    union = np.sum(mask_gt_flat) + np.sum(pred_mask_flat) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


# ==================== Summary Report ====================

def generate_summary_report(filenames, dice_scores, iou_scores, summary_dir):
    """
    Generate summary report with histograms and CSV files
    
    Args:
        filenames: List of file names
        dice_scores: List of Dice scores
        iou_scores: List of IoU scores
        summary_dir: Directory to save summary files
    """
    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'dice_score': dice_scores,
        'iou_score': iou_scores
    })
    
    # Save detailed CSV
    csv_path = summary_dir / 'detailed_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Detailed metrics saved to: {csv_path}")
    
    # Calculate statistics
    dice_mean = np.mean(dice_scores)
    dice_std = np.std(dice_scores)
    dice_median = np.median(dice_scores)
    iou_mean = np.mean(iou_scores)
    iou_std = np.std(iou_scores)
    iou_median = np.median(iou_scores)
    
    # Define bins for histogram
    dice_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    iou_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Count samples in each bin
    dice_hist, _ = np.histogram(dice_scores, bins=dice_bins)
    iou_hist, _ = np.histogram(iou_scores, bins=iou_bins)
    
    # Create bin labels
    dice_labels = [f'{dice_bins[i]:.1f}-{dice_bins[i+1]:.1f}' for i in range(len(dice_bins)-1)]
    iou_labels = [f'{iou_bins[i]:.1f}-{iou_bins[i+1]:.1f}' for i in range(len(iou_bins)-1)]
    
    # Save histogram data to CSV
    hist_df = pd.DataFrame({
        'dice_range': dice_labels,
        'dice_count': dice_hist,
        'iou_range': iou_labels,
        'iou_count': iou_hist
    })
    hist_csv_path = summary_dir / 'histogram_data.csv'
    hist_df.to_csv(hist_csv_path, index=False)
    print(f"✅ Histogram data saved to: {hist_csv_path}")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame({
        'metric': ['Dice', 'IoU'],
        'mean': [dice_mean, iou_mean],
        'std': [dice_std, iou_std],
        'median': [dice_median, iou_median],
        'min': [np.min(dice_scores), np.min(iou_scores)],
        'max': [np.max(dice_scores), np.max(iou_scores)]
    })
    stats_csv_path = summary_dir / 'statistics.csv'
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"✅ Statistics saved to: {stats_csv_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Dice histogram
    bars1 = axes[0].bar(range(len(dice_hist)), dice_hist, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Dice Score Range', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Dice Score Distribution\nMean: {dice_mean:.4f} ± {dice_std:.4f} | Median: {dice_median:.4f}', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(dice_labels)))
    axes[0].set_xticklabels(dice_labels, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars1, dice_hist)):
        if count > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dice_hist)*0.01,
                        f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # IoU histogram
    bars2 = axes[1].bar(range(len(iou_hist)), iou_hist, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('IoU Score Range', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[1].set_title(f'IoU Score Distribution\nMean: {iou_mean:.4f} ± {iou_std:.4f} | Median: {iou_median:.4f}', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(iou_labels)))
    axes[1].set_xticklabels(iou_labels, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars2, iou_hist)):
        if count > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(iou_hist)*0.01,
                        f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot with high DPI
    plot_path = summary_dir / 'metrics_distribution.png'
    plt.savefig(plot_path, dpi=CONFIG['dpi'], bbox_inches='tight', format='png')
    plt.close()
    
    print(f"✅ Distribution plot saved to: {plot_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📈 SUMMARY STATISTICS")
    print("="*60)
    print(f"Total images: {len(dice_scores)}")
    print(f"\nDice Score:")
    print(f"  Mean:   {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"  Median: {dice_median:.4f}")
    print(f"  Min:    {np.min(dice_scores):.4f}")
    print(f"  Max:    {np.max(dice_scores):.4f}")
    print(f"\nIoU Score:")
    print(f"  Mean:   {iou_mean:.4f} ± {iou_std:.4f}")
    print(f"  Median: {iou_median:.4f}")
    print(f"  Min:    {np.min(iou_scores):.4f}")
    print(f"  Max:    {np.max(iou_scores):.4f}")
    print("="*60)


def save_prediction_image(img_slice, mask_gt, pred_mask, save_path, dice_score=None, iou_score=None):
    """
    Save high-quality PNG visualization with 3 panels: Original, Ground Truth, Prediction
    Display Dice and IoU scores on the image
    
    Args:
        img_slice: Normalized image slice (H, W)
        mask_gt: Ground truth binary mask (H, W)
        pred_mask: Predicted binary mask (H, W)
        save_path: Path to save the result
        dice_score: Dice coefficient score (optional)
        iou_score: IoU score (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth (Black and White - no overlay)
    axes[1].imshow(mask_gt, cmap='gray')
    axes[1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction (Green overlay on original)
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(pred_mask, cmap='Greens', alpha=0.5)
    axes[2].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add metrics as text if provided
    if dice_score is not None and iou_score is not None:
        # Add text box with metrics at the top
        metrics_text = f'Dice: {dice_score:.4f}  |  IoU: {iou_score:.4f}'
        fig.text(0.5, 0.95, metrics_text, 
                ha='center', va='top',
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for metrics text
    # Save with high DPI from config
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', format=CONFIG['image_format'])
    plt.close()


# ==================== Main ====================

def main():
    """Main testing function"""
    
    print("="*60)
    print("DWI BASELINE - Test All Data")
    print("="*60)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(CONFIG['output_base'])
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🖼️  Image format: {CONFIG['image_format'].upper()}")
    print(f"📊 Resolution: {CONFIG['dpi']} DPI")
    print(f"🎯 Model: {CONFIG['model_path']}")
    print(f"💻 Device: {CONFIG['device']}")
    print(f"📦 Batch size: {CONFIG['batch_size']}\n")
    
    # Load all data
    print("Loading data...")
    dataset = load_all_data()
    print(f"✅ Found {len(dataset)} test samples\n")
    
    if len(dataset) == 0:
        print("❌ No data found! Please check RAW_DATA_PATH in config.py")
        return
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    # Run testing
    test_all(
        model_path=CONFIG['model_path'],
        dataloader=dataloader,
        output_dir=output_dir,
        device=CONFIG['device']
    )


if __name__ == '__main__':
    main()
