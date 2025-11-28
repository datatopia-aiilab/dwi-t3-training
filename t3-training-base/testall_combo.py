"""
Test All Data Script - Combo Mode (Lesion + Artifact Detection)
Uses 2 models:
  1. Lesion detection model
  2. Artifact detection model
Then removes lesion predictions that overlap with artifacts to get refined results

Usage:
    python testall_combo.py
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
    # Model paths
    'lesion_model_path': 'models/best_model.pth',           # Model สำหรับตรวจ lesion
    'artifact_model_path': 'models_artf/best_model.pth',    # Model สำหรับตรวจ artifact
    
    # Data settings
    'batch_size': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    
    # Output settings
    'output_base': 'test_results_combo',
    'dpi': 300,
    'image_format': 'png',
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
    Returns: TestDataset
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


# ==================== Testing with 2 Models ====================

def test_combo(lesion_model_path, artifact_model_path, dataloader, output_dir, device):
    """
    Run inference with both models and refine lesion predictions
    
    Args:
        lesion_model_path: Path to lesion detection model
        artifact_model_path: Path to artifact detection model
        dataloader: DataLoader with test data
        output_dir: Directory to save results
        device: Device to run on (cuda/cpu)
    """
    print("\n" + "="*60)
    print("🔬 Running Combo Inference (Lesion + Artifact)")
    print("="*60)
    
    # Load lesion model
    print(f"Loading lesion model from: {lesion_model_path}")
    lesion_model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS
    ).to(device)
    lesion_model.load_state_dict(torch.load(lesion_model_path, map_location=device))
    lesion_model.eval()
    print("✅ Lesion model loaded!")
    
    # Load artifact model
    print(f"Loading artifact model from: {artifact_model_path}")
    artifact_model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS
    ).to(device)
    artifact_model.load_state_dict(torch.load(artifact_model_path, map_location=device))
    artifact_model.eval()
    print("✅ Artifact model loaded!\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Store metrics for summary
    all_dice_before = []
    all_iou_before = []
    all_dice_after = []
    all_iou_after = []
    all_filenames_list = []
    
    with torch.no_grad():
        for idx, (images, masks, filenames) in enumerate(tqdm(dataloader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions from both models
            lesion_outputs = lesion_model(images)
            artifact_outputs = artifact_model(images)
            
            lesion_preds = (lesion_outputs > 0.5).float()
            artifact_preds = (artifact_outputs > 0.5).float()
            
            # Process each item in batch
            for i in range(images.shape[0]):
                # Get data
                img_slice = images[i, 1].cpu().numpy()  # Middle slice (normalized)
                mask_gt = masks[i, 0].cpu().numpy()
                pred_lesion = lesion_preds[i, 0].cpu().numpy()
                pred_artifact = artifact_preds[i, 0].cpu().numpy()
                filename = filenames[i]
                
                # Refine lesion: remove overlapping with artifact
                # If there's ANY overlap, remove that part
                refined_lesion = pred_lesion.copy()
                refined_lesion[pred_artifact > 0] = 0  # Remove overlapping pixels
                
                # Calculate metrics BEFORE refinement
                dice_before = calculate_dice(mask_gt, pred_lesion)
                iou_before = calculate_iou(mask_gt, pred_lesion)
                
                # Calculate metrics AFTER refinement
                dice_after = calculate_dice(mask_gt, refined_lesion)
                iou_after = calculate_iou(mask_gt, refined_lesion)
                
                all_dice_before.append(dice_before)
                all_iou_before.append(iou_before)
                all_dice_after.append(dice_after)
                all_iou_after.append(iou_after)
                all_filenames_list.append(filename)
                
                # Save visualization
                save_combo_image(
                    img_slice,
                    mask_gt,
                    pred_lesion,
                    pred_artifact,
                    refined_lesion,
                    output_dir / f"{filename}.png",
                    dice_before=dice_before,
                    iou_before=iou_before,
                    dice_after=dice_after,
                    iou_after=iou_after
                )
    
    print(f"\n✅ All results saved to: {output_dir}")
    
    # Generate and save summary
    print("\n" + "="*60)
    print("📊 Generating Summary Report")
    print("="*60)
    
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    generate_combo_summary(
        all_filenames_list,
        all_dice_before,
        all_iou_before,
        all_dice_after,
        all_iou_after,
        summary_dir
    )
    
    print("="*60 + "\n")


# ==================== Metrics Calculation ====================

def calculate_dice(mask_gt, pred_mask, smooth=1e-6):
    """Calculate Dice coefficient"""
    mask_gt_flat = mask_gt.flatten()
    pred_mask_flat = pred_mask.flatten()
    
    intersection = np.sum(mask_gt_flat * pred_mask_flat)
    union = np.sum(mask_gt_flat) + np.sum(pred_mask_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def calculate_iou(mask_gt, pred_mask, smooth=1e-6):
    """Calculate Intersection over Union (IoU)"""
    mask_gt_flat = mask_gt.flatten()
    pred_mask_flat = pred_mask.flatten()
    
    intersection = np.sum(mask_gt_flat * pred_mask_flat)
    union = np.sum(mask_gt_flat) + np.sum(pred_mask_flat) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


# ==================== Visualization ====================

def save_combo_image(img_slice, mask_gt, pred_lesion, pred_artifact, refined_lesion, save_path,
                      dice_before=None, iou_before=None, dice_after=None, iou_after=None):
    """
    Save visualization with 5 panels showing combo results
    
    Args:
        img_slice: Original image
        mask_gt: Ground truth mask
        pred_lesion: Predicted lesion (before refinement)
        pred_artifact: Predicted artifact
        refined_lesion: Refined lesion (after removing artifact overlap)
        save_path: Path to save
        dice_before, iou_before: Metrics before refinement
        dice_after, iou_after: Metrics after refinement
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # 1. Original Image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Ground Truth (Black and White)
    axes[1].imshow(mask_gt, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Predicted Lesion (Green overlay)
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(pred_lesion, cmap='Greens', alpha=0.5)
    axes[2].set_title('Predicted Lesion', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add metrics before refinement
    if dice_before is not None and iou_before is not None:
        metrics_text = f'Dice: {dice_before:.4f}\nIoU: {iou_before:.4f}'
        axes[2].text(0.5, -0.05, metrics_text,
                    transform=axes[2].transAxes,
                    ha='center', va='top',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=2))
    
    # 4. Predicted Artifact (Red overlay)
    axes[3].imshow(img_slice, cmap='gray')
    axes[3].imshow(pred_artifact, cmap='Reds', alpha=0.5)
    axes[3].set_title('Predicted Artifact', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    # 5. Refined Lesion (Blue overlay)
    axes[4].imshow(img_slice, cmap='gray')
    axes[4].imshow(refined_lesion, cmap='Blues', alpha=0.5)
    axes[4].set_title('Refined Lesion\n(Artifact Removed)', fontsize=14, fontweight='bold')
    axes[4].axis('off')
    
    # Add metrics after refinement
    if dice_after is not None and iou_after is not None:
        metrics_text = f'Dice: {dice_after:.4f}\nIoU: {iou_after:.4f}'
        axes[4].text(0.5, -0.05, metrics_text,
                    transform=axes[4].transAxes,
                    ha='center', va='top',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', format=CONFIG['image_format'])
    plt.close()


# ==================== Summary Report ====================

def generate_combo_summary(filenames, dice_before, iou_before, dice_after, iou_after, summary_dir):
    """Generate summary report comparing before/after artifact removal"""
    
    # Create DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'dice_before': dice_before,
        'iou_before': iou_before,
        'dice_after': dice_after,
        'iou_after': iou_after,
        'dice_improvement': np.array(dice_after) - np.array(dice_before),
        'iou_improvement': np.array(iou_after) - np.array(iou_before)
    })
    
    # Save detailed CSV
    csv_path = summary_dir / 'detailed_metrics_combo.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Detailed metrics saved to: {csv_path}")
    
    # Calculate statistics
    stats_data = {
        'metric': ['Dice Before', 'Dice After', 'IoU Before', 'IoU After'],
        'mean': [
            np.mean(dice_before),
            np.mean(dice_after),
            np.mean(iou_before),
            np.mean(iou_after)
        ],
        'std': [
            np.std(dice_before),
            np.std(dice_after),
            np.std(iou_before),
            np.std(iou_after)
        ],
        'median': [
            np.median(dice_before),
            np.median(dice_after),
            np.median(iou_before),
            np.median(iou_after)
        ],
        'min': [
            np.min(dice_before),
            np.min(dice_after),
            np.min(iou_before),
            np.min(iou_after)
        ],
        'max': [
            np.max(dice_before),
            np.max(dice_after),
            np.max(iou_before),
            np.max(iou_after)
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    stats_csv_path = summary_dir / 'statistics_combo.csv'
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"✅ Statistics saved to: {stats_csv_path}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Dice comparison
    x = np.arange(len(filenames))
    axes[0, 0].plot(x, dice_before, 'o-', label='Before', alpha=0.6, color='green')
    axes[0, 0].plot(x, dice_after, 's-', label='After', alpha=0.6, color='blue')
    axes[0, 0].set_xlabel('Sample Index', fontweight='bold')
    axes[0, 0].set_ylabel('Dice Score', fontweight='bold')
    axes[0, 0].set_title('Dice Score: Before vs After Artifact Removal', fontweight='bold', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # IoU comparison
    axes[0, 1].plot(x, iou_before, 'o-', label='Before', alpha=0.6, color='green')
    axes[0, 1].plot(x, iou_after, 's-', label='After', alpha=0.6, color='blue')
    axes[0, 1].set_xlabel('Sample Index', fontweight='bold')
    axes[0, 1].set_ylabel('IoU Score', fontweight='bold')
    axes[0, 1].set_title('IoU Score: Before vs After Artifact Removal', fontweight='bold', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Dice improvement distribution
    dice_improvement = np.array(dice_after) - np.array(dice_before)
    axes[1, 0].hist(dice_improvement, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    axes[1, 0].set_xlabel('Dice Improvement', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title(f'Dice Improvement Distribution\nMean: {np.mean(dice_improvement):.4f}', 
                        fontweight='bold', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # IoU improvement distribution
    iou_improvement = np.array(iou_after) - np.array(iou_before)
    axes[1, 1].hist(iou_improvement, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    axes[1, 1].set_xlabel('IoU Improvement', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title(f'IoU Improvement Distribution\nMean: {np.mean(iou_improvement):.4f}', 
                        fontweight='bold', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = summary_dir / 'comparison_plot.png'
    plt.savefig(plot_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plot saved to: {plot_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("📈 COMBO SUMMARY STATISTICS")
    print("="*60)
    print(f"Total images: {len(filenames)}")
    print(f"\nDice Score:")
    print(f"  Before: {np.mean(dice_before):.4f} ± {np.std(dice_before):.4f}")
    print(f"  After:  {np.mean(dice_after):.4f} ± {np.std(dice_after):.4f}")
    print(f"  Improvement: {np.mean(dice_improvement):.4f}")
    print(f"\nIoU Score:")
    print(f"  Before: {np.mean(iou_before):.4f} ± {np.std(iou_before):.4f}")
    print(f"  After:  {np.mean(iou_after):.4f} ± {np.std(iou_after):.4f}")
    print(f"  Improvement: {np.mean(iou_improvement):.4f}")
    print("="*60)


# ==================== Main ====================

def main():
    """Main testing function"""
    
    print("="*60)
    print("DWI COMBO - Test with Lesion + Artifact Models")
    print("="*60)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(CONFIG['output_base'])
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🖼️  Image format: {CONFIG['image_format'].upper()}")
    print(f"📊 Resolution: {CONFIG['dpi']} DPI")
    print(f"🎯 Lesion model: {CONFIG['lesion_model_path']}")
    print(f"🎯 Artifact model: {CONFIG['artifact_model_path']}")
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
    
    # Run combo testing
    test_combo(
        lesion_model_path=CONFIG['lesion_model_path'],
        artifact_model_path=CONFIG['artifact_model_path'],
        dataloader=dataloader,
        output_dir=output_dir,
        device=CONFIG['device']
    )


if __name__ == '__main__':
    main()
