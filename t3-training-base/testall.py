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
            img_data = np.load(img_path)
            mask_data = np.load(str(mask_path))
        else:
            import nibabel as nib
            img_nii = nib.load(img_path)
            mask_nii = nib.load(str(mask_path))
            img_data = img_nii.get_fdata()
            mask_data = mask_nii.get_fdata()
        
        # Handle different data shapes
        if img_data.ndim == 2:
            img_data = img_data[:, :, np.newaxis]
            mask_data = mask_data[:, :, np.newaxis]
        
        # Process each slice
        for slice_idx in range(img_data.shape[2]):
            # Get current slice
            img_slice = img_data[:, :, slice_idx]
            mask_slice = mask_data[:, :, slice_idx]
            
            # Keep original mask for visualization (don't skip empty masks)
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
    
    return all_images, all_masks, all_filenames


# ==================== Testing ====================

def test_all(model, dataloader, device, output_dir):
    """
    Run inference on all data and save results
    """
    print("\n" + "="*60)
    print("🔬 Running Inference on All Data")
    print("="*60)
    
    model.eval()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
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
                
                # Save visualization
                save_prediction_image(
                    img_slice, 
                    mask_gt, 
                    pred_mask, 
                    output_dir / f"{filename}.png"
                )
    
    print(f"\n✅ All results saved to: {output_dir}")
    print("="*60 + "\n")


def save_prediction_image(img_slice, mask_gt, pred_mask, save_path):
    """
    Save high-quality PNG visualization with 3 panels: Original, Ground Truth, Prediction
    
    Args:
        img_slice: Normalized image slice (H, W)
        mask_gt: Ground truth binary mask (H, W)
        pred_mask: Predicted binary mask (H, W)
        save_path: Path to save the result
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth (Mask - Red overlay)
    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(mask_gt, cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction (Green overlay)
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(pred_mask, cmap='Greens', alpha=0.5)
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
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
