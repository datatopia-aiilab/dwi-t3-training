"""
Augmentation Visualization Tool
แสดงผลการ augmentation ต่างๆ ที่จะใช้ในการ training
สุ่มภาพมา 5 รูป และแสดง augmentation แต่ละแบบ

Usage:
    python visualize_augmentations.py
    
Output:
    - augmentation_results/
        - overview_augmentations.png (รวมทุก augmentation)
        - sample_001/
            - original.png
            - hflip.png
            - vflip.png
            - rotate.png
            - gamma.png
            - elastic.png
            - grid_distortion.png
            - brightness_contrast.png
            - all_augmentations.png
"""

import os
import sys
import glob
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import albumentations as A

# Import config
import config


# ==================== Configuration ====================

AUG_CONFIG = {
    'num_samples': 5,            # จำนวนรูปที่สุ่ม (ลดเหลือ 5 เพราะมี augmentation เยอะ)
    'output_dir': 'augmentation_results',
    'dpi': 150,
    'figsize_grid': (24, 20),
    'cmap': 'gray',
    'random_seed': 42,
    
    # Augmentation Parameters (ตามที่แนะนำ)
    'hflip_prob': 0.5,
    'vflip_prob': 0.3,
    'rotate_limit': 15,
    'rotate_prob': 0.5,
    'gamma_limit': (0.7, 1.5),
    'gamma_prob': 0.5,
    'elastic_alpha': 50,
    'elastic_sigma': 5,
    'elastic_prob': 0.3,
    'grid_prob': 0.2,
    'brightness_limit': 0.2,
    'contrast_limit': 0.2,
    'brightness_prob': 0.5,
}


# ==================== Individual Augmentations ====================

def create_augmentations():
    """สร้าง augmentation แต่ละแบบแยกกัน"""
    
    augmentations = {
        'Original': None,
        
        'Horizontal Flip': A.Compose([
            A.HorizontalFlip(p=1.0)
        ]),
        
        'Vertical Flip': A.Compose([
            A.VerticalFlip(p=1.0)
        ]),
        
        'Rotation ±15°': A.Compose([
            A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT)
        ]),
        
        'Gamma Correction': A.Compose([
            A.RandomGamma(gamma_limit=(70, 150), p=1.0)  # (0.7, 1.5) * 100
        ]),
        
        'Elastic Transform': A.Compose([
            A.ElasticTransform(
                alpha=AUG_CONFIG['elastic_alpha'],
                sigma=AUG_CONFIG['elastic_sigma'],
                p=1.0,
                border_mode=cv2.BORDER_CONSTANT
            )
        ]),
        
        'Grid Distortion': A.Compose([
            A.GridDistortion(p=1.0, border_mode=cv2.BORDER_CONSTANT)
        ]),
        
        'Brightness/Contrast': A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=AUG_CONFIG['brightness_limit'],
                contrast_limit=AUG_CONFIG['contrast_limit'],
                p=1.0
            )
        ]),
    }
    
    return augmentations


def apply_augmentation(image, mask, transform):
    """
    Apply augmentation to image and mask
    
    Args:
        image: numpy array (H, W)
        mask: numpy array (H, W)
        transform: albumentations transform or None
    
    Returns:
        aug_image, aug_mask
    """
    if transform is None:
        return image.copy(), mask.copy()
    
    # Apply transform
    augmented = transform(image=image, mask=mask)
    return augmented['image'], augmented['mask']


# ==================== Visualization ====================

def visualize_single_sample(image, mask, sample_name, save_dir):
    """
    Visualize all augmentations for a single sample
    
    Args:
        image: Original image (H, W)
        mask: Original mask (H, W)
        sample_name: Name for saving
        save_dir: Directory to save visualizations
    """
    augmentations = create_augmentations()
    
    # Create figure with all augmentations
    num_augs = len(augmentations)
    cols = 4
    rows = (num_augs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten() if num_augs > 1 else [axes]
    
    for idx, (aug_name, transform) in enumerate(augmentations.items()):
        if idx >= len(axes):
            break
        
        # Apply augmentation
        aug_img, aug_mask = apply_augmentation(image, mask, transform)
        
        # Plot image with mask overlay
        axes[idx].imshow(aug_img, cmap=AUG_CONFIG['cmap'])
        axes[idx].imshow(aug_mask, cmap='Reds', alpha=0.3)
        axes[idx].set_title(aug_name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        
        # Add statistics
        stats_text = f"Mean: {aug_img.mean():.2f}\nStd: {aug_img.std():.2f}"
        axes[idx].text(0.02, 0.98, stats_text,
                      transform=axes[idx].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save individual augmentation
        fig_single, ax_single = plt.subplots(1, 2, figsize=(12, 6))
        
        # Image only
        ax_single[0].imshow(aug_img, cmap=AUG_CONFIG['cmap'])
        ax_single[0].set_title(f'{aug_name} - Image', fontsize=12, fontweight='bold')
        ax_single[0].axis('off')
        
        # Image + Mask overlay
        ax_single[1].imshow(aug_img, cmap=AUG_CONFIG['cmap'])
        ax_single[1].imshow(aug_mask, cmap='Reds', alpha=0.4)
        ax_single[1].set_title(f'{aug_name} - With Mask', fontsize=12, fontweight='bold')
        ax_single[1].axis('off')
        
        plt.tight_layout()
        save_path = save_dir / f"{aug_name.lower().replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(save_path, dpi=AUG_CONFIG['dpi'], bbox_inches='tight')
        plt.close(fig_single)
    
    # Hide unused subplots
    for idx in range(num_augs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'All Augmentations: {sample_name}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save combined view
    save_path = save_dir / 'all_augmentations.png'
    plt.savefig(save_path, dpi=AUG_CONFIG['dpi'], bbox_inches='tight')
    plt.close()


def create_overview_grid(all_samples, save_path):
    """
    Create overview grid showing original + key augmentations for all samples
    
    Args:
        all_samples: List of (image, mask, name) tuples
        save_path: Path to save the grid
    """
    # Key augmentations to show in overview
    key_augs = {
        'Original': None,
        'H-Flip': A.HorizontalFlip(p=1.0),
        'Rotate': A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT),
        'Gamma': A.RandomGamma(gamma_limit=(70, 150), p=1.0),
        'Elastic': A.ElasticTransform(alpha=50, sigma=5, p=1.0, border_mode=cv2.BORDER_CONSTANT),
        'Bright/Cont': A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    }
    
    num_samples = len(all_samples)
    num_augs = len(key_augs)
    
    # Create figure
    fig = plt.figure(figsize=AUG_CONFIG['figsize_grid'])
    gs = GridSpec(num_samples, num_augs, figure=fig, hspace=0.2, wspace=0.15)
    
    for row_idx, (image, mask, sample_name) in enumerate(all_samples):
        for col_idx, (aug_name, transform) in enumerate(key_augs.items()):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Apply augmentation
            if transform is None:
                aug_img, aug_mask = image.copy(), mask.copy()
            else:
                aug_img, aug_mask = apply_augmentation(image, mask, A.Compose([transform]))
            
            # Plot
            ax.imshow(aug_img, cmap=AUG_CONFIG['cmap'])
            ax.imshow(aug_mask, cmap='Reds', alpha=0.25)
            ax.axis('off')
            
            # Add column title for first row
            if row_idx == 0:
                ax.set_title(aug_name, fontsize=11, fontweight='bold')
            
            # Add sample name on first column
            if col_idx == 0:
                ax.text(-0.15, 0.5, sample_name,
                       transform=ax.transAxes,
                       fontsize=10, fontweight='bold',
                       verticalalignment='center',
                       horizontalalignment='right',
                       rotation=0)
    
    plt.suptitle('Augmentation Overview - Key Techniques',
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=AUG_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✅ Overview grid saved to: {save_path}")


# ==================== Main Function ====================

def main():
    """Main visualization function"""
    
    print("\n" + "="*70)
    print("🎨 AUGMENTATION VISUALIZATION TOOL")
    print("="*70)
    print(f"Output directory: {AUG_CONFIG['output_dir']}")
    print(f"Number of samples: {AUG_CONFIG['num_samples']}")
    print(f"Augmentations: 8 types")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path(AUG_CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Find all image files
    raw_data_path = Path(config.RAW_DATA_PATH)
    images_dir = raw_data_path / "images"
    masks_dir = raw_data_path / "masks"
    
    # Try .npy files first, then .nii.gz
    image_files = sorted(glob.glob(str(images_dir / "*.npy")))
    if len(image_files) == 0:
        image_files = sorted(glob.glob(str(images_dir / "*.nii.gz")))
        file_type = "nii.gz"
    else:
        file_type = "npy"
    
    print(f"Found {len(image_files)} {file_type} files in {images_dir}")
    
    if len(image_files) == 0:
        print("❌ No image files found!")
        return
    
    # Randomly sample files
    random.seed(AUG_CONFIG['random_seed'])
    sampled_files = random.sample(image_files, min(AUG_CONFIG['num_samples'], len(image_files)))
    
    print(f"\nRandomly selected {len(sampled_files)} files for visualization\n")
    
    # Process each sample
    all_samples = []
    
    for idx, img_path in enumerate(tqdm(sampled_files, desc="Processing samples")):
        # Load image
        img_name = Path(img_path).stem
        sample_name = f"Sample_{idx+1:02d}"
        
        # Get corresponding mask
        if file_type == "npy":
            mask_path = masks_dir / f"{img_name}.npy"
        else:
            mask_path = masks_dir / f"{img_name}.nii.gz"
        
        if not mask_path.exists():
            print(f"⚠️  Mask not found for {img_name}, skipping...")
            continue
        
        # Load data
        if file_type == "npy":
            img_data = np.load(img_path)
            mask_data = np.load(str(mask_path))
        else:
            img_nii = nib.load(img_path)
            mask_nii = nib.load(str(mask_path))
            img_data = img_nii.get_fdata()
            mask_data = mask_nii.get_fdata()
        
        # Handle 2D/3D data
        if img_data.ndim == 2:
            img_slice = img_data
            mask_slice = mask_data
        elif img_data.ndim == 3:
            # Take middle slice
            if img_data.shape[0] == 3:
                # Already 2.5D format (3, H, W)
                img_slice = img_data[1, :, :]  # Middle channel
                mask_slice = mask_data  # Assume mask is 2D
            else:
                # 3D volume (H, W, D)
                mid_slice = img_data.shape[2] // 2
                img_slice = img_data[:, :, mid_slice]
                mask_slice = mask_data[:, :, mid_slice]
        else:
            print(f"⚠️  Unexpected shape: {img_data.shape}, skipping...")
            continue
        
        # Resize
        img_resized = cv2.resize(img_slice, config.IMAGE_SIZE)
        mask_resized = cv2.resize(mask_slice, config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to 0-1 for better visualization
        img_min, img_max = img_resized.min(), img_resized.max()
        img_normalized = (img_resized - img_min) / (img_max - img_min + 1e-8)
        
        # Binarize mask
        mask_binary = (mask_resized > 0.5).astype(np.float32)
        
        # Store for overview
        all_samples.append((img_normalized, mask_binary, sample_name))
        
        # Save individual sample visualization
        sample_dir = output_dir / f"sample_{idx+1:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        visualize_single_sample(img_normalized, mask_binary, sample_name, sample_dir)
    
    # Create overview grid
    if len(all_samples) > 0:
        overview_path = output_dir / 'overview_augmentations.png'
        create_overview_grid(all_samples, overview_path)
    
    # Save configuration
    import json
    config_path = output_dir / 'augmentation_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(all_samples),
            'random_seed': AUG_CONFIG['random_seed'],
            'augmentations': {
                'horizontal_flip': {'prob': AUG_CONFIG['hflip_prob']},
                'vertical_flip': {'prob': AUG_CONFIG['vflip_prob']},
                'rotation': {'limit': AUG_CONFIG['rotate_limit'], 'prob': AUG_CONFIG['rotate_prob']},
                'gamma': {'limit': AUG_CONFIG['gamma_limit'], 'prob': AUG_CONFIG['gamma_prob']},
                'elastic': {'alpha': AUG_CONFIG['elastic_alpha'], 'sigma': AUG_CONFIG['elastic_sigma'], 'prob': AUG_CONFIG['elastic_prob']},
                'grid_distortion': {'prob': AUG_CONFIG['grid_prob']},
                'brightness_contrast': {'brightness_limit': AUG_CONFIG['brightness_limit'], 'contrast_limit': AUG_CONFIG['contrast_limit'], 'prob': AUG_CONFIG['brightness_prob']},
            },
            'sample_files': [Path(f).name for f in sampled_files]
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ AUGMENTATION VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    print(f"  • overview_augmentations.png - Overview of key augmentations")
    print(f"  • sample_001/ to sample_{len(all_samples):03d}/ - Detailed augmentations")
    print(f"  • augmentation_config.json - Configuration details")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
