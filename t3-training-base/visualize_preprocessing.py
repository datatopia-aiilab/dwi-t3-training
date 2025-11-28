"""
Preprocessing Visualization Tool
แสดงผลการ preprocessing ในแต่ละขั้นตอน (N4, CLAHE, Normalization, etc.)
สุ่มภาพจาก 1_data_raw มา 10 รูป และแสดงผลแต่ละ step

Usage:
    python visualize_preprocessing.py
    
Output:
    - visualization_results/
        - overview_grid.png (รวม 10 รูป)
        - sample_001/ (แยกรายภาพ)
            - step1_original.png
            - step2_n4_corrected.png
            - step3_clahe.png
            - step4_normalized.png
            - comparison.png (4 steps ในภาพเดียว)
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

# Import config
import config

# Optional: N4 Bias Correction (ถ้าติดตั้ง SimpleITK แล้ว)
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    print("⚠️  SimpleITK not installed. N4 Bias Correction will be skipped.")
    print("   Install with: pip install SimpleITK")


# ==================== Configuration ====================

VIS_CONFIG = {
    'num_samples': 10,           # จำนวนรูปที่สุ่ม
    'output_dir': 'visualization_results',
    'dpi': 150,                  # ความละเอียดของภาพ
    'figsize_single': (20, 5),  # ขนาดภาพสำหรับแต่ละ sample
    'figsize_grid': (20, 25),   # ขนาดภาพสำหรับ grid overview
    'cmap': 'gray',             # Colormap
    'random_seed': 42,          # Random seed
    
    # N4 Parameters (ถ้าใช้)
    'n4_enabled': True,
    'n4_shrink_factor': 4,
    'n4_num_iterations': [50, 50, 50, 50],
    
    # CLAHE Parameters
    'clahe_enabled': False,      # ปกติไม่ใช้ถ้ามี N4 แล้ว
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    
    # Normalization
    'norm_method': 'zscore',    # 'zscore' or 'minmax'
}


# ==================== N4 Bias Correction ====================

def apply_n4_bias_correction(img_data, shrink_factor=4, num_iterations=[50, 50, 50, 50]):
    """
    Apply N4 Bias Field Correction using SimpleITK
    
    Args:
        img_data: numpy array (H, W) or (H, W, D)
        shrink_factor: Downsampling factor for speed (1=no downsampling, 4=recommended)
        num_iterations: List of iterations per level
    
    Returns:
        corrected_img: numpy array with same shape as input
    """
    if not HAS_SITK:
        print("⚠️  SimpleITK not available, returning original image")
        return img_data
    
    # Handle 2D or 3D
    is_2d = (img_data.ndim == 2)
    if is_2d:
        img_data = img_data[:, :, np.newaxis]
    
    # Convert to SimpleITK image
    img_sitk = sitk.GetImageFromArray(img_data.astype(np.float32))
    
    # Create mask (entire image)
    mask_sitk = sitk.OtsuThreshold(img_sitk, 0, 1, 200)
    
    # Setup N4 corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(num_iterations)
    corrector.SetConvergenceThreshold(0.001)
    
    if shrink_factor > 1:
        # Shrink image for faster processing
        img_shrunk = sitk.Shrink(img_sitk, [shrink_factor] * img_sitk.GetDimension())
        mask_shrunk = sitk.Shrink(mask_sitk, [shrink_factor] * mask_sitk.GetDimension())
        
        # Run N4 on shrunk image
        corrected_shrunk = corrector.Execute(img_shrunk, mask_shrunk)
        
        # Get bias field and resample to original size
        log_bias_field = corrector.GetLogBiasFieldAsImage(img_sitk)
        corrected_sitk = img_sitk / sitk.Exp(log_bias_field)
    else:
        # Run N4 on full resolution
        corrected_sitk = corrector.Execute(img_sitk, mask_sitk)
    
    # Convert back to numpy
    corrected_img = sitk.GetArrayFromImage(corrected_sitk)
    
    # Return to original shape
    if is_2d:
        corrected_img = corrected_img[:, :, 0]
    
    return corrected_img


# ==================== CLAHE ====================

def apply_clahe(img_data, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization
    
    Args:
        img_data: numpy array (H, W)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        clahe_img: numpy array with CLAHE applied
    """
    # Normalize to 0-255 for CLAHE
    img_min, img_max = img_data.min(), img_data.max()
    img_norm = ((img_data - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_norm)
    
    # Convert back to float
    img_clahe = img_clahe.astype(np.float32) / 255.0 * (img_max - img_min) + img_min
    
    return img_clahe


# ==================== Normalization ====================

def normalize_image(img_data, method='zscore'):
    """
    Normalize image
    
    Args:
        img_data: numpy array (H, W)
        method: 'zscore' or 'minmax'
    
    Returns:
        normalized_img: numpy array
    """
    if method == 'zscore':
        # Z-score normalization
        mean = img_data.mean()
        std = img_data.std()
        normalized = (img_data - mean) / (std + 1e-8)
    elif method == 'minmax':
        # Min-Max normalization to [0, 1]
        img_min = img_data.min()
        img_max = img_data.max()
        normalized = (img_data - img_min) / (img_max - img_min + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


# ==================== Preprocessing Pipeline ====================

def preprocess_image(img_slice, apply_n4=True, apply_clahe=False, norm_method='zscore'):
    """
    Full preprocessing pipeline with intermediate results
    
    Args:
        img_slice: numpy array (H, W)
        apply_n4: Whether to apply N4 bias correction
        apply_clahe: Whether to apply CLAHE
        norm_method: Normalization method
    
    Returns:
        steps: Dictionary with keys 'original', 'n4', 'clahe', 'normalized'
    """
    steps = {}
    
    # Step 1: Original (resized)
    img_resized = cv2.resize(img_slice, config.IMAGE_SIZE)
    steps['original'] = img_resized.copy()
    
    # Step 2: N4 Bias Correction
    if apply_n4 and HAS_SITK:
        img_n4 = apply_n4_bias_correction(
            img_resized,
            shrink_factor=VIS_CONFIG['n4_shrink_factor'],
            num_iterations=VIS_CONFIG['n4_num_iterations']
        )
        steps['n4'] = img_n4.copy()
    else:
        steps['n4'] = img_resized.copy()
    
    # Step 3: CLAHE (optional)
    if apply_clahe:
        img_clahe = apply_clahe(
            steps['n4'],
            clip_limit=VIS_CONFIG['clahe_clip_limit'],
            tile_grid_size=VIS_CONFIG['clahe_tile_grid_size']
        )
        steps['clahe'] = img_clahe.copy()
    else:
        steps['clahe'] = steps['n4'].copy()
    
    # Step 4: Normalization
    img_normalized = normalize_image(steps['clahe'], method=norm_method)
    steps['normalized'] = img_normalized.copy()
    
    return steps


# ==================== Visualization ====================

def visualize_single_sample(steps, sample_name, save_dir):
    """
    Visualize all preprocessing steps for a single sample
    
    Args:
        steps: Dictionary with preprocessing steps
        sample_name: Name for saving
        save_dir: Directory to save visualizations
    """
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=VIS_CONFIG['figsize_single'])
    
    step_names = ['original', 'n4', 'clahe', 'normalized']
    titles = [
        'Step 1: Original\n(Resized)',
        'Step 2: N4 Bias Correction\n(Intensity Homogenization)',
        'Step 3: CLAHE\n(Contrast Enhancement)',
        'Step 4: Normalized\n(Z-score)'
    ]
    
    for idx, (step_name, title) in enumerate(zip(step_names, titles)):
        img = steps[step_name]
        
        # Plot
        im = axes[idx].imshow(img, cmap=VIS_CONFIG['cmap'])
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        
        # Add statistics
        stats_text = f"Mean: {img.mean():.3f}\nStd: {img.std():.3f}\nMin: {img.min():.3f}\nMax: {img.max():.3f}"
        axes[idx].text(0.02, 0.98, stats_text, 
                      transform=axes[idx].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Preprocessing Pipeline: {sample_name}', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    save_path = save_dir / 'comparison.png'
    plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    # Save individual steps
    for step_name in step_names:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        img = steps[step_name]
        im = ax.imshow(img, cmap=VIS_CONFIG['cmap'])
        ax.set_title(f'{sample_name} - {step_name}', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        save_path = save_dir / f'step_{step_names.index(step_name)+1}_{step_name}.png'
        plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
        plt.close()


def create_overview_grid(all_steps, sample_names, save_path):
    """
    Create overview grid showing all samples
    
    Args:
        all_steps: List of step dictionaries
        sample_names: List of sample names
        save_path: Path to save the grid
    """
    num_samples = len(all_steps)
    
    # Create figure
    fig = plt.figure(figsize=VIS_CONFIG['figsize_grid'])
    gs = GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.2)
    
    step_names = ['original', 'n4', 'clahe', 'normalized']
    
    # Column titles
    col_titles = ['Original', 'N4 Corrected', 'CLAHE', 'Normalized']
    
    for row_idx, (steps, sample_name) in enumerate(zip(all_steps, sample_names)):
        for col_idx, step_name in enumerate(step_names):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            img = steps[step_name]
            ax.imshow(img, cmap=VIS_CONFIG['cmap'])
            ax.axis('off')
            
            # Add title for first row
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, fontweight='bold')
            
            # Add sample name on first column
            if col_idx == 0:
                ax.text(-0.1, 0.5, sample_name, 
                       transform=ax.transAxes,
                       fontsize=10, fontweight='bold',
                       verticalalignment='center',
                       horizontalalignment='right',
                       rotation=0)
    
    plt.suptitle('Preprocessing Pipeline Overview (10 Random Samples)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✅ Overview grid saved to: {save_path}")


# ==================== Main Function ====================

def main():
    """Main visualization function"""
    
    print("\n" + "="*70)
    print("🎨 PREPROCESSING VISUALIZATION TOOL")
    print("="*70)
    print(f"Output directory: {VIS_CONFIG['output_dir']}")
    print(f"Number of samples: {VIS_CONFIG['num_samples']}")
    print(f"N4 Bias Correction: {'ENABLED ✓' if VIS_CONFIG['n4_enabled'] and HAS_SITK else 'DISABLED ✗'}")
    print(f"CLAHE: {'ENABLED ✓' if VIS_CONFIG['clahe_enabled'] else 'DISABLED ✗'}")
    print(f"Normalization: {VIS_CONFIG['norm_method']}")
    print("="*70 + "\n")
    
    # Create output directory
    output_dir = Path(VIS_CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Find all image files
    raw_data_path = Path(config.RAW_DATA_PATH)
    images_dir = raw_data_path / "images"
    
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
    random.seed(VIS_CONFIG['random_seed'])
    sampled_files = random.sample(image_files, min(VIS_CONFIG['num_samples'], len(image_files)))
    
    print(f"\nRandomly selected {len(sampled_files)} files for visualization\n")
    
    # Process each sample
    all_steps = []
    sample_names = []
    
    for idx, img_path in enumerate(tqdm(sampled_files, desc="Processing samples")):
        # Load image
        img_name = Path(img_path).stem
        sample_name = f"Sample_{idx+1:02d}_{img_name[:20]}"
        sample_names.append(sample_name)
        
        # Load data
        if file_type == "npy":
            img_data = np.load(img_path)
        else:
            img_nii = nib.load(img_path)
            img_data = img_nii.get_fdata()
        
        # Handle 2D/3D data
        if img_data.ndim == 2:
            img_slice = img_data
        elif img_data.ndim == 3:
            # Take middle slice
            if img_data.shape[0] == 3:
                # Already 2.5D format (3, H, W)
                img_slice = img_data[1, :, :]  # Middle channel
            else:
                # 3D volume (H, W, D)
                mid_slice = img_data.shape[2] // 2
                img_slice = img_data[:, :, mid_slice]
        else:
            print(f"⚠️  Unexpected shape: {img_data.shape}, skipping...")
            continue
        
        # Run preprocessing pipeline
        steps = preprocess_image(
            img_slice,
            apply_n4=VIS_CONFIG['n4_enabled'],
            apply_clahe=VIS_CONFIG['clahe_enabled'],
            norm_method=VIS_CONFIG['norm_method']
        )
        
        all_steps.append(steps)
        
        # Save individual sample visualization
        sample_dir = output_dir / f"sample_{idx+1:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        visualize_single_sample(steps, sample_name, sample_dir)
    
    # Create overview grid
    if len(all_steps) > 0:
        overview_path = output_dir / 'overview_grid.png'
        create_overview_grid(all_steps, sample_names, overview_path)
    
    # Save configuration
    import json
    config_path = output_dir / 'visualization_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(all_steps),
            'random_seed': VIS_CONFIG['random_seed'],
            'n4_enabled': VIS_CONFIG['n4_enabled'] and HAS_SITK,
            'clahe_enabled': VIS_CONFIG['clahe_enabled'],
            'normalization': VIS_CONFIG['norm_method'],
            'image_size': config.IMAGE_SIZE,
            'n4_shrink_factor': VIS_CONFIG['n4_shrink_factor'],
            'sample_files': [Path(f).name for f in sampled_files]
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    print(f"  • overview_grid.png - Overview of all {len(all_steps)} samples")
    print(f"  • sample_001/ to sample_{len(all_steps):03d}/ - Individual visualizations")
    print(f"  • visualization_config.json - Configuration details")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
