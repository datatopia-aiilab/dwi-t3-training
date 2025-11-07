"""
Debug Script: Check Preprocessing Quality
Verifies that preprocessing and .npy files are correct
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import config


def check_npy_files_integrity():
    """Check if .npy files are valid and contain expected data"""
    
    print("\n" + "="*70)
    print("🔍 CHECKING PREPROCESSED DATA INTEGRITY")
    print("="*70)
    
    issues = []
    
    # Check all splits
    splits = {
        'train': (config.PROCESSED_TRAIN_IMG, config.PROCESSED_TRAIN_MASK),
        'val': (config.PROCESSED_VAL_IMG, config.PROCESSED_VAL_MASK),
        'test': (config.PROCESSED_TEST_IMG, config.PROCESSED_TEST_MASK)
    }
    
    for split_name, (img_dir, mask_dir) in splits.items():
        print(f"\n📂 Checking {split_name.upper()} set...")
        
        if not img_dir.exists():
            print(f"   ❌ Directory not found: {img_dir}")
            issues.append(f"{split_name}: Image directory missing")
            continue
        
        img_files = list(img_dir.glob('*.npy'))
        mask_files = list(mask_dir.glob('*.npy'))
        
        print(f"   Found {len(img_files)} images, {len(mask_files)} masks")
        
        if len(img_files) != len(mask_files):
            issues.append(f"{split_name}: Mismatch in image/mask count")
            print(f"   ⚠️  Image and mask counts don't match!")
        
        # Sample statistics
        img_stats = {
            'min': [], 'max': [], 'mean': [], 'std': [],
            'has_nan': 0, 'has_inf': 0, 'negative': 0
        }
        
        mask_stats = {
            'unique_values': set(),
            'min': [], 'max': [], 'mean': [],
            'lesion_pixels': []
        }
        
        print(f"   Analyzing samples...")
        sample_size = min(50, len(img_files))  # Check first 50 files
        
        for img_file in tqdm(img_files[:sample_size], desc=f"   {split_name}"):
            try:
                # Load image
                img = np.load(img_file)
                
                # Check shape
                if img.ndim != 2:
                    issues.append(f"{split_name}/{img_file.name}: Wrong dimensions {img.shape}")
                
                # Check values
                img_stats['min'].append(img.min())
                img_stats['max'].append(img.max())
                img_stats['mean'].append(img.mean())
                img_stats['std'].append(img.std())
                
                if np.isnan(img).any():
                    img_stats['has_nan'] += 1
                    issues.append(f"{split_name}/{img_file.name}: Contains NaN")
                
                if np.isinf(img).any():
                    img_stats['has_inf'] += 1
                    issues.append(f"{split_name}/{img_file.name}: Contains Inf")
                
                if (img < 0).any():
                    img_stats['negative'] += 1
                
                # Load corresponding mask
                mask_file = mask_dir / img_file.name
                if mask_file.exists():
                    mask = np.load(mask_file)
                    
                    # Check mask properties
                    mask_stats['unique_values'].update(np.unique(mask))
                    mask_stats['min'].append(mask.min())
                    mask_stats['max'].append(mask.max())
                    mask_stats['mean'].append(mask.mean())
                    mask_stats['lesion_pixels'].append((mask > 0).sum())
                
            except Exception as e:
                issues.append(f"{split_name}/{img_file.name}: Load error - {e}")
        
        # Print statistics
        print(f"\n   📊 {split_name.upper()} Image Statistics:")
        if img_stats['min']:
            print(f"      Min:    {np.min(img_stats['min']):.4f} to {np.max(img_stats['min']):.4f}")
            print(f"      Max:    {np.min(img_stats['max']):.4f} to {np.max(img_stats['max']):.4f}")
            print(f"      Mean:   {np.mean(img_stats['mean']):.4f} ± {np.std(img_stats['mean']):.4f}")
            print(f"      Std:    {np.mean(img_stats['std']):.4f} ± {np.std(img_stats['std']):.4f}")
            print(f"      Has NaN: {img_stats['has_nan']}/{sample_size} files")
            print(f"      Has Inf: {img_stats['has_inf']}/{sample_size} files")
            print(f"      Has Negative: {img_stats['negative']}/{sample_size} files")
            
            # Check if normalization looks correct
            mean_of_means = np.mean(img_stats['mean'])
            if abs(mean_of_means) < 0.1:
                print(f"      ✅ Normalization looks good (mean ≈ 0)")
            else:
                print(f"      ⚠️  Mean is far from 0, normalization might be incorrect")
                issues.append(f"{split_name}: Suspicious normalization (mean={mean_of_means:.4f})")
        
        print(f"\n   📊 {split_name.upper()} Mask Statistics:")
        if mask_stats['min']:
            print(f"      Unique values: {sorted(mask_stats['unique_values'])}")
            print(f"      Min: {np.min(mask_stats['min']):.4f}")
            print(f"      Max: {np.max(mask_stats['max']):.4f}")
            print(f"      Avg lesion pixels: {np.mean(mask_stats['lesion_pixels']):.0f}")
            
            # Check if mask is binary
            if mask_stats['unique_values'] == {0.0, 1.0} or mask_stats['unique_values'] == {0.0}:
                print(f"      ✅ Mask is binary")
            else:
                print(f"      ⚠️  Mask has non-binary values: {mask_stats['unique_values']}")
                issues.append(f"{split_name}: Non-binary mask values")
    
    # Final report
    print("\n" + "="*70)
    if len(issues) == 0:
        print("✅ ALL CHECKS PASSED! Data looks good.")
    else:
        print(f"⚠️  FOUND {len(issues)} ISSUES:")
        for i, issue in enumerate(issues[:20], 1):  # Show first 20
            print(f"   {i}. {issue}")
        if len(issues) > 20:
            print(f"   ... and {len(issues)-20} more issues")
    print("="*70)
    
    return issues


def visualize_preprocessing_comparison():
    """Visualize original vs preprocessed images"""
    
    print("\n" + "="*70)
    print("🖼️  VISUALIZING PREPROCESSING COMPARISON")
    print("="*70)
    
    # Find a sample file
    img_dir = config.PROCESSED_TRAIN_IMG
    if not img_dir.exists():
        print("❌ No preprocessed data found")
        return
    
    img_files = list(img_dir.glob('*.npy'))
    if len(img_files) == 0:
        print("❌ No .npy files found")
        return
    
    # Load 4 random samples
    import random
    samples = random.sample(img_files, min(4, len(img_files)))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, img_file in enumerate(samples):
        # Load image and mask
        img = np.load(img_file)
        mask_file = config.PROCESSED_TRAIN_MASK / img_file.name
        mask = np.load(mask_file)
        
        # Plot image
        ax_img = axes[0, idx]
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(f"{img_file.stem}\nMin:{img.min():.2f} Max:{img.max():.2f}")
        ax_img.axis('off')
        
        # Plot mask
        ax_mask = axes[1, idx]
        ax_mask.imshow(mask, cmap='gray')
        ax_mask.set_title(f"Lesion pixels: {(mask>0).sum()}")
        ax_mask.axis('off')
    
    axes[0, 0].set_ylabel('Preprocessed Image', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Mask', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = config.RESULTS_DIR / 'preprocessing_check.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    plt.close()


def check_normalization_consistency():
    """Check if normalization is consistent across splits"""
    
    print("\n" + "="*70)
    print("📏 CHECKING NORMALIZATION CONSISTENCY")
    print("="*70)
    
    splits = {
        'train': config.PROCESSED_TRAIN_IMG,
        'val': config.PROCESSED_VAL_IMG,
        'test': config.PROCESSED_TEST_IMG
    }
    
    split_stats = {}
    
    for split_name, img_dir in splits.items():
        if not img_dir.exists():
            continue
        
        img_files = list(img_dir.glob('*.npy'))
        if len(img_files) == 0:
            continue
        
        print(f"\n📊 {split_name.upper()} set:")
        
        # Sample files
        sample_size = min(100, len(img_files))
        all_values = []
        
        for img_file in tqdm(img_files[:sample_size], desc=f"   Loading {split_name}"):
            img = np.load(img_file)
            all_values.append(img.flatten())
        
        all_values = np.concatenate(all_values)
        
        mean = all_values.mean()
        std = all_values.std()
        
        split_stats[split_name] = {'mean': mean, 'std': std}
        
        print(f"   Mean: {mean:.6f}")
        print(f"   Std:  {std:.6f}")
    
    # Compare splits
    print("\n" + "="*70)
    print("🔍 COMPARISON:")
    
    if len(split_stats) < 2:
        print("⚠️  Not enough splits to compare")
        return
    
    # Check if all splits have similar statistics
    means = [s['mean'] for s in split_stats.values()]
    stds = [s['std'] for s in split_stats.values()]
    
    mean_diff = max(means) - min(means)
    std_diff = max(stds) - min(stds)
    
    print(f"   Mean difference across splits: {mean_diff:.6f}")
    print(f"   Std difference across splits:  {std_diff:.6f}")
    
    if mean_diff < 0.1 and std_diff < 0.1:
        print("   ✅ Normalization is CONSISTENT across splits")
    else:
        print("   ⚠️  WARNING: Normalization INCONSISTENT!")
        print("   This could cause val/test performance issues!")
        print("\n   Recommendation: Re-run preprocessing with global statistics")
    
    print("="*70)


def main():
    """Run all checks"""
    
    print("\n" + "🔬"*35)
    print("DEBUG: PREPROCESSING & DATA QUALITY CHECK")
    print("🔬"*35)
    
    # Check 1: File integrity
    issues = check_npy_files_integrity()
    
    # Check 2: Normalization consistency
    check_normalization_consistency()
    
    # Check 3: Visualization
    try:
        visualize_preprocessing_comparison()
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
    
    print("\n" + "="*70)
    print("✅ DEBUG COMPLETE!")
    print("="*70)
    
    if len(issues) > 0:
        print(f"\n⚠️  Found {len(issues)} potential issues.")
        print("   Review the output above and consider re-preprocessing if needed.")
    else:
        print("\n✅ No issues detected. Data preprocessing looks good!")


if __name__ == "__main__":
    main()
