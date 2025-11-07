"""
Data Preprocessing Script for DWI Ischemic Stroke Segmentation
Performs: Data splitting, Resize, CLAHE, Normalization, Save as .npy
"""

import numpy as np
from pathlib import Path
from skimage import exposure
import cv2
from tqdm import tqdm
import json
import random
from collections import defaultdict

# Import config and utils
import config
from utils import parse_filename, build_slice_mapping, get_patient_statistics


def apply_clahe(image, clip_limit=0.03, tile_grid_size=None):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    This is CRUCIAL for enhancing faint lesions!
    
    Args:
        image: 2D numpy array (H, W)
        clip_limit: Clipping limit (0.01-0.05 recommended)
        tile_grid_size: Tuple (height, width) for grid size (None = auto)
    
    Returns:
        Enhanced image
    """
    # Ensure image is in [0, 1] range
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        image_normalized = (image - img_min) / (img_max - img_min)
    else:
        image_normalized = image
    
    # Apply CLAHE using skimage
    enhanced = exposure.equalize_adapthist(
        image_normalized,
        clip_limit=clip_limit,
        nbins=256
    )
    
    return enhanced.astype(np.float32)


def resize_image(image, target_size, is_mask=False):
    """
    Resize image or mask to target size
    
    Args:
        image: 2D numpy array
        target_size: Tuple (height, width)
        is_mask: If True, use nearest neighbor interpolation
    
    Returns:
        Resized image
    """
    if image.shape[:2] == target_size:
        return image
    
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    
    resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
    
    return resized


def split_data_by_slice(slice_mapping, test_size=48, train_val_ratio=0.80, random_seed=42):
    """
    Split data into train/val/test sets by SLICE (not by patient)
    
    New splitting strategy:
    - Test: Fixed number of slices (e.g., 48)
    - Remaining slices split into Train/Val (e.g., 80%/20%)
    
    Args:
        slice_mapping: Dictionary from build_slice_mapping()
        test_size: Number of slices for test set
        train_val_ratio: Ratio of train/(train+val) from remaining data
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]} with slice filenames
    """
    # Get all slice filenames
    all_slices = list(slice_mapping.keys())
    total_slices = len(all_slices)
    
    # Shuffle slices
    random.seed(random_seed)
    random.shuffle(all_slices)
    
    # Split: first test_size slices → test set
    test_slices = all_slices[:test_size]
    remaining_slices = all_slices[test_size:]
    
    # Split remaining into train/val
    remaining_count = len(remaining_slices)
    train_count = int(remaining_count * train_val_ratio)
    
    train_slices = remaining_slices[:train_count]
    val_slices = remaining_slices[train_count:]
    
    print(f"\n📊 Data Split Summary (Slice-Based):")
    print(f"   Total Slices: {total_slices}")
    print(f"   Test:  {len(test_slices)} slices ({len(test_slices)/total_slices*100:.1f}%)")
    print(f"   Train: {len(train_slices)} slices ({len(train_slices)/total_slices*100:.1f}%)")
    print(f"   Val:   {len(val_slices)} slices ({len(val_slices)/total_slices*100:.1f}%)")
    print(f"   Train+Val: {len(train_slices)+len(val_slices)} slices")
    
    return {
        'train': train_slices,
        'val': val_slices,
        'test': test_slices
    }


def split_data_by_patient(slice_mapping, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data into train/val/test sets by PATIENT (not by slice)
    
    This is important to avoid data leakage!
    
    Args:
        slice_mapping: Dictionary from build_slice_mapping()
        train_ratio, val_ratio, test_ratio: Split ratios
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]} with slice filenames
    """
    # Group slices by patient
    patient_slices = defaultdict(list)
    
    for filename, info in slice_mapping.items():
        patient_slices[info['patient']].append(filename)
    
    # Get list of unique patients
    patients = list(patient_slices.keys())
    num_patients = len(patients)
    
    # Shuffle patients
    random.seed(random_seed)
    random.shuffle(patients)
    
    # Calculate split indices
    train_end = int(num_patients * train_ratio)
    val_end = train_end + int(num_patients * val_ratio)
    
    # Split patients
    train_patients = patients[:train_end]
    val_patients = patients[train_end:val_end]
    test_patients = patients[val_end:]
    
    # Collect all slices for each split
    train_slices = []
    val_slices = []
    test_slices = []
    
    for patient in train_patients:
        train_slices.extend(patient_slices[patient])
    
    for patient in val_patients:
        val_slices.extend(patient_slices[patient])
    
    for patient in test_patients:
        test_slices.extend(patient_slices[patient])
    
    print(f"\n📊 Data Split Summary:")
    print(f"   Total Patients: {num_patients}")
    print(f"   Train: {len(train_patients)} patients, {len(train_slices)} slices")
    print(f"   Val:   {len(val_patients)} patients, {len(val_slices)} slices")
    print(f"   Test:  {len(test_patients)} patients, {len(test_slices)} slices")
    
    return {
        'train': train_slices,
        'val': val_slices,
        'test': test_slices
    }


def compute_normalization_stats(image_files, raw_images_dir):
    """
    Compute mean and std from training set for Z-score normalization
    
    Args:
        image_files: List of image filenames
        raw_images_dir: Path to raw images directory
    
    Returns:
        tuple: (mean, std)
    """
    print("\n📐 Computing normalization statistics from training set...")
    
    all_pixels = []
    
    for filename in tqdm(image_files, desc="Loading images"):
        img_path = Path(raw_images_dir) / filename
        
        # Load image (assume .npy format, adjust if different)
        if img_path.suffix == '.npy':
            img = np.load(img_path)
        else:
            # Try loading as image file
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32)
        
        all_pixels.append(img.flatten())
    
    # Concatenate all pixels
    all_pixels = np.concatenate(all_pixels)
    
    # Compute statistics
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)
    
    print(f"   Mean: {mean:.4f}")
    print(f"   Std:  {std:.4f}")
    
    return mean, std


def process_and_save(filename, raw_img_dir, raw_mask_dir, output_img_dir, output_mask_dir,
                    target_size, apply_clahe_flag, clahe_params, mean, std):
    """
    Process a single image-mask pair and save to output directory
    
    Args:
        filename: Name of file to process
        raw_img_dir, raw_mask_dir: Input directories
        output_img_dir, output_mask_dir: Output directories
        target_size: Target image size (H, W)
        apply_clahe_flag: Whether to apply CLAHE
        clahe_params: Dictionary with CLAHE parameters
        mean, std: Normalization parameters
    
    Returns:
        bool: True if successful
    """
    try:
        # Load image and mask
        img_path = Path(raw_img_dir) / filename
        mask_path = Path(raw_mask_dir) / filename
        
        # Load image
        if img_path.suffix == '.npy':
            image = np.load(img_path)
        else:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"⚠️  Could not load {filename}")
                return False
            image = image.astype(np.float32)
        
        # Load mask
        if mask_path.suffix == '.npy':
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️  Could not load mask for {filename}")
                return False
            mask = mask.astype(np.float32)
        
        # Resize if needed
        if target_size is not None:
            image = resize_image(image, target_size, is_mask=False)
            mask = resize_image(mask, target_size, is_mask=True)
        
        # Apply CLAHE (CRUCIAL for faint lesions!)
        if apply_clahe_flag:
            image = apply_clahe(
                image,
                clip_limit=clahe_params['clip_limit'],
                tile_grid_size=clahe_params['tile_grid_size']
            )
        
        # Normalize (Z-score)
        if mean is not None and std is not None and std > 0:
            image = (image - mean) / std
        
        # Ensure mask is binary
        mask = (mask > 0).astype(np.float32)
        
        # Save as .npy
        output_img_path = Path(output_img_dir) / filename.replace(img_path.suffix, '.npy')
        output_mask_path = Path(output_mask_dir) / filename.replace(mask_path.suffix, '.npy')
        
        np.save(output_img_path, image.astype(np.float32))
        np.save(output_mask_path, mask.astype(np.float32))
        
        return True
    
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return False


def main():
    """Main preprocessing pipeline"""
    
    print("="*70)
    print("🔬 DWI IMAGE PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Create directories
    print("\n📁 Step 1: Creating output directories...")
    config.create_directories()
    
    # Step 2: Build slice mapping from raw data
    print("\n📋 Step 2: Building slice mapping from raw data...")
    
    if not config.RAW_IMAGES_DIR.exists():
        print(f"❌ Raw images directory not found: {config.RAW_IMAGES_DIR}")
        print(f"   Please organize your data as described in the instructions.")
        return
    
    slice_mapping = build_slice_mapping(config.RAW_IMAGES_DIR, config.PATIENT_PATTERN)
    
    if len(slice_mapping) == 0:
        print(f"❌ No valid files found in {config.RAW_IMAGES_DIR}")
        print(f"   Make sure your files follow the naming pattern: Patient_XXX_Slice_YYY")
        return
    
    # Print statistics
    stats = get_patient_statistics(slice_mapping)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Patients: {stats['num_patients']}")
    print(f"   Total Slices: {stats['num_slices']}")
    print(f"   Avg Slices/Patient: {stats['avg_slices_per_patient']:.1f}")
    print(f"   Min/Max Slices: {stats['min_slices']}/{stats['max_slices']}")
    
    # Step 3: Split data by slice (NEW METHOD)
    print("\n✂️  Step 3: Splitting data by slice...")
    
    data_splits = split_data_by_slice(
        slice_mapping,
        test_size=config.TEST_SIZE,
        train_val_ratio=config.TRAIN_VAL_SPLIT_RATIO,
        random_seed=config.RANDOM_SEED
    )
    
    # Save split information
    split_info_path = config.DATA_PROCESSED / "data_splits.json"
    with open(split_info_path, 'w') as f:
        json.dump(data_splits, f, indent=4)
    print(f"   ✅ Split information saved to {split_info_path}")
    
    # Step 4: Compute normalization statistics from TRAINING SET ONLY
    print("\n📏 Step 4: Computing normalization statistics...")
    
    if config.NORMALIZE_METHOD == 'zscore':
        mean, std = compute_normalization_stats(
            data_splits['train'],
            config.RAW_IMAGES_DIR
        )
        
        # Save normalization stats
        norm_stats = {'mean': float(mean), 'std': float(std)}
        norm_stats_path = config.DATA_PROCESSED / "normalization_stats.json"
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=4)
        print(f"   ✅ Normalization stats saved to {norm_stats_path}")
    else:
        mean, std = None, None
        print(f"   ⏭️  Skipping (normalization method: {config.NORMALIZE_METHOD})")
    
    # Step 5: Process and save all images
    print("\n🔄 Step 5: Processing and saving images...")
    
    clahe_params = {
        'clip_limit': config.CLAHE_CLIP_LIMIT,
        'tile_grid_size': config.CLAHE_KERNEL_SIZE
    }
    
    # Process each split
    for split_name, filenames in data_splits.items():
        print(f"\n   Processing {split_name.upper()} set ({len(filenames)} files)...")
        
        if split_name == 'train':
            output_img_dir = config.PROCESSED_TRAIN_IMG
            output_mask_dir = config.PROCESSED_TRAIN_MASK
        elif split_name == 'val':
            output_img_dir = config.PROCESSED_VAL_IMG
            output_mask_dir = config.PROCESSED_VAL_MASK
        else:  # test
            output_img_dir = config.PROCESSED_TEST_IMG
            output_mask_dir = config.PROCESSED_TEST_MASK
        
        success_count = 0
        
        for filename in tqdm(filenames, desc=f"   {split_name}"):
            success = process_and_save(
                filename,
                config.RAW_IMAGES_DIR,
                config.RAW_MASKS_DIR,
                output_img_dir,
                output_mask_dir,
                config.IMAGE_SIZE,
                config.CLAHE_ENABLED,
                clahe_params,
                mean,
                std
            )
            
            if success:
                success_count += 1
        
        print(f"   ✅ {split_name.upper()}: {success_count}/{len(filenames)} files processed successfully")
    
    # Step 6: Save preprocessing config
    print("\n💾 Step 6: Saving preprocessing configuration...")
    
    preprocess_config = {
        'image_size': config.IMAGE_SIZE,
        'clahe_enabled': config.CLAHE_ENABLED,
        'clahe_clip_limit': config.CLAHE_CLIP_LIMIT,
        'normalize_method': config.NORMALIZE_METHOD,
        'mean': float(mean) if mean is not None else None,
        'std': float(std) if std is not None else None,
        'train_ratio': config.TRAIN_RATIO,
        'val_ratio': config.VAL_RATIO,
        'test_ratio': config.TEST_RATIO,
        'random_seed': config.RANDOM_SEED,
        'patient_pattern': config.PATIENT_PATTERN,
        'min_slices_per_patient': config.MIN_SLICES_PER_PATIENT
    }
    
    preprocess_config_path = config.DATA_PROCESSED / "preprocess_config.json"
    with open(preprocess_config_path, 'w') as f:
        json.dump(preprocess_config, f, indent=4)
    print(f"   ✅ Preprocessing config saved to {preprocess_config_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nProcessed data saved to: {config.DATA_PROCESSED}")
    print(f"\nNext steps:")
    print(f"   1. Check the processed images in {config.DATA_PROCESSED}")
    print(f"   2. Run training: python train.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
