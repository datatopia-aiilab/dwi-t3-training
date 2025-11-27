"""
PyTorch Dataset for 2.5D DWI Image Loading
Handles loading of 2.5D input (N-1, N, N+1 slices) with augmentation
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DWIDataset25D(Dataset):
    """
    PyTorch Dataset for 2.5D DWI images
    
    Loads 3 consecutive slices (N-1, N, N+1) as input channels
    Handles edge cases with zero padding
    Applies data augmentation on-the-fly
    
    Args:
        image_folder: Path to folder containing processed images (.npy)
        mask_folder: Path to folder containing processed masks (.npy)
        slice_names: List of slice filenames to include
        slice_mapping: Dictionary mapping filenames to neighbor info
        augmentations: Albumentations transforms (optional)
        is_test: If True, don't apply augmentation
    """
    
    def __init__(self, 
                 image_folder, 
                 mask_folder,
                 slice_names,
                 slice_mapping,
                 augmentations=None,
                 is_test=False):
        
        self.image_folder = Path(image_folder)
        self.mask_folder = Path(mask_folder)
        self.slice_names = slice_names
        self.slice_mapping = slice_mapping
        self.augmentations = augmentations if not is_test else None
        self.is_test = is_test
    
    def __len__(self):
        return len(self.slice_names)
    
    def __getitem__(self, idx):
        """
        Load and return a 2.5D sample
        
        Returns:
            image: Tensor (3, H, W) - 2.5D input [N-1, N, N+1]
            mask: Tensor (1, H, W) - Binary mask
            filename: str - Slice filename for reference
        """
        slice_name = self.slice_names[idx]
        
        # Load 2.5D input
        image_25d = self.load_25d_input(slice_name)
        
        # Load mask
        mask = self.load_mask(slice_name)
        
        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image_25d, mask=mask)
            image_25d = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensor
        if not isinstance(image_25d, torch.Tensor):
            # Albumentations with ToTensorV2 already converts to tensor
            # If not using ToTensorV2, manually convert
            image_25d = torch.from_numpy(image_25d).permute(2, 0, 1).float()
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image_25d, mask, slice_name
    
    def load_25d_input(self, slice_name):
        """
        Load 2.5D input: [N-1, N, N+1] slices
        
        Args:
            slice_name: Name of current slice
        
        Returns:
            stacked_image: numpy array (H, W, 3)
        """
        # Load current slice (N)
        current_path = self.image_folder / slice_name
        current_img = np.load(current_path)
        
        h, w = current_img.shape
        
        # Get neighbor information
        info = self.slice_mapping[slice_name]
        
        # Load N-1 (previous slice)
        if info['prev'] is not None:
            prev_path = self.image_folder / info['prev']
            prev_img = np.load(prev_path)
        else:
            # No previous slice -> use zero padding
            prev_img = np.zeros((h, w), dtype=current_img.dtype)
        
        # Load N+1 (next slice)
        if info['next'] is not None:
            next_path = self.image_folder / info['next']
            next_img = np.load(next_path)
        else:
            # No next slice -> use zero padding
            next_img = np.zeros((h, w), dtype=current_img.dtype)
        
        # Stack along channel dimension (H, W, 3)
        stacked_image = np.stack([prev_img, current_img, next_img], axis=-1)
        
        return stacked_image
    
    def load_mask(self, slice_name):
        """
        Load binary mask
        
        Args:
            slice_name: Name of slice
        
        Returns:
            mask: numpy array (H, W) with values {0, 1}
        """
        mask_path = self.mask_folder / slice_name
        mask = np.load(mask_path)
        
        # Ensure binary mask
        mask = (mask > 0).astype(np.float32)
        
        return mask


# ==================== Augmentation Transforms ====================

def get_training_augmentation(config=None):
    """
    Create augmentation pipeline for training
    
    Args:
        config: Configuration module with augmentation parameters
    
    Returns:
        albumentations.Compose object
    """
    if config is None:
        # Default augmentations
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussianNoise(var_limit=(10.0, 50.0), p=0.2),
            ToTensorV2()  # Convert to tensor and normalize to [0, 1]
        ])
    
    # Load from config
    transforms = []
    
    # Horizontal Flip
    if config.AUG_HORIZONTAL_FLIP_PROB > 0:
        transforms.append(A.HorizontalFlip(p=config.AUG_HORIZONTAL_FLIP_PROB))
    
    # Vertical Flip
    if config.AUG_VERTICAL_FLIP_PROB > 0:
        transforms.append(A.VerticalFlip(p=config.AUG_VERTICAL_FLIP_PROB))
    
    # Rotation
    if config.AUG_ROTATE_PROB > 0:
        transforms.append(A.Rotate(limit=config.AUG_ROTATE_LIMIT, p=config.AUG_ROTATE_PROB))
    
    # Elastic Transform (IMPORTANT for medical images!)
    if config.AUG_ELASTIC_TRANSFORM_PROB > 0:
        transforms.append(A.ElasticTransform(
            alpha=config.AUG_ELASTIC_ALPHA,
            sigma=config.AUG_ELASTIC_SIGMA,
            p=config.AUG_ELASTIC_TRANSFORM_PROB
        ))
    
    # Brightness/Contrast
    if config.AUG_BRIGHTNESS_CONTRAST_PROB > 0:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
            contrast_limit=config.AUG_CONTRAST_LIMIT,
            p=config.AUG_BRIGHTNESS_CONTRAST_PROB
        ))
    
    # Gaussian Noise
    if config.AUG_GAUSSIAN_NOISE_PROB > 0:
        transforms.append(A.GaussNoise(
            p=config.AUG_GAUSSIAN_NOISE_PROB
        ))
    
    # Gamma Correction (for intensity variation)
    # Note: Gamma requires non-negative values, so we clip first
    if hasattr(config, 'AUG_GAMMA_PROB') and config.AUG_GAMMA_PROB > 0:
        transforms.append(A.Lambda(
            image=lambda img, **kwargs: np.clip(img, 0, None),  # Ensure non-negative
            p=1.0
        ))
        transforms.append(A.RandomGamma(
            gamma_limit=config.AUG_GAMMA_LIMIT,
            p=config.AUG_GAMMA_PROB
        ))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_validation_augmentation():
    """
    Create augmentation pipeline for validation/test (no augmentation)
    
    Returns:
        albumentations.Compose object
    """
    return A.Compose([
        ToTensorV2()  # Only convert to tensor
    ])


# ==================== DataLoader Creation ====================

def create_dataloaders(config):
    """
    Create train, val, test dataloaders from config
    
    Args:
        config: Configuration module
    
    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    from torch.utils.data import DataLoader
    from utils import build_slice_mapping
    
    # Build slice mapping
    print("Building slice mappings...")
    
    train_mapping = build_slice_mapping(config.PROCESSED_TRAIN_IMG, config.PATIENT_PATTERN)
    val_mapping = build_slice_mapping(config.PROCESSED_VAL_IMG, config.PATIENT_PATTERN)
    test_mapping = build_slice_mapping(config.PROCESSED_TEST_IMG, config.PATIENT_PATTERN)
    
    # Get slice names
    train_names = list(train_mapping.keys())
    val_names = list(val_mapping.keys())
    test_names = list(test_mapping.keys())
    
    print(f"Train: {len(train_names)} slices")
    print(f"Val: {len(val_names)} slices")
    print(f"Test: {len(test_names)} slices")
    
    # Create augmentations
    train_aug = get_training_augmentation(config) if config.AUGMENTATION_ENABLED else None
    val_aug = get_validation_augmentation()
    
    # Create datasets
    train_dataset = DWIDataset25D(
        config.PROCESSED_TRAIN_IMG,
        config.PROCESSED_TRAIN_MASK,
        train_names,
        train_mapping,
        augmentations=train_aug,
        is_test=False
    )
    
    val_dataset = DWIDataset25D(
        config.PROCESSED_VAL_IMG,
        config.PROCESSED_VAL_MASK,
        val_names,
        val_mapping,
        augmentations=val_aug,
        is_test=True
    )
    
    test_dataset = DWIDataset25D(
        config.PROCESSED_TEST_IMG,
        config.PROCESSED_TEST_MASK,
        test_names,
        test_mapping,
        augmentations=val_aug,
        is_test=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.USE_CUDA
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.USE_CUDA
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test one at a time for easier visualization
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.USE_CUDA
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return dataloaders


# ==================== Testing Functions ====================

def test_dataset():
    """ทดสอบ dataset และ dataloader"""
    print("🧪 Testing DWI Dataset 2.5D...\n")
    
    # Create dummy data
    import os
    import tempfile
    
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp())
    img_dir = temp_dir / "images"
    mask_dir = temp_dir / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    
    print(f"Created temp directory: {temp_dir}")
    
    # Create dummy data: 2 patients, 3 slices each
    h, w = 128, 128
    
    for patient_id in range(1, 3):
        for slice_num in range(1, 4):
            filename = f"Patient_{patient_id:03d}_Slice_{slice_num:03d}.npy"
            
            # Create dummy image and mask
            img = np.random.rand(h, w).astype(np.float32)
            mask = (np.random.rand(h, w) > 0.8).astype(np.float32)
            
            np.save(img_dir / filename, img)
            np.save(mask_dir / filename, mask)
    
    print(f"Created 6 dummy files (2 patients x 3 slices)\n")
    
    # Build slice mapping
    from utils import build_slice_mapping
    
    slice_mapping = build_slice_mapping(img_dir)
    slice_names = list(slice_mapping.keys())
    
    print("Slice mapping:")
    for name, info in slice_mapping.items():
        print(f"  {name}: prev={info['prev']}, next={info['next']}")
    
    # Create dataset
    print("\n" + "="*60)
    print("Testing Dataset")
    print("="*60)
    
    augmentations = get_validation_augmentation()  # No augmentation for testing
    
    dataset = DWIDataset25D(
        img_dir,
        mask_dir,
        slice_names,
        slice_mapping,
        augmentations=augmentations,
        is_test=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading samples
    for i in range(len(dataset)):
        image, mask = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Mask unique: {mask.unique()}")
        
        # Check shapes
        assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
        assert mask.shape[0] == 1, f"Expected 1 channel mask, got {mask.shape[0]}"
    
    # Test DataLoader
    print("\n" + "="*60)
    print("Testing DataLoader")
    print("="*60)
    
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    for batch_idx, (images, masks) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Check batch shapes
        assert images.shape[1] == 3, "Expected 3 channels"
        assert masks.shape[1] == 1, "Expected 1 mask channel"
    
    # Test augmentation
    print("\n" + "="*60)
    print("Testing Augmentation")
    print("="*60)
    
    aug_transforms = get_training_augmentation()
    
    dataset_aug = DWIDataset25D(
        img_dir,
        mask_dir,
        slice_names,
        slice_mapping,
        augmentations=aug_transforms,
        is_test=False
    )
    
    # Load same sample multiple times to see augmentation variations
    sample_idx = 0
    print(f"\nLoading sample {sample_idx} multiple times with augmentation:")
    
    for i in range(3):
        image, mask = dataset_aug[sample_idx]
        print(f"  Iteration {i+1}: Image range [{image.min():.3f}, {image.max():.3f}]")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up temp directory")
    
    print("\n✅ All dataset tests passed!")


if __name__ == "__main__":
    test_dataset()
