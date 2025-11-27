"""
Test script to verify that model receives 3-channel 2.5D input
"""

import torch
import numpy as np
from pathlib import Path

import config
from model import get_attention_unet
from dataset import DWIDataset25D, get_training_augmentation, get_validation_augmentation
from utils import build_slice_mapping


def test_model_input():
    """
    ทดสอบว่า Model รับ input กี่ channels จริงๆ
    """
    print("="*70)
    print("🧪 TESTING MODEL INPUT SHAPE")
    print("="*70)
    
    # 1. สร้าง Model
    print("\n1️⃣ Creating model...")
    model = get_attention_unet(config)
    print(f"   Model input channels: {model.in_channels}")
    print(f"   Model output channels: {model.out_channels}")
    
    # 2. ทดสอบด้วย dummy data
    print("\n2️⃣ Testing with dummy input...")
    
    # สร้าง dummy input
    batch_size = 2
    height, width = 256, 256
    
    # Test 1: 1-channel input
    print("\n   Test A: 1-channel input (2D)")
    try:
        dummy_1ch = torch.randn(batch_size, 1, height, width)
        output_1ch = model(dummy_1ch)
        print(f"      ✅ Input shape: {dummy_1ch.shape}")
        print(f"      ✅ Output shape: {output_1ch.shape}")
        print(f"      ⚠️  Model ACCEPTS 1-channel input (but was designed for 3!)")
    except Exception as e:
        print(f"      ❌ Model REJECTS 1-channel input: {e}")
    
    # Test 2: 3-channel input
    print("\n   Test B: 3-channel input (2.5D)")
    try:
        dummy_3ch = torch.randn(batch_size, 3, height, width)
        output_3ch = model(dummy_3ch)
        print(f"      ✅ Input shape: {dummy_3ch.shape}")
        print(f"      ✅ Output shape: {output_3ch.shape}")
        print(f"      ✅ Model ACCEPTS 3-channel input (as designed!)")
    except Exception as e:
        print(f"      ❌ Model REJECTS 3-channel input: {e}")
    
    # 3. ทดสอบด้วย real data (ถ้ามี)
    print("\n3️⃣ Testing with real data...")
    
    train_img_dir = config.PROCESSED_TRAIN_IMG
    train_mask_dir = config.PROCESSED_TRAIN_MASK
    
    if train_img_dir.exists() and train_mask_dir.exists():
        print(f"   Loading from: {train_img_dir}")
        
        # Build slice mapping
        slice_mapping_file = config.DATA_PROCESSED / 'slice_mapping.json'
        if slice_mapping_file.exists():
            import json
            with open(slice_mapping_file, 'r') as f:
                slice_mapping = json.load(f)
        else:
            print("   ⚠️  slice_mapping.json not found, building from scratch...")
            slice_mapping = build_slice_mapping(config.RAW_IMAGES_DIR, config.PATIENT_PATTERN)
        
        # Get slice names
        slice_names = list(slice_mapping.keys())[:5]  # First 5 samples
        
        # Create dataset
        dataset = DWIDataset25D(
            image_folder=train_img_dir,
            mask_folder=train_mask_dir,
            slice_names=slice_names,
            slice_mapping=slice_mapping,
            augmentations=None,
            is_test=True
        )
        
        print(f"   Dataset size: {len(dataset)} samples")
        
        # Load one sample
        image, mask = dataset[0]
        
        print(f"\n   Real data from dataset:")
        print(f"      Image shape: {image.shape}")
        print(f"      Image channels: {image.shape[0]}")
        print(f"      Mask shape: {mask.shape}")
        
        if image.shape[0] == 3:
            print(f"      ✅ Dataset provides 3-channel input (2.5D)")
            print(f"         Channel 0: Previous slice (N-1)")
            print(f"         Channel 1: Current slice (N)")
            print(f"         Channel 2: Next slice (N+1)")
        elif image.shape[0] == 1:
            print(f"      ⚠️  Dataset provides 1-channel input (2D only)")
        else:
            print(f"      ❓ Dataset provides {image.shape[0]}-channel input")
        
        # Test with model
        image_batch = image.unsqueeze(0)  # Add batch dimension
        print(f"\n   Testing model with real data batch: {image_batch.shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(image_batch)
        
        print(f"   ✅ Model output shape: {output.shape}")
        print(f"   ✅ Model successfully processed {image.shape[0]}-channel input!")
        
    else:
        print("   ⚠️  Processed data not found. Run preprocessing first:")
        print("      python 01_preprocess.py")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETED")
    print("="*70)
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   • Model is configured for: {model.in_channels} input channels")
    print(f"   • Config IN_CHANNELS: {config.IN_CHANNELS}")
    print(f"   • Model architecture: {'2.5D' if model.in_channels == 3 else '2D' if model.in_channels == 1 else 'Custom'}")
    
    if model.in_channels == 3:
        print(f"\n   ✅ Conclusion: Model uses 2.5D (3 slices)")
        print(f"      Each input has shape: (3, H, W)")
        print(f"      - Channel 0: Previous slice")
        print(f"      - Channel 1: Current slice")
        print(f"      - Channel 2: Next slice")
    elif model.in_channels == 1:
        print(f"\n   ⚠️  Conclusion: Model uses 2D (1 slice only)")
        print(f"      Each input has shape: (1, H, W)")
    
    print("\n")


if __name__ == "__main__":
    test_model_input()
