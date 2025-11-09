"""
Complete Pipeline Testing Script
Creates dummy data and tests the entire pipeline end-to-end
"""

import numpy as np
import torch
from pathlib import Path
import shutil
import sys

# Import our modules
import config
from utils import build_slice_mapping, calculate_all_metrics, visualize_sample_advanced
from loss import get_loss_function
from model import get_attention_unet
from dataset import DWIDataset25D, get_validation_augmentation


def create_dummy_data(num_patients=3, slices_per_patient=5, image_size=(128, 128)):
    """
    สร้างข้อมูลจำลองสำหรับทดสอบ
    
    Args:
        num_patients: จำนวน patients
        slices_per_patient: จำนวน slices ต่อ patient
        image_size: ขนาดภาพ (H, W)
    
    Returns:
        Path to created data directory
    """
    print("\n" + "="*70)
    print("🎭 CREATING DUMMY DATA")
    print("="*70)
    
    # Create temporary data directories
    raw_img_dir = config.RAW_IMAGES_DIR
    raw_mask_dir = config.RAW_MASKS_DIR
    
    raw_img_dir.mkdir(parents=True, exist_ok=True)
    raw_mask_dir.mkdir(parents=True, exist_ok=True)
    
    h, w = image_size
    total_files = 0
    
    for patient_id in range(1, num_patients + 1):
        for slice_num in range(1, slices_per_patient + 1):
            filename = f"Patient_{patient_id:03d}_Slice_{slice_num:03d}.npy"
            
            # Create dummy image with some structure
            img = np.random.rand(h, w).astype(np.float32) * 0.5
            
            # Add a "lesion-like" bright spot randomly
            if np.random.rand() > 0.3:  # 70% chance of lesion
                center_y = np.random.randint(h//4, 3*h//4)
                center_x = np.random.randint(w//4, 3*w//4)
                radius = np.random.randint(10, 30)
                
                Y, X = np.ogrid[:h, :w]
                dist = np.sqrt((Y - center_y)**2 + (X - center_x)**2)
                lesion_mask = dist <= radius
                
                # Add bright lesion
                img[lesion_mask] = np.random.rand() * 0.5 + 0.5
                
                # Create corresponding mask
                mask = lesion_mask.astype(np.float32)
            else:
                # No lesion
                mask = np.zeros((h, w), dtype=np.float32)
            
            # Save files
            np.save(raw_img_dir / filename, img)
            np.save(raw_mask_dir / filename, mask)
            total_files += 1
    
    print(f"✅ Created {total_files} dummy files")
    print(f"   Images: {raw_img_dir}")
    print(f"   Masks:  {raw_mask_dir}")
    print(f"   {num_patients} patients × {slices_per_patient} slices = {total_files} files")
    
    return raw_img_dir


def test_preprocessing():
    """ทดสอบ preprocessing pipeline"""
    print("\n" + "="*70)
    print("🔬 TEST 1: PREPROCESSING")
    print("="*70)
    
    # Import preprocessing functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("preprocess", "01_preprocess.py")
    preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess)
    
    split_data_by_patient = preprocess.split_data_by_patient
    apply_clahe = preprocess.apply_clahe
    compute_normalization_stats = preprocess.compute_normalization_stats
    process_and_save = preprocess.process_and_save
    
    # Build slice mapping
    print("\n1. Building slice mapping...")
    slice_mapping = build_slice_mapping(config.RAW_IMAGES_DIR, config.PATIENT_PATTERN)
    print(f"   ✅ Found {len(slice_mapping)} slices")
    
    # Test data splitting
    print("\n2. Testing data split...")
    splits = split_data_by_patient(slice_mapping, 0.6, 0.2, 0.2, random_seed=42)
    print(f"   ✅ Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Test CLAHE
    print("\n3. Testing CLAHE...")
    dummy_img = np.random.rand(128, 128).astype(np.float32)
    enhanced = apply_clahe(dummy_img, clip_limit=0.03)
    print(f"   ✅ Input range: [{dummy_img.min():.3f}, {dummy_img.max():.3f}]")
    print(f"   ✅ Output range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    
    # Test normalization stats
    print("\n4. Testing normalization stats computation...")
    mean, std = compute_normalization_stats(splits['train'][:5], config.RAW_IMAGES_DIR)
    print(f"   ✅ Mean: {mean:.4f}, Std: {std:.4f}")
    
    print("\n✅ Preprocessing tests passed!")


def test_dataset_and_dataloader():
    """ทดสอบ dataset และ dataloader"""
    print("\n" + "="*70)
    print("📦 TEST 2: DATASET & DATALOADER")
    print("="*70)
    
    # Assume preprocessing has been run
    if not config.PROCESSED_TRAIN_IMG.exists():
        print("⚠️  Processed data not found. Please run preprocessing first.")
        return False
    
    # Build slice mapping
    print("\n1. Building slice mapping for processed data...")
    train_mapping = build_slice_mapping(config.PROCESSED_TRAIN_IMG, config.PATIENT_PATTERN)
    print(f"   ✅ Found {len(train_mapping)} training slices")
    
    # Create dataset
    print("\n2. Creating dataset...")
    train_names = list(train_mapping.keys())[:10]  # Test with first 10
    
    augmentations = get_validation_augmentation()
    
    dataset = DWIDataset25D(
        config.PROCESSED_TRAIN_IMG,
        config.PROCESSED_TRAIN_MASK,
        train_names,
        train_mapping,
        augmentations=augmentations,
        is_test=True
    )
    
    print(f"   ✅ Dataset size: {len(dataset)}")
    
    # Test loading samples
    print("\n3. Testing sample loading...")
    for i in range(min(3, len(dataset))):
        image, mask = dataset[i]
        print(f"   Sample {i}: Image {image.shape}, Mask {mask.shape}")
        assert image.shape[0] == 3, "Expected 3 channels (2.5D)"
        assert mask.shape[0] == 1, "Expected 1 mask channel"
    
    # Test DataLoader
    print("\n4. Testing DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    for batch_idx, (images, masks) in enumerate(loader):
        print(f"   Batch {batch_idx}: Images {images.shape}, Masks {masks.shape}")
        if batch_idx >= 2:  # Test only first 3 batches
            break
    
    print("\n✅ Dataset and DataLoader tests passed!")
    return True


def test_model_architecture():
    """ทดสอบ model architecture"""
    print("\n" + "="*70)
    print("🏗️  TEST 3: MODEL ARCHITECTURE")
    print("="*70)
    
    print("\n1. Creating model...")
    model = get_attention_unet(config)
    
    params = model.count_parameters()
    print(f"   ✅ Model created")
    print(f"   Parameters: {params['total']:,}")
    print(f"   Model size: ~{params['total'] * 4 / (1024**2):.2f} MB")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 128, 128)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    assert output.shape == (batch_size, 1, 128, 128), "Output shape mismatch!"
    assert output.min() >= 0 and output.max() <= 1, "Output not in [0, 1]!"
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    dummy_input_grad = torch.randn(2, 3, 128, 128, requires_grad=True)
    dummy_target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    
    output = model(dummy_input_grad)
    loss = torch.nn.functional.binary_cross_entropy(output, dummy_target)
    loss.backward()
    
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Gradient computed: {dummy_input_grad.grad is not None}")
    
    print("\n✅ Model architecture tests passed!")


def test_loss_functions():
    """ทดสอบ loss functions"""
    print("\n" + "="*70)
    print("📉 TEST 4: LOSS FUNCTIONS")
    print("="*70)
    
    # Create dummy predictions and targets
    pred = torch.rand(4, 1, 128, 128)
    target = (torch.rand(4, 1, 128, 128) > 0.5).float()
    
    print("\n1. Testing loss functions...")
    
    loss_types = ['focal', 'dice', 'combo']
    
    for loss_type in loss_types:
        loss_fn = get_loss_function(loss_type)
        loss_value = loss_fn(pred, target)
        print(f"   {loss_type.capitalize()} Loss: {loss_value.item():.6f}")
    
    # Test backward pass
    print("\n2. Testing backward pass...")
    pred_grad = torch.rand(2, 1, 64, 64, requires_grad=True)
    target_grad = (torch.rand(2, 1, 64, 64) > 0.5).float()
    
    combo_loss = get_loss_function('combo')
    loss = combo_loss(pred_grad, target_grad)
    loss.backward()
    
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Gradient mean: {pred_grad.grad.mean().item():.6f}")
    
    print("\n✅ Loss function tests passed!")


def test_metrics():
    """ทดสอบ metrics calculation"""
    print("\n" + "="*70)
    print("📊 TEST 5: METRICS CALCULATION")
    print("="*70)
    
    # Create dummy predictions and targets
    pred = np.random.rand(128, 128) > 0.6
    target = np.random.rand(128, 128) > 0.5
    
    print("\n1. Calculating metrics...")
    metrics = calculate_all_metrics(pred.astype(float), target.astype(float))
    
    print(f"   Dice Score: {metrics['dice']:.4f}")
    print(f"   IoU: {metrics['iou']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1 Score: {metrics['f1']:.4f}")
    
    print("\n✅ Metrics calculation tests passed!")


def test_visualization():
    """ทดสอบ visualization"""
    print("\n" + "="*70)
    print("🎨 TEST 6: VISUALIZATION")
    print("="*70)
    
    print("\n1. Creating sample visualization...")
    
    # Create dummy data
    image = np.random.rand(128, 128, 3)  # 2.5D
    mask = np.random.rand(128, 128) > 0.7
    prediction = np.random.rand(128, 128) > 0.6
    
    # Calculate volumes for visualization
    pixel_spacing = 4.0  # mm
    gt_volume_ml = np.sum(mask) * (pixel_spacing ** 2) * 1.0 / 1000
    pred_volume_ml = np.sum(prediction) * (pixel_spacing ** 2) * 1.0 / 1000
    
    fig = visualize_sample_advanced(
        image, mask, prediction, 
        filename="Test Sample",
        gt_volume_ml=gt_volume_ml,
        pred_volume_ml=pred_volume_ml,
        pixel_spacing=pixel_spacing
    )
    
    # Save figure
    test_viz_path = config.RESULTS_DIR / "test_visualization.png"
    fig.savefig(test_viz_path, dpi=150, bbox_inches='tight')
    
    print(f"   ✅ Visualization saved to {test_viz_path}")
    
    import matplotlib.pyplot as plt
    plt.close(fig)
    
    print("\n✅ Visualization tests passed!")


def test_complete_pipeline():
    """ทดสอบ pipeline ทั้งหมดตั้งแต่ต้นจนจบ"""
    print("\n" + "="*70)
    print("🚀 COMPLETE PIPELINE TEST")
    print("="*70)
    
    # Step 1: Create dummy data
    create_dummy_data(num_patients=3, slices_per_patient=5, image_size=(128, 128))
    
    # Step 2: Test preprocessing
    try:
        test_preprocessing()
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False
    
    # Run actual preprocessing
    print("\n" + "="*70)
    print("🔄 RUNNING ACTUAL PREPROCESSING...")
    print("="*70)
    
    try:
        # Import and run preprocessing
        import importlib.util
        spec = importlib.util.spec_from_file_location("preprocess", "01_preprocess.py")
        preprocess = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocess)
        preprocess.main()
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test dataset
    try:
        test_dataset_and_dataloader()
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test model
    try:
        test_model_architecture()
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    # Step 5: Test loss
    try:
        test_loss_functions()
    except Exception as e:
        print(f"❌ Loss test failed: {e}")
        return False
    
    # Step 6: Test metrics
    try:
        test_metrics()
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False
    
    # Step 7: Test visualization
    try:
        test_visualization()
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False
    
    # Step 8: Mini training test (1 batch)
    print("\n" + "="*70)
    print("🎓 TEST 7: MINI TRAINING (1 BATCH)")
    print("="*70)
    
    try:
        # Create mini dataset
        train_mapping = build_slice_mapping(config.PROCESSED_TRAIN_IMG, config.PATIENT_PATTERN)
        train_names = list(train_mapping.keys())[:4]  # Just 4 samples
        
        dataset = DWIDataset25D(
            config.PROCESSED_TRAIN_IMG,
            config.PROCESSED_TRAIN_MASK,
            train_names,
            train_mapping,
            augmentations=get_validation_augmentation(),
            is_test=True
        )
        
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Create model, loss, optimizer
        model = get_attention_unet(config)
        criterion = get_loss_function('combo')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Train on 1 batch
        images, masks = next(iter(loader))
        
        model.train()
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        print(f"   ✅ Training step completed")
        print(f"   Loss: {loss.item():.6f}")
        
        # Test inference
        model.eval()
        with torch.no_grad():
            pred = model(images)
        
        print(f"   ✅ Inference completed")
        print(f"   Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")
        
    except Exception as e:
        print(f"❌ Mini training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n🎉 The complete pipeline is working correctly!")
    print(f"\nYou can now:")
    print(f"   1. Replace dummy data with real data in: {config.RAW_IMAGES_DIR}")
    print(f"   2. Run preprocessing: python 01_preprocess.py")
    print(f"   3. Run training: python train.py")
    print(f"   4. Run evaluation: python evaluate.py")
    
    return True


def cleanup_test_data():
    """ลบข้อมูลทดสอบ"""
    print("\n" + "="*70)
    print("🧹 CLEANING UP TEST DATA")
    print("="*70)
    
    dirs_to_clean = [
        config.DATA_RAW,
        config.DATA_PROCESSED,
        config.RESULTS_DIR
    ]
    
    for directory in dirs_to_clean:
        if directory.exists():
            shutil.rmtree(directory)
            print(f"   ✅ Removed {directory}")
    
    print("\n✅ Cleanup completed!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 DWI SEGMENTATION - COMPLETE PIPELINE TEST")
    print("="*70)
    
    # Run complete pipeline test
    success = test_complete_pipeline()
    
    if success:
        print("\n" + "="*70)
        print("✅ SUCCESS! All components are working correctly.")
        print("="*70)
        
        # Ask if user wants to cleanup
        response = input("\n Do you want to cleanup test data? (y/n): ")
        if response.lower() == 'y':
            cleanup_test_data()
        else:
            print("\nTest data kept. You can inspect it manually.")
    
    else:
        print("\n" + "="*70)
        print("❌ FAILURE! Some tests failed. Please check the errors above.")
        print("="*70)
        sys.exit(1)
