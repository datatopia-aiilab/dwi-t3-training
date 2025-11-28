"""
Test By Case - Full Pipeline with Classification
1. Neg/Pos Classification (มี lesion หรือไม่)
2. Artifact/Non Classification (มี artifact หรือไม่)
3. Lesion Segmentation (segment ด้วย PyTorch model)

Usage:
    python testall_bycase_full.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# TensorFlow for classification models
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import config and model
import config
from model import AttentionUNet


# ============================================================================
# CONFIGURATION - กำหนดค่าที่นี่
# ============================================================================
CONFIG = {
    # Models
    'seg_model_path': 'models/best_model.pth',                                  # PyTorch segmentation model
    'neg_pos_model_path': '/mnt/d/AiiLAB_PROJECTS/DWI/NovEdition/dwi-t3-training/t3-training-base/models_cls/model_neg_pos.h5',     # TensorFlow Neg/Pos
    'artifact_model_path': '/mnt/d/AiiLAB_PROJECTS/DWI/NovEdition/dwi-t3-training/t3-training-base/models_cls/model_artifact.h5',   # TensorFlow Artifact
    
    # Classification Options
    'use_neg_pos': True,      # Enable Neg/Pos classification
    'use_artifact': True,     # Enable Artifact classification
    
    # Data
    'test_data_path': '/mnt/d/AiiLAB_PROJECTS/DWI/NovEdition/dwi-t3-training/t3-training-base/validation_dataset',
    'batch_size': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_base': 'test_results_bycase_full',
    'dpi': 300,
    'image_format': 'png',
    
    # Image settings
    'classify_img_size': (256, 256),   # For TF classification models
    'segment_img_size': config.IMAGE_SIZE,  # For PyTorch segmentation
    
    # Image extensions
    'image_extensions': ['*.png', '*.jpg', '*.jpeg', '*.npy', '*.nii.gz'],
}

# Classification labels
NEG_POS_CLASSES = {0: "Negative", 1: "Positive"}
ARTIFACT_CLASSES = {0: "Artifact", 1: "NonArtifact"}


# ==================== TensorFlow Classification Functions ====================

def zscore_normalization(image):
    """Z-score normalization"""
    mean = np.mean(image)
    std = np.std(image)
    return image - mean if std == 0 else (image - mean) / std


def duplicate_channels(image):
    """Convert grayscale to 3-channel RGB"""
    if len(image.shape) == 2 or image.shape[-1] == 1:
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        return np.concatenate([image, image, image], axis=-1)
    return image


def preprocess_classify_image(image_array):
    """
    Preprocess image for TensorFlow classification
    Input: (H, W) or (H, W, 1) or (H, W, 3)
    Output: (H, W, 3) normalized
    """
    # Normalize
    image = zscore_normalization(image_array)
    # Convert to 3 channels
    image = duplicate_channels(image)
    return image


def classify_image_array(model, image_array, model_type='neg_pos'):
    """
    Classify image using TensorFlow model
    
    Args:
        model: TensorFlow model
        image_array: numpy array (H, W) or (H, W, 3)
        model_type: 'neg_pos' or 'artifact'
    
    Returns:
        result: 0 or 1
        confidence: float (0-1)
    """
    try:
        # Resize to classification size
        img_resized = cv2.resize(image_array, CONFIG['classify_img_size'])
        
        # Preprocess
        img_processed = preprocess_classify_image(img_resized)
        
        # Add batch dimension
        img_batch = np.expand_dims(img_processed, axis=0)
        
        # Predict
        prediction = model.predict(img_batch, batch_size=1, verbose=0)
        result = (prediction > 0.5).astype(int)[0][0]
        confidence = float(prediction[0][0])
        
        return result, confidence
    except Exception as e:
        print(f"Error in classification: {e}")
        return None, None


# ==================== Dataset ====================

class FullPipelineDataset(Dataset):
    """Dataset for full pipeline with classification"""
    
    def __init__(self, images, filenames, case_names):
        """
        Args:
            images: numpy array of shape (N, 3, H, W) - 2.5D images
            filenames: list of original filenames
            case_names: list of case identifiers (folder paths)
        """
        self.images = images
        self.filenames = filenames
        self.case_names = case_names
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float()  # (3, H, W)
        filename = self.filenames[idx]
        case_name = self.case_names[idx]
        
        return image, filename, case_name


# ==================== Data Loading ====================

def find_all_images(root_path, extensions=['*.npy', '*.png', '*.jpg', '*.jpeg']):
    """Recursively find all image files"""
    all_files = []
    root = Path(root_path)
    
    if not root.exists():
        raise FileNotFoundError(f"Test data path does not exist: {root}")
    
    print(f"Scanning for images in: {root}")
    
    for ext in extensions:
        pattern = f"**/{ext}"
        files = list(root.glob(pattern))
        all_files.extend(files)
        if files:
            print(f"  Found {len(files)} {ext} files")
    
    return sorted(set(all_files))


def load_full_pipeline_data():
    """
    Load all images from test_data_path
    Returns: FullPipelineDataset
    """
    print("\n" + "="*60)
    print("📂 Loading Data for Full Pipeline")
    print("="*60)
    
    test_path = Path(CONFIG['test_data_path'])
    
    # Find all image files
    image_files = find_all_images(test_path, CONFIG['image_extensions'])
    
    print(f"\nTotal files found: {len(image_files)}")
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {test_path}")
    
    # Load all images
    all_images = []
    all_filenames = []
    all_case_names = []
    skipped = 0
    
    print(f"\nLoading images...")
    for img_path in tqdm(image_files, desc="Loading"):
        try:
            # Store full relative path (including folder structure)
            relative_path = img_path.relative_to(test_path)
            # case_name now includes full folder structure: e.g., "A/203"
            case_name = str(relative_path.parent) if relative_path.parent != Path('.') else 'root'
            
            # Load based on extension
            if img_path.suffix == '.npy':
                img_data = np.load(str(img_path))
                
                # Check if already preprocessed (2.5D: 3, H, W)
                if img_data.ndim == 3 and img_data.shape[0] == 3:
                    # Already 2.5D
                    if img_data.shape[1:] != tuple(config.IMAGE_SIZE):
                        img_3d = np.stack([
                            cv2.resize(img_data[0], config.IMAGE_SIZE),
                            cv2.resize(img_data[1], config.IMAGE_SIZE),
                            cv2.resize(img_data[2], config.IMAGE_SIZE)
                        ], axis=0)
                    else:
                        img_3d = img_data
                        
                elif img_data.ndim == 3:
                    # 3D volume (H, W, D) - process each slice
                    for slice_idx in range(img_data.shape[2]):
                        img_slice = img_data[:, :, slice_idx]
                        img_slice_resized = cv2.resize(img_slice, config.IMAGE_SIZE)
                        img_slice_norm = (img_slice_resized - img_slice_resized.mean()) / (img_slice_resized.std() + 1e-8)
                        
                        # Create 2.5D
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
                        
                        img_3d = np.stack([prev_slice, img_slice_norm, next_slice], axis=0)
                        
                        all_images.append(img_3d)
                        all_filenames.append(f"{img_path.stem}_slice_{slice_idx:03d}")
                        all_case_names.append(case_name)
                    
                    continue  # Skip the append below for 3D volumes
                    
                elif img_data.ndim == 2:
                    # Single 2D image
                    img_slice = cv2.resize(img_data, config.IMAGE_SIZE)
                    img_slice_norm = (img_slice - img_slice.mean()) / (img_slice.std() + 1e-8)
                    # Use same slice for all 3 channels
                    img_3d = np.stack([img_slice_norm, img_slice_norm, img_slice_norm], axis=0)
                else:
                    print(f"Warning: Unexpected shape {img_data.shape} for {img_path}, skipping...")
                    skipped += 1
                    continue
                    
            elif img_path.suffix in ['.png', '.jpg', '.jpeg']:
                # RGB/Grayscale images
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Cannot read {img_path}, skipping...")
                    skipped += 1
                    continue
                
                img_resized = cv2.resize(img, config.IMAGE_SIZE)
                img_norm = (img_resized - img_resized.mean()) / (img_resized.std() + 1e-8)
                # Use same slice for all 3 channels (2.5D format)
                img_3d = np.stack([img_norm, img_norm, img_norm], axis=0)
                
            elif img_path.suffix == '.gz':  # .nii.gz
                import nibabel as nib
                nii = nib.load(str(img_path))
                img_data = nii.get_fdata()
                
                # Process 3D volume
                if img_data.ndim == 2:
                    img_data = img_data[:, :, np.newaxis]
                
                for slice_idx in range(img_data.shape[2]):
                    img_slice = img_data[:, :, slice_idx]
                    img_slice_resized = cv2.resize(img_slice, config.IMAGE_SIZE)
                    img_slice_norm = (img_slice_resized - img_slice_resized.mean()) / (img_slice_resized.std() + 1e-8)
                    
                    # Create 2.5D
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
                    
                    img_3d = np.stack([prev_slice, img_slice_norm, next_slice], axis=0)
                    
                    all_images.append(img_3d)
                    all_filenames.append(f"{img_path.stem}_slice_{slice_idx:03d}")
                    all_case_names.append(case_name)
                
                continue  # Skip the append below for 3D volumes
            else:
                print(f"Warning: Unsupported format {img_path.suffix}, skipping...")
                skipped += 1
                continue
            
            # Add to lists (for single 2D images)
            all_images.append(img_3d)
            all_filenames.append(img_path.stem)
            all_case_names.append(case_name)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            skipped += 1
            continue
    
    # Convert to numpy arrays
    all_images = np.array(all_images, dtype=np.float32)
    
    print(f"\n✅ Data Loading Summary:")
    print(f"  Total loaded: {len(all_images)} images")
    print(f"  Skipped: {skipped}")
    print(f"  Image shape: {all_images.shape} (N, 3, H, W) - 2.5D")
    
    if len(all_images) == 0:
        raise ValueError("No valid images loaded. Check your data!")
    
    # Create dataset
    dataset = FullPipelineDataset(all_images, all_filenames, all_case_names)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    print("="*60 + "\n")
    
    return dataloader


# ==================== Statistics Calculation ====================

def calculate_stats(pred_mask):
    """Calculate lesion statistics"""
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    
    # Total lesion area
    total_pixels = binary_mask.size
    lesion_pixels = np.sum(binary_mask)
    lesion_percentage = (lesion_pixels / total_pixels) * 100
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    num_regions = num_labels - 1  # Exclude background
    
    if num_regions > 0:
        # Get areas of all regions (exclude background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_region = int(np.max(areas))
        mean_region_size = float(np.mean(areas))
    else:
        largest_region = 0
        mean_region_size = 0.0
    
    return {
        'lesion_percentage': float(lesion_percentage),
        'num_regions': int(num_regions),
        'largest_region': largest_region,
        'mean_region_size': mean_region_size
    }


# ==================== Main Pipeline ====================

def run_full_pipeline(seg_model_path, neg_pos_model, artifact_model, dataloader, output_dir, device):
    """
    Full pipeline with classification and segmentation
    
    Pipeline:
    1. Neg/Pos Classification (skip if Negative)
    2. Artifact Classification (skip if Artifact)
    3. Segmentation (only if Positive and NonArtifact)
    """
    print("\n" + "="*60)
    print("🚀 Running Full Pipeline")
    print("="*60)
    
    # Load segmentation model
    print(f"Loading segmentation model from: {seg_model_path}")
    seg_model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS
    ).to(device)
    seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
    seg_model.eval()
    print("✓ Segmentation model loaded")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Statistics tracking
    all_stats = []
    all_filenames = []
    all_cases = []
    all_neg_pos_results = []
    all_artifact_results = []
    
    skipped_negative = 0
    skipped_artifact = 0
    processed_count = 0
    
    with torch.no_grad():
        for images, filenames, case_names in tqdm(dataloader, desc="Processing"):
            images = images.to(device)
            
            for i in range(len(filenames)):
                img_2_5d = images[i].cpu().numpy()  # (3, H, W)
                img_slice = img_2_5d[1]  # Middle slice for classification and visualization
                filename = filenames[i]
                case_name = case_names[i]
                
                # Initialize results
                neg_pos_result = None
                neg_pos_confidence = None
                artifact_result = None
                artifact_confidence = None
                skip_reason = None
                
                # Step 1: Neg/Pos Classification
                if CONFIG['use_neg_pos'] and neg_pos_model is not None:
                    neg_pos_result, neg_pos_confidence = classify_image_array(
                        neg_pos_model, img_slice, 'neg_pos'
                    )
                    
                    if neg_pos_result is not None:
                        label = NEG_POS_CLASSES[neg_pos_result]
                        print(f"\n{filename}: Neg/Pos = {label} (confidence: {neg_pos_confidence:.4f})")
                        
                        # Skip if Negative
                        if neg_pos_result == 0:  # Negative
                            skip_reason = "Negative"
                            skipped_negative += 1
                            all_neg_pos_results.append(label)
                            all_artifact_results.append("N/A")
                            all_stats.append({
                                'lesion_percentage': 0.0,
                                'num_regions': 0,
                                'largest_region': 0,
                                'mean_region_size': 0.0,
                                'skip_reason': skip_reason
                            })
                            all_filenames.append(filename)
                            all_cases.append(case_name)
                            continue
                    else:
                        print(f"\n{filename}: Neg/Pos classification failed, skipping...")
                        continue
                else:
                    neg_pos_result = 1  # Assume Positive if not using classification
                    all_neg_pos_results.append("Assumed Positive")
                
                # Step 2: Artifact Classification
                if CONFIG['use_artifact'] and artifact_model is not None:
                    artifact_result, artifact_confidence = classify_image_array(
                        artifact_model, img_slice, 'artifact'
                    )
                    
                    if artifact_result is not None:
                        label = ARTIFACT_CLASSES[artifact_result]
                        print(f"{filename}: Artifact = {label} (confidence: {artifact_confidence:.4f})")
                        
                        # Skip if Artifact
                        if artifact_result == 0:  # Artifact
                            skip_reason = "Artifact"
                            skipped_artifact += 1
                            all_artifact_results.append(label)
                            all_stats.append({
                                'lesion_percentage': 0.0,
                                'num_regions': 0,
                                'largest_region': 0,
                                'mean_region_size': 0.0,
                                'skip_reason': skip_reason
                            })
                            all_filenames.append(filename)
                            all_cases.append(case_name)
                            continue
                    else:
                        print(f"{filename}: Artifact classification failed, skipping...")
                        continue
                else:
                    artifact_result = 1  # Assume NonArtifact if not using classification
                    all_artifact_results.append("Assumed NonArtifact")
                
                # Step 3: Segmentation (only if Positive and NonArtifact)
                print(f"{filename}: Running segmentation...")
                outputs = seg_model(images[i:i+1])
                preds = (outputs > 0.5).float()
                pred_mask = preds[0, 0].cpu().numpy()
                
                # Calculate statistics
                stats = calculate_stats(pred_mask)
                stats['skip_reason'] = None
                
                all_stats.append(stats)
                all_filenames.append(filename)
                all_cases.append(case_name)
                all_artifact_results.append(ARTIFACT_CLASSES.get(artifact_result, "N/A"))
                
                # Create case-specific output directory (preserving folder structure)
                case_output_dir = output_dir / case_name
                case_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save visualization with original filename
                save_full_pipeline_image(
                    img_slice,
                    pred_mask,
                    case_output_dir / f"{filename}.{CONFIG['image_format']}",
                    stats=stats,
                    neg_pos_result=NEG_POS_CLASSES.get(neg_pos_result, "N/A"),
                    neg_pos_conf=neg_pos_confidence,
                    artifact_result=ARTIFACT_CLASSES.get(artifact_result, "N/A"),
                    artifact_conf=artifact_confidence
                )
                
                processed_count += 1
    
    print(f"\n✅ Pipeline completed!")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (Negative): {skipped_negative}")
    print(f"  Skipped (Artifact): {skipped_artifact}")
    print(f"  Total: {processed_count + skipped_negative + skipped_artifact}")
    
    # Generate summary
    print("\n" + "="*60)
    print("📊 Generating Summary Report")
    print("="*60)
    
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    generate_full_pipeline_summary(
        all_filenames,
        all_cases,
        all_stats,
        all_neg_pos_results,
        all_artifact_results,
        summary_dir
    )
    
    print("="*60 + "\n")


# ==================== Visualization ====================

def save_full_pipeline_image(img_slice, pred_mask, save_path, stats=None, 
                              neg_pos_result=None, neg_pos_conf=None,
                              artifact_result=None, artifact_conf=None):
    """
    Save visualization with classification results
    
    Layout: Original | Predicted | Classification + Stats
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original Image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Prediction (Green overlay)
    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(pred_mask, cmap='Greens', alpha=0.5)
    axes[1].set_title('Predicted Lesion', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Classification + Statistics
    info_text = ""
    
    # Classification results
    if neg_pos_result:
        info_text += f"Neg/Pos: {neg_pos_result}\n"
        if neg_pos_conf:
            info_text += f"  Conf: {neg_pos_conf:.4f}\n\n"
    
    if artifact_result:
        info_text += f"Artifact: {artifact_result}\n"
        if artifact_conf:
            info_text += f"  Conf: {artifact_conf:.4f}\n\n"
    
    # Statistics
    if stats is not None:
        if stats.get('skip_reason'):
            info_text += f"Status: Skipped\n"
            info_text += f"Reason: {stats['skip_reason']}"
        else:
            info_text += f"Lesion Area: {stats['lesion_percentage']:.2f}%\n"
            info_text += f"Regions: {stats['num_regions']}\n"
            info_text += f"Largest: {stats['largest_region']} px\n"
            info_text += f"Mean Size: {stats['mean_region_size']:.1f} px"
    
    axes[2].text(0.5, 0.5, info_text,
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.95, edgecolor='black', linewidth=2))
    axes[2].set_title('Classification + Stats', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', format=CONFIG['image_format'])
    plt.close()


# ==================== Summary Report ====================

def generate_full_pipeline_summary(filenames, cases, stats_list, neg_pos_results, artifact_results, summary_dir):
    """Generate comprehensive summary with classification results"""
    
    # Detailed results CSV
    detailed_data = []
    for i, (filename, case, stats, neg_pos, artifact) in enumerate(zip(
        filenames, cases, stats_list, neg_pos_results, artifact_results
    )):
        detailed_data.append({
            'filename': filename,
            'case': case,
            'neg_pos': neg_pos,
            'artifact': artifact,
            'skip_reason': stats.get('skip_reason', None),
            'lesion_percentage': stats['lesion_percentage'],
            'num_regions': stats['num_regions'],
            'largest_region': stats['largest_region'],
            'mean_region_size': stats['mean_region_size']
        })
    
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(summary_dir / 'detailed_results.csv', index=False)
    print(f"✓ Detailed results saved: {summary_dir / 'detailed_results.csv'}")
    
    # Per-case summary
    case_stats = {}
    for case, stats, neg_pos, artifact in zip(cases, stats_list, neg_pos_results, artifact_results):
        if case not in case_stats:
            case_stats[case] = {
                'slices': 0,
                'processed': 0,
                'skipped_negative': 0,
                'skipped_artifact': 0,
                'total_lesion_area': 0.0,
                'total_regions': 0,
                'max_region': 0
            }
        
        case_stats[case]['slices'] += 1
        
        if stats.get('skip_reason') == 'Negative':
            case_stats[case]['skipped_negative'] += 1
        elif stats.get('skip_reason') == 'Artifact':
            case_stats[case]['skipped_artifact'] += 1
        else:
            case_stats[case]['processed'] += 1
            case_stats[case]['total_lesion_area'] += stats['lesion_percentage']
            case_stats[case]['total_regions'] += stats['num_regions']
            case_stats[case]['max_region'] = max(case_stats[case]['max_region'], stats['largest_region'])
    
    case_summary = []
    for case, stats in case_stats.items():
        avg_lesion = stats['total_lesion_area'] / stats['slices'] if stats['slices'] > 0 else 0
        case_summary.append({
            'case': case,
            'total_slices': stats['slices'],
            'processed_slices': stats['processed'],
            'skipped_negative': stats['skipped_negative'],
            'skipped_artifact': stats['skipped_artifact'],
            'avg_lesion_percentage': avg_lesion,
            'total_regions': stats['total_regions'],
            'max_region_size': stats['max_region']
        })
    
    df_case = pd.DataFrame(case_summary)
    df_case.to_csv(summary_dir / 'per_case_summary.csv', index=False)
    print(f"✓ Per-case summary saved: {summary_dir / 'per_case_summary.csv'}")
    
    # Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Neg/Pos Distribution (Pie Chart)
    neg_pos_counts = pd.Series(neg_pos_results).value_counts()
    axes[0, 0].pie(neg_pos_counts.values, labels=neg_pos_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Neg/Pos Distribution', fontweight='bold')
    
    # 2. Artifact Distribution (Pie Chart)
    artifact_counts = pd.Series(artifact_results).value_counts()
    axes[0, 1].pie(artifact_counts.values, labels=artifact_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Artifact Distribution', fontweight='bold')
    
    # 3. Processing Status (Pie Chart)
    status_counts = {
        'Processed': sum(1 for s in stats_list if not s.get('skip_reason')),
        'Skipped (Negative)': sum(1 for s in stats_list if s.get('skip_reason') == 'Negative'),
        'Skipped (Artifact)': sum(1 for s in stats_list if s.get('skip_reason') == 'Artifact')
    }
    axes[0, 2].pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Processing Status', fontweight='bold')
    
    # 4. Lesion Area Distribution (only processed)
    processed_lesions = [s['lesion_percentage'] for s in stats_list if not s.get('skip_reason')]
    if processed_lesions:
        axes[1, 0].hist(processed_lesions, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Lesion Area (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Lesion Area Distribution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Regions per Slice (only processed)
    processed_regions = [s['num_regions'] for s in stats_list if not s.get('skip_reason')]
    if processed_regions:
        axes[1, 1].hist(processed_regions, bins=range(0, max(processed_regions)+2), edgecolor='black', alpha=0.7, color='blue')
        axes[1, 1].set_xlabel('Number of Regions')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Regions Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Per-Case Average Lesion Area
    if len(df_case) > 0:
        sorted_cases = df_case.sort_values('avg_lesion_percentage', ascending=True)
        axes[1, 2].barh(range(len(sorted_cases)), sorted_cases['avg_lesion_percentage'], color='orange')
        axes[1, 2].set_yticks(range(len(sorted_cases)))
        axes[1, 2].set_yticklabels(sorted_cases['case'], fontsize=8)
        axes[1, 2].set_xlabel('Avg Lesion Area (%)')
        axes[1, 2].set_title('Average Lesion Area by Case', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(summary_dir / 'summary_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Summary plots saved: {summary_dir / 'summary_plots.png'}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Slices: {len(filenames)}")
    print(f"  Processed: {status_counts['Processed']}")
    print(f"  Skipped (Negative): {status_counts['Skipped (Negative)']}")
    print(f"  Skipped (Artifact): {status_counts['Skipped (Artifact)']}")
    print(f"\nNeg/Pos Distribution:")
    for label, count in neg_pos_counts.items():
        print(f"  {label}: {count} ({count/len(neg_pos_results)*100:.1f}%)")
    print(f"\nArtifact Distribution:")
    for label, count in artifact_counts.items():
        print(f"  {label}: {count} ({count/len(artifact_results)*100:.1f}%)")
    if processed_lesions:
        print(f"\nLesion Statistics (Processed Only):")
        print(f"  Mean Area: {np.mean(processed_lesions):.2f}%")
        print(f"  Max Area: {np.max(processed_lesions):.2f}%")
        print(f"  Min Area: {np.min(processed_lesions):.2f}%")
    print("="*60)


# ==================== Main ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 DWI Full Pipeline - Classification + Segmentation")
    print("="*60)
    print(f"Segmentation Model: {CONFIG['seg_model_path']}")
    print(f"Neg/Pos Model: {CONFIG['neg_pos_model_path']}")
    print(f"Artifact Model: {CONFIG['artifact_model_path']}")
    print(f"Test Data: {CONFIG['test_data_path']}")
    print(f"Device: {CONFIG['device']}")
    print("="*60 + "\n")
    
    # Load TensorFlow models
    neg_pos_model = None
    artifact_model = None
    
    if CONFIG['use_neg_pos']:
        print(f"Loading Neg/Pos model...")
        try:
            neg_pos_model = tf.keras.models.load_model(CONFIG['neg_pos_model_path'])
            print(f"✓ Neg/Pos model loaded\n")
        except Exception as e:
            print(f"❌ Error loading Neg/Pos model: {e}")
            print("Continuing without Neg/Pos classification...\n")
            CONFIG['use_neg_pos'] = False
    
    if CONFIG['use_artifact']:
        print(f"Loading Artifact model...")
        try:
            artifact_model = tf.keras.models.load_model(CONFIG['artifact_model_path'])
            print(f"✓ Artifact model loaded\n")
        except Exception as e:
            print(f"❌ Error loading Artifact model: {e}")
            print("Continuing without Artifact classification...\n")
            CONFIG['use_artifact'] = False
    
    # Load data
    dataloader = load_full_pipeline_data()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(CONFIG['output_base'])
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Run pipeline
    run_full_pipeline(
        seg_model_path=CONFIG['seg_model_path'],
        neg_pos_model=neg_pos_model,
        artifact_model=artifact_model,
        dataloader=dataloader,
        output_dir=output_dir,
        device=torch.device(CONFIG['device'])
    )
    
    print("\n🎉 Full pipeline completed successfully!")
    print(f"📁 Results saved to: {output_dir}")
