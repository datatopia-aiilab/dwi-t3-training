"""
Test By Case - Real World Inference
Test model on real cases without ground truth labels
Flexible folder structure scanning (e.g., A/203/image or B/image)

Usage:
    python testall_bycase.py
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
    'model_path': 'models/best_1/best_model.pth',      # path to trained model
    'test_data_path': '/mnt/d/AiiLAB_PROJECTS/DWI/NovEdition/dwi-t3-training/t3-training-base/validation_dataset',  # folder containing test cases
    'batch_size': 1,                             # process one at a time for real cases
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_base': 'test_results_bycase',        # base folder for results
    'dpi': 300,                                  # PNG resolution (high quality)
    'image_format': 'png',                       # output format
    
    # Image extensions to search
    'image_extensions': ['*.npy', '*.png', '*.jpg', '*.jpeg', '*.nii.gz'],
}


# ==================== Dataset ====================

class RealCaseDataset(Dataset):
    """Dataset for real cases without ground truth"""
    
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
    """
    Recursively find all image files in directory tree
    
    Args:
        root_path: Root directory to search
        extensions: List of file patterns to search for
    
    Returns:
        list of Path objects
    """
    all_files = []
    root = Path(root_path)
    
    if not root.exists():
        raise FileNotFoundError(f"Test data path does not exist: {root}")
    
    print(f"Scanning for images in: {root}")
    
    for ext in extensions:
        # Search recursively
        pattern = f"**/{ext}"
        files = list(root.glob(pattern))
        all_files.extend(files)
        if files:
            print(f"  Found {len(files)} {ext} files")
    
    return sorted(set(all_files))  # Remove duplicates and sort


def load_real_cases():
    """
    Load all images from test_data_path (flexible folder structure)
    Returns: RealCaseDataset
    """
    print("\n" + "="*60)
    print("📂 Loading Real Cases for Inference")
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
                
                continue  # Skip the append below
                
            else:  # PNG, JPG, JPEG
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Failed to load {img_path}, skipping...")
                    skipped += 1
                    continue
                
                img_resized = cv2.resize(img, config.IMAGE_SIZE)
                img_norm = (img_resized - img_resized.mean()) / (img_resized.std() + 1e-8)
                # Use same slice for all 3 channels
                img_3d = np.stack([img_norm, img_norm, img_norm], axis=0)
            
            all_images.append(img_3d)
            all_filenames.append(img_path.stem)
            all_case_names.append(case_name)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            skipped += 1
            continue
    
    # Convert to numpy arrays
    all_images = np.array(all_images, dtype=np.float32)
    
    print(f"\nTotal slices loaded: {len(all_images)}")
    print(f"Skipped: {skipped}")
    print(f"Image shape: {all_images.shape}")
    print(f"Unique cases: {len(set(all_case_names))}")
    print("="*60 + "\n")
    
    # Create and return dataset
    dataset = RealCaseDataset(all_images, all_filenames, all_case_names)
    return dataset


# ==================== Inference ====================

def calculate_stats(pred_mask):
    """
    Calculate statistics about the prediction
    
    Returns:
        dict with statistics
    """
    total_pixels = pred_mask.size
    lesion_pixels = np.sum(pred_mask > 0)
    lesion_percentage = (lesion_pixels / total_pixels) * 100
    
    # Find connected components
    from scipy import ndimage
    labeled, num_features = ndimage.label(pred_mask > 0)
    
    # Calculate sizes of each component
    sizes = []
    if num_features > 0:
        for i in range(1, num_features + 1):
            size = np.sum(labeled == i)
            sizes.append(size)
    
    stats = {
        'lesion_percentage': lesion_percentage,
        'num_regions': num_features,
        'largest_region': max(sizes) if sizes else 0,
        'mean_region_size': np.mean(sizes) if sizes else 0
    }
    
    return stats


def test_real_cases(model_path, dataloader, output_dir, device):
    """
    Run inference on real cases (no ground truth)
    
    Args:
        model_path: Path to trained model checkpoint
        dataloader: DataLoader with test data
        output_dir: Directory to save results
        device: Device to run on (cuda/cpu)
    """
    print("\n" + "="*60)
    print("🔬 Running Inference on Real Cases")
    print("="*60)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = AttentionUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Store results for summary
    all_stats = []
    all_filenames = []
    all_cases = []
    
    with torch.no_grad():
        for idx, (images, filenames, case_names) in enumerate(tqdm(dataloader, desc="Processing")):
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            # Process each item in batch
            for i in range(images.shape[0]):
                # Get data
                img_slice = images[i, 1].cpu().numpy()  # Middle slice (normalized)
                pred_mask = preds[i, 0].cpu().numpy()
                filename = filenames[i]
                case_name = case_names[i]
                
                # Calculate statistics
                stats = calculate_stats(pred_mask)
                
                all_stats.append(stats)
                all_filenames.append(filename)
                all_cases.append(case_name)
                
                # Create case-specific output directory (preserving folder structure)
                case_output_dir = output_dir / case_name
                case_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save visualization with original filename
                save_realcase_image(
                    img_slice,
                    pred_mask,
                    case_output_dir / f"{filename}.{CONFIG['image_format']}",
                    stats=stats,
                    original_filename=filename
                )
    
    print(f"\n✅ All results saved to: {output_dir}")
    
    # Generate and save summary
    print("\n" + "="*60)
    print("📊 Generating Summary Report")
    print("="*60)
    
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    generate_summary_report(
        all_filenames,
        all_cases,
        all_stats,
        summary_dir
    )
    
    print("="*60 + "\n")


# ==================== Visualization ====================

def save_realcase_image(img_slice, pred_mask, save_path, stats=None, original_filename=None):
    """
    Save visualization for real case: Original | Prediction | Metrics
    
    Args:
        img_slice: Original image slice
        pred_mask: Predicted mask
        save_path: Path to save
        stats: Statistics dict
        original_filename: Original filename (for display)
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
    
    # 3. Metrics
    if stats is not None:
        metrics_text = (
            f"Lesion Area: {stats['lesion_percentage']:.2f}%\n"
            f"Regions: {stats['num_regions']}\n"
            f"Largest: {stats['largest_region']} px\n"
            f"Mean Size: {stats['mean_region_size']:.1f} px"
        )
        axes[2].text(0.5, 0.5, metrics_text,
                    ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.95, edgecolor='black', linewidth=2))
        axes[2].set_title('Statistics', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight', format=CONFIG['image_format'])
    plt.close()


# ==================== Summary Report ====================

def generate_summary_report(filenames, cases, stats_list, summary_dir):
    """Generate summary report for all real cases"""
    
    # Create DataFrame
    df = pd.DataFrame({
        'case': cases,
        'filename': filenames,
        'lesion_percentage': [s['lesion_percentage'] for s in stats_list],
        'num_regions': [s['num_regions'] for s in stats_list],
        'largest_region': [s['largest_region'] for s in stats_list],
        'mean_region_size': [s['mean_region_size'] for s in stats_list]
    })
    
    # Save detailed CSV
    csv_path = summary_dir / 'detailed_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Detailed results saved to: {csv_path}")
    
    # Calculate per-case statistics
    case_stats = df.groupby('case').agg({
        'lesion_percentage': ['mean', 'std', 'min', 'max'],
        'num_regions': ['mean', 'sum'],
        'filename': 'count'
    }).round(2)
    
    case_stats.columns = ['_'.join(col).strip() for col in case_stats.columns.values]
    case_stats = case_stats.rename(columns={'filename_count': 'num_slices'})
    
    case_csv_path = summary_dir / 'per_case_summary.csv'
    case_stats.to_csv(case_csv_path)
    print(f"✅ Per-case summary saved to: {case_csv_path}")
    
    # Overall statistics
    print("\n" + "="*60)
    print("📈 OVERALL STATISTICS")
    print("="*60)
    print(f"Total cases: {df['case'].nunique()}")
    print(f"Total slices: {len(df)}")
    print(f"\nLesion Area (%):")
    print(f"  Mean: {df['lesion_percentage'].mean():.2f}%")
    print(f"  Std: {df['lesion_percentage'].std():.2f}%")
    print(f"  Min: {df['lesion_percentage'].min():.2f}%")
    print(f"  Max: {df['lesion_percentage'].max():.2f}%")
    print(f"\nRegions per slice:")
    print(f"  Mean: {df['num_regions'].mean():.2f}")
    print(f"  Max: {df['num_regions'].max():.0f}")
    print("="*60)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Lesion percentage distribution
    axes[0, 0].hist(df['lesion_percentage'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Lesion Percentage (%)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Distribution of Lesion Area', fontweight='bold', fontsize=14)
    axes[0, 0].grid(alpha=0.3)
    
    # Number of regions distribution
    axes[0, 1].hist(df['num_regions'], bins=range(0, int(df['num_regions'].max()) + 2), 
                    color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Number of Regions', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Distribution of Region Count', fontweight='bold', fontsize=14)
    axes[0, 1].grid(alpha=0.3)
    
    # Per-case lesion percentage
    case_means = df.groupby('case')['lesion_percentage'].mean().sort_values(ascending=False)
    axes[1, 0].barh(range(len(case_means)), case_means.values, color='green', alpha=0.7)
    axes[1, 0].set_yticks(range(len(case_means)))
    axes[1, 0].set_yticklabels(case_means.index, fontsize=8)
    axes[1, 0].set_xlabel('Mean Lesion Percentage (%)', fontweight='bold')
    axes[1, 0].set_title('Average Lesion Area by Case', fontweight='bold', fontsize=14)
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # Slices per case
    slices_per_case = df['case'].value_counts().sort_values(ascending=False)
    axes[1, 1].barh(range(len(slices_per_case)), slices_per_case.values, color='purple', alpha=0.7)
    axes[1, 1].set_yticks(range(len(slices_per_case)))
    axes[1, 1].set_yticklabels(slices_per_case.index, fontsize=8)
    axes[1, 1].set_xlabel('Number of Slices', fontweight='bold')
    axes[1, 1].set_title('Slices Processed per Case', fontweight='bold', fontsize=14)
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plot_path = summary_dir / 'summary_plots.png'
    plt.savefig(plot_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary plots saved to: {plot_path}")


# ==================== Main ====================

def main():
    """Main testing function"""
    
    print("="*60)
    print("DWI BASELINE - Test Real Cases (No Labels)")
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
    print(f"📂 Test data: {CONFIG['test_data_path']}")
    print(f"💻 Device: {CONFIG['device']}\n")
    
    # Load all data
    print("Loading test cases...")
    dataset = load_real_cases()
    print(f"✅ Found {len(dataset)} slices from real cases\n")
    
    if len(dataset) == 0:
        print("❌ No data found! Please check test_data_path in CONFIG")
        return
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,  # Single process for simplicity
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )
    
    # Run testing
    test_real_cases(
        model_path=CONFIG['model_path'],
        dataloader=dataloader,
        output_dir=output_dir,
        device=CONFIG['device']
    )


if __name__ == '__main__':
    main()
