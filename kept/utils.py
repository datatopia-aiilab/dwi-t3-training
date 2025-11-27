"""
Utility functions for DWI Ischemic Stroke Segmentation
Contains helper functions for metrics, visualization, and file management
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import defaultdict
import json
from datetime import datetime


# ==================== Metrics Calculation ====================

def calculate_dice_score(pred, target, smooth=1e-6):
    """
    คำนวณ Dice Score (F1 Score for segmentation)
    
    Args:
        pred: Predicted mask (numpy array or tensor)
        target: Ground truth mask (numpy array or tensor)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        dice_score: float (0.0 - 1.0)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


def calculate_iou(pred, target, smooth=1e-6):
    """
    คำนวณ IoU (Intersection over Union / Jaccard Index)
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
    
    Returns:
        iou_score: float (0.0 - 1.0)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def calculate_precision_recall(pred, target, smooth=1e-6):
    """
    คำนวณ Precision และ Recall
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
    
    Returns:
        precision, recall: tuple of floats
    """
    pred = pred.flatten()
    target = target.flatten()
    
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    actual_positive = target.sum()
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    recall = (true_positive + smooth) / (actual_positive + smooth)
    
    return precision, recall


def calculate_f1_score(pred, target, smooth=1e-6):
    """
    คำนวณ F1 Score จาก Precision และ Recall
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor
    
    Returns:
        f1_score: float
    """
    precision, recall = calculate_precision_recall(pred, target, smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    
    return f1


def calculate_all_metrics(pred, target, smooth=1e-6):
    """
    คำนวณ metrics ทั้งหมดในครั้งเดียว
    
    Returns:
        dict: {'dice': float, 'iou': float, 'precision': float, 'recall': float, 'f1': float}
    """
    dice = calculate_dice_score(pred, target, smooth)
    iou = calculate_iou(pred, target, smooth)
    precision, recall = calculate_precision_recall(pred, target, smooth)
    f1 = calculate_f1_score(pred, target, smooth)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ==================== File Management ====================

def parse_filename(filename, pattern=r'Patient_(\d+)_Slice_(\d+)'):
    """
    แยกข้อมูล Patient ID และ Slice Number จากชื่อไฟล์
    
    Args:
        filename: str, ชื่อไฟล์ (เช่น 'Patient_001_Slice_005.npy')
        pattern: str, regex pattern สำหรับการแยกข้อมูล
    
    Returns:
        tuple: (patient_id, slice_number) หรือ (None, None) ถ้าไม่ตรง pattern
    """
    match = re.search(pattern, filename)
    if match:
        patient_id = match.group(1)
        slice_num = int(match.group(2))
        return patient_id, slice_num
    return None, None


def build_slice_mapping(image_folder, pattern=r'Patient_(\d+)_Slice_(\d+)'):
    """
    สร้าง dictionary ที่เก็บข้อมูล slice ข้างเคียงของแต่ละไฟล์
    
    Args:
        image_folder: Path to folder containing images
        pattern: Regex pattern for parsing filenames
    
    Returns:
        dict: {
            'filename.npy': {
                'prev': 'prev_filename.npy' or None,
                'next': 'next_filename.npy' or None,
                'patient': 'patient_id',
                'slice_num': int,
                'index': int (position in patient's slices),
                'total': int (total slices for this patient)
            }
        }
    """
    from pathlib import Path
    
    image_folder = Path(image_folder)
    
    # Group slices by patient
    patient_slices = defaultdict(list)
    
    for filepath in sorted(image_folder.glob('*')):
        if filepath.suffix in ['.npy', '.nii', '.gz', '.png']:
            filename = filepath.name
            patient_id, slice_num = parse_filename(filename, pattern)
            
            if patient_id is not None:
                patient_slices[patient_id].append((filename, slice_num))
    
    # Sort slices for each patient
    for patient_id in patient_slices:
        patient_slices[patient_id] = sorted(patient_slices[patient_id], key=lambda x: x[1])
    
    # Build mapping
    slice_mapping = {}
    
    for patient_id, slices in patient_slices.items():
        total_slices = len(slices)
        
        for idx, (filename, slice_num) in enumerate(slices):
            slice_mapping[filename] = {
                'prev': slices[idx-1][0] if idx > 0 else None,
                'next': slices[idx+1][0] if idx < total_slices - 1 else None,
                'patient': patient_id,
                'slice_num': slice_num,
                'index': idx,
                'total': total_slices
            }
    
    return slice_mapping


def get_patient_statistics(slice_mapping):
    """
    คำนวณสถิติของข้อมูล (จำนวน patients, slices ต่อ patient, etc.)
    
    Args:
        slice_mapping: dict from build_slice_mapping()
    
    Returns:
        dict: สถิติต่างๆ
    """
    patient_slice_counts = defaultdict(int)
    
    for filename, info in slice_mapping.items():
        patient_slice_counts[info['patient']] = info['total']
    
    slice_counts = list(patient_slice_counts.values())
    
    stats = {
        'num_patients': len(patient_slice_counts),
        'num_slices': len(slice_mapping),
        'avg_slices_per_patient': np.mean(slice_counts),
        'min_slices': np.min(slice_counts),
        'max_slices': np.max(slice_counts),
        'median_slices': np.median(slice_counts),
        'std_slices': np.std(slice_counts)
    }
    
    return stats


def filter_patients_by_min_slices(slice_mapping, min_slices=3):
    """
    กรองเฉพาะ patients ที่มีจำนวน slices >= min_slices
    
    Args:
        slice_mapping: dict from build_slice_mapping()
        min_slices: int, จำนวน slices ขั้นต่ำ
    
    Returns:
        list: รายชื่อไฟล์ที่ผ่านการกรอง
    """
    valid_filenames = []
    
    for filename, info in slice_mapping.items():
        if info['total'] >= min_slices:
            valid_filenames.append(filename)
    
    return valid_filenames


# ==================== Visualization ====================
# Note: Old visualize_sample() removed - use visualize_sample_advanced() for 4-panel layout


# ==================== Logging & Saving ====================

def save_training_history(history, filepath):
    """
    บันทึก training history เป็น JSON file
    
    Args:
        history: dict with training metrics
        filepath: Path to save JSON file
    """
    filepath = Path(filepath)
    
    # Convert numpy arrays and handle NaN/inf values for JSON serialization
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            # Convert to list and handle NaN/inf
            value_list = value.tolist()
            value_list = [float(v) if not (np.isnan(v) or np.isinf(v)) else None for v in value_list]
            history_serializable[key] = value_list
        elif isinstance(value, list):
            # Handle NaN/inf in lists
            value_list = []
            for v in value:
                if isinstance(v, (np.floating, float)):
                    if np.isnan(v) or np.isinf(v):
                        value_list.append(None)
                    else:
                        value_list.append(float(v))
                else:
                    value_list.append(v)
            history_serializable[key] = value_list
        else:
            # Handle single values
            if isinstance(value, (np.floating, float)):
                if np.isnan(value) or np.isinf(value):
                    history_serializable[key] = None
                else:
                    history_serializable[key] = float(value)
            else:
                history_serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(history_serializable, f, indent=4)
    
    print(f"✅ Training history saved to {filepath}")


def load_training_history(filepath):
    """
    โหลด training history จาก JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        dict: training history
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        history = json.load(f)
    
    return history


def create_experiment_folder(base_dir, experiment_name=None):
    """
    สร้างโฟลเดอร์สำหรับ experiment พร้อม timestamp
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional name for experiment
    
    Returns:
        Path: path to created experiment folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        folder_name = f"{experiment_name}_{timestamp}"
    else:
        folder_name = f"experiment_{timestamp}"
    
    exp_folder = Path(base_dir) / folder_name
    exp_folder.mkdir(parents=True, exist_ok=True)
    
    return exp_folder


def print_metrics_table(metrics_dict, title="Metrics"):
    """
    พิมพ์ metrics ในรูปแบบตาราง
    
    Args:
        metrics_dict: dict with metric names and values
        title: str, title of the table
    """
    print("\n" + "="*50)
    print(f"📊 {title}")
    print("="*50)
    
    for metric_name, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{metric_name.upper():15s}: {value:.4f}")
        else:
            print(f"{metric_name.upper():15s}: {value}")
    
    print("="*50 + "\n")


# ==================== Advanced Volume & Spatial Metrics ====================

def calculate_infarction_volume(mask, pixel_spacing=4.0, slice_thickness=4.0):
    """
    Calculate infarction volume from binary mask
    
    Args:
        mask: Binary mask (H, W) or (1, H, W)
        pixel_spacing: Pixel spacing in mm (default: 4.0mm)
        slice_thickness: Slice thickness in mm (default: 4.0mm)
    
    Returns:
        volume_ml: Volume in milliliters (ml)
    """
    # Handle shape
    if len(mask.shape) == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
    
    # Count non-zero voxels
    voxel_count = np.sum(mask > 0)
    
    # Calculate volume in mm³
    voxel_volume_mm3 = pixel_spacing * pixel_spacing * slice_thickness
    total_volume_mm3 = voxel_count * voxel_volume_mm3
    
    # Convert to ml (1 ml = 1000 mm³)
    volume_ml = total_volume_mm3 / 1000.0
    
    return volume_ml


def calculate_hausdorff_distance(pred, target, percentile=95):
    """
    Calculate Hausdorff Distance (HD95) between prediction and target
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
        percentile: Percentile for robust HD (default: 95)
    
    Returns:
        hd95: Hausdorff distance at 95th percentile (in pixels)
    """
    try:
        # Handle shape
        if len(pred.shape) == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if len(target.shape) == 3 and target.shape[0] == 1:
            target = target[0]
        
        # Get boundary points
        pred_points = np.argwhere(pred > 0)
        target_points = np.argwhere(target > 0)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')  # No overlap
        
        # Calculate distances
        distances_pred_to_target = np.min(
            np.sqrt(np.sum((pred_points[:, None, :] - target_points[None, :, :]) ** 2, axis=2)),
            axis=1
        )
        distances_target_to_pred = np.min(
            np.sqrt(np.sum((target_points[:, None, :] - pred_points[None, :, :]) ** 2, axis=2)),
            axis=1
        )
        
        # Get 95th percentile
        hd95 = max(
            np.percentile(distances_pred_to_target, percentile),
            np.percentile(distances_target_to_pred, percentile)
        )
        
        return hd95
        
    except Exception as e:
        return None  # Error


def calculate_average_surface_distance(pred, target):
    """
    Calculate Average Surface Distance (ASD)
    
    Args:
        pred: Binary prediction mask
        target: Binary ground truth mask
    
    Returns:
        asd: Average surface distance (in pixels)
    """
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
        
        # Handle shape
        if len(pred.shape) == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if len(target.shape) == 3 and target.shape[0] == 1:
            target = target[0]
        
        # Get boundaries (surface)
        pred_border = pred.astype(bool) ^ binary_erosion(pred.astype(bool))
        target_border = target.astype(bool) ^ binary_erosion(target.astype(bool))
        
        # Distance transforms
        pred_dt = distance_transform_edt(~pred_border)
        target_dt = distance_transform_edt(~target_border)
        
        # Average distances
        pred_surface_distances = pred_dt[target_border]
        target_surface_distances = target_dt[pred_border]
        
        if len(pred_surface_distances) == 0 or len(target_surface_distances) == 0:
            return float('inf')
        
        asd = (pred_surface_distances.sum() + target_surface_distances.sum()) / \
              (len(pred_surface_distances) + len(target_surface_distances))
        
        return asd
        
    except Exception as e:
        return None  # Error


# ==================== Enhanced Visualization ====================

def plot_training_curves_advanced(history, best_epoch=None, save_path=None):
    """
    Plot professional training curves with dual y-axis
    
    Args:
        history: dict with keys ['train_loss', 'val_loss', 'train_dice', 'val_dice']
        best_epoch: int, epoch with best validation score (optional)
        save_path: Path to save the plot (optional)
    
    Returns:
        fig: matplotlib figure
    """
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Colors (professional palette)
    color_train_loss = '#2E86DE'  # Blue
    color_val_loss = '#EE5A6F'    # Red
    color_train_dice = '#26DE81'  # Green
    color_val_dice = '#FD79A8'    # Pink
    
    # Primary y-axis: Loss
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold', color='black')
    
    line1 = ax1.plot(epochs, history['train_loss'], color=color_train_loss, 
                     linewidth=2.5, label='Train Loss', alpha=0.8)
    line2 = ax1.plot(epochs, history['val_loss'], color=color_val_loss, 
                     linewidth=2.5, label='Val Loss', alpha=0.8)
    
    ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Secondary y-axis: Dice Score
    ax2 = ax1.twinx()
    ax2.set_ylabel('Dice Score', fontsize=14, fontweight='bold', color='black')
    
    line3 = ax2.plot(epochs, history['train_dice'], color=color_train_dice, 
                     linewidth=2.5, label='Train Dice', alpha=0.8, linestyle='--')
    line4 = ax2.plot(epochs, history['val_dice'], color=color_val_dice, 
                     linewidth=2.5, label='Val Dice', alpha=0.8, linestyle='--')
    
    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax2.set_ylim([0, 1.0])
    
    # Mark best epoch
    if best_epoch is not None:
        ax1.axvline(x=best_epoch, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax1.text(best_epoch, ax1.get_ylim()[1] * 0.95, f'Best: {best_epoch}', 
                ha='center', va='top', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Combined legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=12, framealpha=0.9)
    
    # Title
    plt.title('Training History: Loss & Dice Score (Combined)', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Advanced training curves (combined) saved to {save_path}")
    
    return fig


def plot_training_curves_separated(history, best_epoch=None, save_path=None):
    """
    Plot professional training curves with SEPARATED subplots (Loss and Dice in separate panels)
    
    Args:
        history: dict with keys ['train_loss', 'val_loss', 'train_dice', 'val_dice']
        best_epoch: int, epoch with best validation score (optional)
        save_path: Path to save the plot (optional)
    
    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Colors (professional palette)
    color_train = '#2E86DE'  # Blue
    color_val = '#EE5A6F'    # Red
    
    # ==================== Plot 1: Loss ====================
    axes[0].plot(epochs, history['train_loss'], color=color_train, 
                linewidth=2.5, label='Train Loss', alpha=0.8, marker='o', markersize=4, markevery=max(1, len(epochs)//20))
    axes[0].plot(epochs, history['val_loss'], color=color_val, 
                linewidth=2.5, label='Val Loss', alpha=0.8, marker='s', markersize=4, markevery=max(1, len(epochs)//20))
    
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(fontsize=12, framealpha=0.9, loc='best')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Mark best epoch
    if best_epoch is not None:
        axes[0].axvline(x=best_epoch, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        y_pos = axes[0].get_ylim()[1] * 0.95
        axes[0].text(best_epoch, y_pos, f'Best: {best_epoch}', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # ==================== Plot 2: Dice Score ====================
    axes[1].plot(epochs, history['train_dice'], color=color_train, 
                linewidth=2.5, label='Train Dice', alpha=0.8, marker='o', markersize=4, markevery=max(1, len(epochs)//20))
    axes[1].plot(epochs, history['val_dice'], color=color_val, 
                linewidth=2.5, label='Val Dice', alpha=0.8, marker='s', markersize=4, markevery=max(1, len(epochs)//20))
    
    axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Dice Score', fontsize=13, fontweight='bold')
    axes[1].set_title('Training and Validation Dice Score', fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(fontsize=12, framealpha=0.9, loc='best')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 1.0])
    
    # Mark best epoch
    if best_epoch is not None:
        axes[1].axvline(x=best_epoch, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        y_pos = 0.95
        axes[1].text(best_epoch, y_pos, f'Best: {best_epoch}', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Overall title
    fig.suptitle('Training History: Loss & Dice Score (Separated)', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves (separated) saved to {save_path}")
    
    return fig


def visualize_sample_advanced(image, mask, prediction, filename="", 
                             pixel_spacing=4.0, slice_thickness=4.0, save_path=None):
    """
    Advanced visualization with 4 panels: Original, GT, Prediction, Overlap
    
    Args:
        image: 2D/2.5D image
        mask: Ground truth mask
        prediction: Predicted mask
        filename: Sample filename
        pixel_spacing: Pixel spacing in mm
        slice_thickness: Slice thickness in mm
        save_path: Path to save figure
    
    Returns:
        fig: matplotlib figure
    """
    # Handle shapes
    if len(image.shape) == 3:
        if image.shape[0] == 3:
            image = image[1, :, :]  # Middle slice
        elif image.shape[-1] == 3:
            image = image[:, :, 1]
        elif image.shape[0] == 1:
            image = image[0, :, :]
    
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask[0, :, :]
    if len(prediction.shape) == 3 and prediction.shape[0] == 1:
        prediction = prediction[0, :, :]
    
    # Normalize image
    image_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Calculate volumes
    gt_volume = calculate_infarction_volume(mask, pixel_spacing, slice_thickness)
    pred_volume = calculate_infarction_volume(prediction, pixel_spacing, slice_thickness)
    volume_error = abs(pred_volume - gt_volume) / (gt_volume + 1e-6) * 100
    
    # Calculate overlap metrics
    tp = np.logical_and(mask > 0, prediction > 0).sum()
    fp = np.logical_and(mask == 0, prediction > 0).sum()
    fn = np.logical_and(mask > 0, prediction == 0).sum()
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Panel 1: Original Image
    axes[0].imshow(image_display, cmap='gray')
    axes[0].set_title('Original DWI', fontsize=13, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # Panel 2: Ground Truth
    axes[1].imshow(image_display, cmap='gray')
    if mask.max() > 0:
        # Red contour
        axes[1].contour(mask, levels=[0.5], colors='red', linewidths=2.5, alpha=0.9)
    axes[1].set_title(f'Ground Truth\nVolume: {gt_volume:.2f} ml', 
                     fontsize=13, fontweight='bold', pad=10, color='#C0392B')
    axes[1].axis('off')
    
    # Panel 3: Prediction
    axes[2].imshow(image_display, cmap='gray')
    if prediction.max() > 0:
        # Blue fill
        pred_overlay = np.zeros((*prediction.shape, 4))
        pred_overlay[prediction > 0] = [0, 0, 1, 0.6]  # Blue with alpha
        axes[2].imshow(pred_overlay)
    axes[2].set_title(f'Prediction\nVolume: {pred_volume:.2f} ml\nError: {volume_error:.1f}%', 
                     fontsize=13, fontweight='bold', pad=10, color='#2980B9')
    axes[2].axis('off')
    
    # Panel 4: Overlap Analysis
    axes[3].imshow(image_display, cmap='gray')
    
    # Create overlay with different colors
    overlap_img = np.zeros((*mask.shape, 4))
    
    # True Positive (both GT and Pred) - Purple
    overlap_img[np.logical_and(mask > 0, prediction > 0)] = [0.5, 0, 0.5, 0.8]
    
    # False Positive (Pred only) - Blue
    overlap_img[np.logical_and(mask == 0, prediction > 0)] = [0, 0, 1, 0.6]
    
    # False Negative (GT only) - Red  
    overlap_img[np.logical_and(mask > 0, prediction == 0)] = [1, 0, 0, 0.6]
    
    axes[3].imshow(overlap_img)
    axes[3].set_title(f'Overlap Analysis\nTP: {tp} | FP: {fp} | FN: {fn}', 
                     fontsize=13, fontweight='bold', pad=10, color='#8E44AD')
    axes[3].axis('off')
    
    # Overall title
    if filename:
        fig.suptitle(f'{filename}', fontsize=15, fontweight='bold', y=0.98)
    
    # Legend for overlap
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.5, 0, 0.5, 0.8], label='True Positive'),
        Patch(facecolor=[0, 0, 1, 0.6], label='False Positive'),
        Patch(facecolor=[1, 0, 0, 0.6], label='False Negative')
    ]
    axes[3].legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_bland_altman_plot(gt_volumes, pred_volumes, save_path=None):
    """
    Create Bland-Altman plot for volume agreement analysis
    
    Args:
        gt_volumes: List of ground truth volumes
        pred_volumes: List of predicted volumes
        save_path: Path to save plot
    
    Returns:
        fig: matplotlib figure
    """
    gt_volumes = np.array(gt_volumes)
    pred_volumes = np.array(pred_volumes)
    
    # Calculate mean and difference
    mean_volumes = (gt_volumes + pred_volumes) / 2
    diff_volumes = pred_volumes - gt_volumes
    
    # Calculate statistics
    mean_diff = np.mean(diff_volumes)
    std_diff = np.std(diff_volumes)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(mean_volumes, diff_volumes, alpha=0.6, s=50, color='#3498DB')
    
    # Mean line
    ax.axhline(mean_diff, color='#E74C3C', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_diff:.2f} ml')
    
    # Limits of agreement
    ax.axhline(mean_diff + 1.96*std_diff, color='#95A5A6', linestyle=':', linewidth=2,
              label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f} ml')
    ax.axhline(mean_diff - 1.96*std_diff, color='#95A5A6', linestyle=':', linewidth=2,
              label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f} ml')
    
    ax.set_xlabel('Mean Volume [(GT + Pred) / 2] (ml)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Difference [Pred - GT] (ml)', fontsize=13, fontweight='bold')
    ax.set_title('Bland-Altman Plot: Volume Agreement Analysis', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Bland-Altman plot saved to {save_path}")
    
    return fig


def create_volume_correlation_plot(gt_volumes, pred_volumes, save_path=None):
    """
    Create scatter plot showing volume correlation
    
    Args:
        gt_volumes: List of ground truth volumes
        pred_volumes: List of predicted volumes
        save_path: Path to save plot
    
    Returns:
        fig: matplotlib figure
    """
    gt_volumes = np.array(gt_volumes)
    pred_volumes = np.array(pred_volumes)
    
    # Calculate correlation
    correlation = np.corrcoef(gt_volumes, pred_volumes)[0, 1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(gt_volumes, pred_volumes, alpha=0.6, s=70, color='#3498DB', edgecolors='black', linewidth=0.5)
    
    # Identity line
    max_val = max(gt_volumes.max(), pred_volumes.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Agreement', alpha=0.7)
    
    # Regression line
    z = np.polyfit(gt_volumes, pred_volumes, 1)
    p = np.poly1d(z)
    ax.plot(gt_volumes, p(gt_volumes), 'g-', linewidth=2, alpha=0.7, 
           label=f'Regression: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax.set_xlabel('Ground Truth Volume (ml)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Volume (ml)', fontsize=13, fontweight='bold')
    ax.set_title(f'Volume Correlation Analysis\nCorrelation: {correlation:.3f}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Volume correlation plot saved to {save_path}")
    
    return fig


# ==================== Testing Functions ====================

def test_utils():
    """ทดสอบ utility functions"""
    print("🧪 Testing Utils Module...\n")
    
    # Test 1: Metrics calculation
    print("Test 1: Metrics Calculation")
    pred = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]])
    target = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 1]])
    
    metrics = calculate_all_metrics(pred, target)
    print_metrics_table(metrics, "Test Metrics")
    
    # Test 2: Filename parsing
    print("Test 2: Filename Parsing")
    test_filenames = [
        "Patient_001_Slice_005.npy",
        "Patient_042_Slice_123.npy",
        "Invalid_filename.npy"
    ]
    
    for filename in test_filenames:
        patient_id, slice_num = parse_filename(filename)
        print(f"{filename:30s} -> Patient: {patient_id}, Slice: {slice_num}")
    
    # Test 3: Visualization (using advanced version)
    print("\nTest 3: Advanced Visualization")
    dummy_image = np.random.rand(3, 256, 256)  # 2.5D format
    dummy_mask = np.random.rand(256, 256) > 0.7
    dummy_pred = np.random.rand(256, 256) > 0.6
    
    fig = visualize_sample_advanced(
        dummy_image, dummy_mask, dummy_pred, 
        filename="Test Sample",
        pixel_spacing=4.0,
        slice_thickness=4.0
    )
    plt.close(fig)
    print("✅ Advanced visualization test passed")
    
    print("\n✅ All utils tests passed!")


if __name__ == "__main__":
    test_utils()
