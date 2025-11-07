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

def visualize_sample(image, mask, prediction=None, title="", 
                    alpha=0.5, gt_color=(1.0, 0.0, 0.0), pred_color=(0.0, 1.0, 1.0)):
    """
    แสดงภาพ original, ground truth, และ prediction (ถ้ามี)
    
    Args:
        image: 2D image (H, W) หรือ (H, W, 3)
        mask: Ground truth mask (H, W)
        prediction: Predicted mask (H, W) - optional
        title: str, หัวข้อของรูป
        alpha: float, ความโปร่งใสของ mask overlay
        gt_color: tuple, สี RGB สำหรับ ground truth (0.0-1.0)
        pred_color: tuple, สี RGB สำหรับ prediction (0.0-1.0)
    
    Returns:
        fig: matplotlib figure
    """
    # ถ้า image เป็น 3 channels (2.5D) ให้เอาแค่ channel กลาง
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = image[:, :, 1]  # Middle slice (N)
    
    # Normalize image to 0-1 for display
    image_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # จำนวน subplot
    num_plots = 3 if prediction is not None else 2
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
    
    # Plot 1: Original Image
    axes[0].imshow(image_display, cmap='gray')
    axes[0].set_title('Original DWI', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot 2: Ground Truth Overlay
    axes[1].imshow(image_display, cmap='gray')
    
    # Create color overlay for ground truth
    if mask.max() > 0:
        mask_overlay = np.zeros((*mask.shape, 3))
        mask_overlay[mask > 0] = gt_color
        axes[1].imshow(mask_overlay, alpha=alpha)
    
    axes[1].set_title('Ground Truth (Red)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Plot 3: Prediction Overlay (if provided)
    if prediction is not None:
        axes[2].imshow(image_display, cmap='gray')
        
        # Create color overlay for prediction
        if prediction.max() > 0:
            pred_overlay = np.zeros((*prediction.shape, 3))
            pred_overlay[prediction > 0] = pred_color
            axes[2].imshow(pred_overlay, alpha=alpha)
        
        axes[2].set_title('Prediction (Cyan)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    return fig


def plot_training_curves(history, save_path=None):
    """
    พล็อตกราฟ Loss และ Dice Score ตลอดการเทรน
    
    Args:
        history: dict with keys ['train_loss', 'val_loss', 'train_dice', 'val_dice']
        save_path: Path to save the plot (optional)
    
    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Dice Score
    axes[1].plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    axes[1].plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to {save_path}")
    
    return fig


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
    
    # Test 3: Visualization
    print("\nTest 3: Visualization")
    dummy_image = np.random.rand(256, 256)
    dummy_mask = np.random.rand(256, 256) > 0.7
    dummy_pred = np.random.rand(256, 256) > 0.6
    
    fig = visualize_sample(dummy_image, dummy_mask, dummy_pred, title="Test Sample")
    plt.close(fig)
    print("✅ Visualization test passed")
    
    print("\n✅ All utils tests passed!")


if __name__ == "__main__":
    test_utils()
