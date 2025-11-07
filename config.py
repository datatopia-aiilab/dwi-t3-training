"""
Configuration file for DWI Ischemic Stroke Segmentation Project
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# ==================== Project Paths ====================
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "1_data_raw"
DATA_PROCESSED = PROJECT_ROOT / "2_data_processed"
MODEL_WEIGHTS = PROJECT_ROOT / "3_model_weights"
RESULTS_DIR = PROJECT_ROOT / "4_results"

# Raw data subdirectories
RAW_IMAGES_DIR = DATA_RAW / "images"
RAW_MASKS_DIR = DATA_RAW / "masks"

# Processed data subdirectories
PROCESSED_TRAIN_IMG = DATA_PROCESSED / "train" / "images"
PROCESSED_TRAIN_MASK = DATA_PROCESSED / "train" / "masks"
PROCESSED_VAL_IMG = DATA_PROCESSED / "val" / "images"
PROCESSED_VAL_MASK = DATA_PROCESSED / "val" / "masks"
PROCESSED_TEST_IMG = DATA_PROCESSED / "test" / "images"
PROCESSED_TEST_MASK = DATA_PROCESSED / "test" / "masks"

# Results subdirectories
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# ==================== Data Parameters ====================
# Image specifications
IMAGE_SIZE = (512, 512)  # (Height, Width) - จะ resize ทุกภาพให้เป็นขนาดนี้
ORIGINAL_SIZE = None  # จะถูกตรวจจับอัตโนมัติจากข้อมูล

# Data split ratio
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Minimum slices per patient (for filtering)
MIN_SLICES_PER_PATIENT = 1  # ตั้งเป็น 1 เพื่อรวมทุก patient (ใช้ zero padding)

# File naming pattern
# Format: Patient_{XXX}_Slice_{YYY}.{extension}
PATIENT_PATTERN = r'Patient_(\d+)_Slice_(\d+)'  # Regex pattern for parsing filenames

# ==================== Preprocessing Parameters ====================
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_ENABLED = False
CLAHE_CLIP_LIMIT = 0.03  # จำกัดการเพิ่ม contrast (ค่าต่ำ = อ่อนโยน, ค่าสูง = แรง)
CLAHE_KERNEL_SIZE = None  # None = auto-calculate based on image size

# Normalization
NORMALIZE_METHOD = 'zscore'  # 'zscore', 'minmax', or 'none'
# Z-score parameters (จะคำนวณจาก training set)
TRAIN_MEAN = None  # จะถูกคำนวณและบันทึกใน preprocessing
TRAIN_STD = None   # จะถูกคำนวณและบันทึกใน preprocessing

# ==================== Model Architecture Parameters ====================
# Input
IN_CHANNELS = 3  # 2.5D input: [N-1, N, N+1] slices

# U-Net architecture
ENCODER_CHANNELS = [64, 128, 256, 512]  # Channels ในแต่ละ layer ของ encoder
DECODER_CHANNELS = [512, 256, 128, 64]  # Channels ในแต่ละ layer ของ decoder
BOTTLENECK_CHANNELS = 1024  # Channels ที่จุดกึ่งกลาง (ลึกที่สุด)

# Output
OUT_CHANNELS = 1  # Binary segmentation (background vs lesion)

# Attention Gate
USE_ATTENTION = True  # เปิด/ปิด Attention Gates

# ==================== Training Parameters ====================
# Basic training settings
NUM_EPOCHS = 200
BATCH_SIZE = 16  # ปรับตาม GPU memory (ถ้า out of memory ให้ลดลง)
NUM_WORKERS = 4  # จำนวน workers สำหรับ DataLoader

# Optimizer
OPTIMIZER = 'adamw'  # 'adam' or 'adamw'
LEARNING_RATE = 3e-5  # ⬇️ ลดลงจาก 1e-4 เพื่อป้องกัน gradient explosion
WEIGHT_DECAY = 1e-5  # L2 regularization

# Gradient clipping (ป้องกัน exploding gradients)
GRADIENT_CLIP_VALUE = 1.0  # Clip gradients ที่มีค่ามากกว่า 1.0

# Learning rate scheduler
SCHEDULER = 'reduce_on_plateau'  # 'reduce_on_plateau' or 'cosine'
SCHEDULER_PATIENCE = 5  # จำนวน epochs ที่รอก่อนลด LR
SCHEDULER_FACTOR = 0.5  # ลด LR เป็น 0.5 เท่า
SCHEDULER_MIN_LR = 1e-7  # LR ต่ำสุด

# Loss function
LOSS_TYPE = 'combo'  # 'focal', 'dice', or 'combo'
FOCAL_ALPHA = 0.25  # Weight for positive class in Focal Loss
FOCAL_GAMMA = 2.0   # Focusing parameter (ยิ่งสูง ยิ่งโฟกัสที่ hard examples)
DICE_SMOOTH = 1e-6  # Smoothing factor for Dice Loss
COMBO_FOCAL_WEIGHT = 0.3  # ⬇️ ลดน้ำหนัก Focal Loss (มักทำให้ไม่เสถียร)
COMBO_DICE_WEIGHT = 0.7   # ⬆️ เพิ่มน้ำหนัก Dice Loss (เสถียรกว่า)

# Early stopping
EARLY_STOPPING_PATIENCE = 15  # หยุดถ้า val dice ไม่ดีขึ้นเป็นเวลา 15 epochs
EARLY_STOPPING_MIN_DELTA = 1e-4  # การเปลี่ยนแปลงขั้นต่ำที่ถือว่า "ดีขึ้น"

# Checkpointing
SAVE_BEST_ONLY = True  # บันทึกเฉพาะโมเดลที่ดีที่สุด
CHECKPOINT_METRIC = 'val_dice'  # Metric ที่ใช้ในการตัดสินใจ
CHECKPOINT_MODE = 'max'  # 'max' (สูงกว่า = ดีกว่า) or 'min' (ต่ำกว่า = ดีกว่า)

# ==================== Data Augmentation Parameters ====================
AUGMENTATION_ENABLED = False

# Augmentation probabilities (0.0 = ไม่ใช้, 1.0 = ใช้ทุกครั้ง)
AUG_HORIZONTAL_FLIP_PROB = 0.5
AUG_VERTICAL_FLIP_PROB = 0.0  # ไม่แนะนำสำหรับ medical images
AUG_ROTATE_PROB = 0.3
AUG_ROTATE_LIMIT = 15  # หมุนได้สูงสุด ±15 องศา

AUG_ELASTIC_TRANSFORM_PROB = 0.4  # สำคัญมาก! จำลองการบิดเบี้ยวของเนื้อเยื่อ
AUG_ELASTIC_ALPHA = 1.0
AUG_ELASTIC_SIGMA = 50.0

AUG_BRIGHTNESS_CONTRAST_PROB = 0.3
AUG_BRIGHTNESS_LIMIT = 0.1
AUG_CONTRAST_LIMIT = 0.1

AUG_GAUSSIAN_NOISE_PROB = 0.2
AUG_GAUSSIAN_NOISE_VAR = (10.0, 50.0)

# ==================== Evaluation Parameters ====================
# Metrics
EVAL_METRICS = ['dice', 'iou', 'precision', 'recall', 'f1']

# Visualization
NUM_QUALITATIVE_SAMPLES = 10  # จำนวนตัวอย่างที่จะแสดงใน evaluation
VIZ_ALPHA = 0.5  # ความโปร่งใสของ mask overlay (0.0-1.0)
VIZ_GT_COLOR = (1.0, 0.0, 0.0)  # สีแดงสำหรับ Ground Truth
VIZ_PRED_COLOR = (0.0, 1.0, 1.0)  # สีฟ้าสำหรับ Prediction

# Threshold for binary mask
PREDICTION_THRESHOLD = 0.5  # Threshold สำหรับแปลง probability เป็น binary mask

# ==================== Hardware Settings ====================
import torch

# Device configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
NUM_GPUS = torch.cuda.device_count() if USE_CUDA else 0

# Mixed precision training (faster training on modern GPUs)
USE_MIXED_PRECISION = True if USE_CUDA else False

# ==================== Logging Settings ====================
LOG_INTERVAL = 10  # พิมพ์ progress ทุกๆ N batches
SAVE_LOG_FILE = True
LOG_FILE = RESULTS_DIR / "training_log.txt"

# Tensorboard
USE_TENSORBOARD = False  # เปิด/ปิด Tensorboard logging
TENSORBOARD_DIR = RESULTS_DIR / "tensorboard"

# ==================== Helper Functions ====================
def create_directories():
    """สร้าง directories ทั้งหมดที่จำเป็น"""
    dirs = [
        DATA_RAW, DATA_PROCESSED, MODEL_WEIGHTS, RESULTS_DIR,
        RAW_IMAGES_DIR, RAW_MASKS_DIR,
        PROCESSED_TRAIN_IMG, PROCESSED_TRAIN_MASK,
        PROCESSED_VAL_IMG, PROCESSED_VAL_MASK,
        PROCESSED_TEST_IMG, PROCESSED_TEST_MASK,
        PLOTS_DIR, PREDICTIONS_DIR
    ]
    
    if USE_TENSORBOARD:
        dirs.append(TENSORBOARD_DIR)
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Created all necessary directories")


def print_config():
    """พิมพ์ configuration สำคัญๆ"""
    print("\n" + "="*60)
    print("🔧 DWI ISCHEMIC STROKE SEGMENTATION - CONFIGURATION")
    print("="*60)
    print(f"\n📁 Paths:")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Raw Data: {DATA_RAW}")
    print(f"   Processed Data: {DATA_PROCESSED}")
    print(f"   Model Weights: {MODEL_WEIGHTS}")
    print(f"   Results: {RESULTS_DIR}")
    
    print(f"\n📊 Data:")
    print(f"   Image Size: {IMAGE_SIZE}")
    print(f"   Train/Val/Test Split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"   Min Slices per Patient: {MIN_SLICES_PER_PATIENT}")
    
    print(f"\n🔬 Preprocessing:")
    print(f"   CLAHE Enabled: {CLAHE_ENABLED}")
    print(f"   CLAHE Clip Limit: {CLAHE_CLIP_LIMIT}")
    print(f"   Normalization: {NORMALIZE_METHOD}")
    
    print(f"\n🏗️ Model:")
    print(f"   Architecture: Attention U-Net (2.5D)")
    print(f"   Input Channels: {IN_CHANNELS}")
    print(f"   Encoder Channels: {ENCODER_CHANNELS}")
    print(f"   Bottleneck: {BOTTLENECK_CHANNELS}")
    print(f"   Use Attention: {USE_ATTENTION}")
    
    print(f"\n🎓 Training:")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Optimizer: {OPTIMIZER.upper()}")
    print(f"   Loss: {LOSS_TYPE.upper()}")
    if LOSS_TYPE == 'combo':
        print(f"      Focal Weight: {COMBO_FOCAL_WEIGHT}, Dice Weight: {COMBO_DICE_WEIGHT}")
    print(f"   Scheduler: {SCHEDULER}")
    print(f"   Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    
    print(f"\n🖼️ Augmentation:")
    print(f"   Enabled: {AUGMENTATION_ENABLED}")
    if AUGMENTATION_ENABLED:
        print(f"   Horizontal Flip: {AUG_HORIZONTAL_FLIP_PROB}")
        print(f"   Rotation: {AUG_ROTATE_PROB} (±{AUG_ROTATE_LIMIT}°)")
        print(f"   Elastic Transform: {AUG_ELASTIC_TRANSFORM_PROB}")
        print(f"   Brightness/Contrast: {AUG_BRIGHTNESS_CONTRAST_PROB}")
    
    print(f"\n💻 Hardware:")
    print(f"   Device: {DEVICE}")
    if USE_CUDA:
        print(f"   GPU(s): {NUM_GPUS} x {torch.cuda.get_device_name(0)}")
        print(f"   Mixed Precision: {USE_MIXED_PRECISION}")
    
    print("\n" + "="*60 + "\n")


def get_model_save_path(name='best_model'):
    """สร้าง path สำหรับบันทึกโมเดล"""
    return MODEL_WEIGHTS / f"{name}.pth"


def get_checkpoint_path(epoch):
    """สร้าง path สำหรับ checkpoint แต่ละ epoch"""
    return MODEL_WEIGHTS / f"checkpoint_epoch_{epoch:03d}.pth"


if __name__ == "__main__":
    # Test configuration
    print_config()
    create_directories()
    print("✅ Configuration loaded successfully!")
