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
IMAGE_SIZE = (384, 384)  # (Height, Width) - จะ resize ทุกภาพให้เป็นขนาดนี้
ORIGINAL_SIZE = None  # จะถูกตรวจจับอัตโนมัติจากข้อมูล

# Data split ratio
# Test: ตายตัว 48 slices (~5.66%)
# ที่เหลือ 800: Train 80% (640) + Val 20% (160)
TRAIN_RATIO = 0.80  # 80% ของ (total - test) = 640 slices
VAL_RATIO = 0.20    # 20% ของ (total - test) = 160 slices  
TEST_RATIO = 0.06 # ตายตัว 48 slices (~5.66% ของ total)

# Random seed for reproducibility
RANDOM_SEED = 10

# Minimum slices per patient (for filtering)
MIN_SLICES_PER_PATIENT = 1  # ตั้งเป็น 1 เพื่อรวมทุก patient (ใช้ zero padding)

# File naming pattern
# Format: Patient_{XXX}_Slice_{YYY}.{extension}
PATIENT_PATTERN = r'Patient_(\d+)_Slice_(\d+)'  # Regex pattern for parsing filenames

# ==================== Preprocessing Parameters ====================
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_ENABLED = False  # ⬇️ ปิด CLAHE เพราะทำให้ผลแย่ลง (55% vs 72%)
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

# Output
OUT_CHANNELS = 1  # Binary segmentation (background vs lesion)

# ==================== Architecture Selection ====================
# Available architectures:
#   'attention_unet' - Custom Attention U-Net (current baseline, 17.5M params)
#   'unet++'         - U-Net++ with nested skip connections (~20M params)
#   'fpn'            - Feature Pyramid Network (~25M params)
#   'deeplabv3+'     - DeepLabV3+ with ASPP (~40M params)
#   'manet'          - Multi-Attention Network (~22M params)
#   'pspnet'         - Pyramid Scene Parsing Network (~45M params)

MODEL_ARCHITECTURE = 'unet++'  # เปลี่ยนตรงนี้เพื่อใช้ architecture อื่น

# ==================== Encoder Selection (for SMP models) ====================
# Available encoders (when using unet++, fpn, deeplabv3+, manet, pspnet):
#   'resnet34'       - ResNet-34 (~21M params, balanced)
#   'resnet50'       - ResNet-50 (~25M params, more capacity)
#   'efficientnet-b0' - EfficientNet-B0 (~5M params, efficient)
#   'efficientnet-b3' - EfficientNet-B3 (~12M params, powerful)
#   'resnext50_32x4d' - ResNeXt-50 (~25M params, strong)
#   'timm-efficientnet-b5' - EfficientNet-B5 from timm (~30M params)

ENCODER_NAME = 'efficientnet-b0'  # Default encoder for SMP models

# Pre-trained weights
ENCODER_WEIGHTS = 'imagenet'  # Options: 'imagenet' (pre-trained), None (random init)

# ==================== Custom U-Net Architecture (for attention_unet only) ====================
# Round 3 (Baseline): [64,128,256,512] → 31M → Val 72%, Test 56%, Gap 16%
# Round 6 (Small): [32,64,128,256] → 7.8M → Val 61% (underfitting)
# Round 7 (Medium, no aug): [48,96,192,384] → 17.5M → Val 67%, Test 53%
# Round 8 (Medium + light aug): [48,96,192,384] → Val 69%, Test 53% ⭐ Best balance
# Round 9 (Large + heavy reg): [64,128,256,512] → Val 64%, Test 55% (underfitting)
# Round 10 (Medium + optimized): [48,96,192,384] → Val 70%, Test 62% ⭐ **BEST**
ENCODER_CHANNELS = [48, 96, 192, 384]  # For attention_unet only
DECODER_CHANNELS = [384, 192, 96, 48]  # For attention_unet only
BOTTLENECK_CHANNELS = 768  # For attention_unet only

# Attention Gate
USE_ATTENTION = True  # เปิด/ปิด Attention Gates (for attention_unet only)

# ==================== Training Parameters ====================
# Basic training settings
NUM_EPOCHS = 200  # ⬇️ ลดลงจาก 250 (ไม่ต้องการ train นานเกินไป)
BATCH_SIZE = 4  # ปรับตาม GPU memory (ถ้า out of memory ให้ลดลง)
NUM_WORKERS = 4  # จำนวน workers สำหรับ DataLoader

# Optimizer
OPTIMIZER = 'adamw'  # 'adam' or 'adamw'
LEARNING_RATE = 8e-5  # ⬆️⬆️ เพิ่มขึ้นอีก (จาก 3e-5) เพื่อให้เรียนรู้เร็วขึ้น
WEIGHT_DECAY = 8e-5  # ⬇️ ลดลง (จาก 2e-4) ให้อยู่กึ่งกลาง 1e-5 กับ 1e-4

# Gradient clipping (ป้องกัน exploding gradients)
GRADIENT_CLIP_VALUE = 1.0  # Clip gradients ที่มีค่ามากกว่า 1.0

# Learning rate scheduler
SCHEDULER = 'cosine'  # 'reduce_on_plateau' or 'cosine'
SCHEDULER_PATIENCE = 12  # ⬆️ เพิ่มขึ้น (เพื่อให้มีเวลาเรียนรู้มากขึ้น)
SCHEDULER_FACTOR = 0.5  # ลด LR เป็น 0.5 เท่า
SCHEDULER_MIN_LR = 1e-7  # LR ต่ำสุด

# Loss function  
LOSS_TYPE = 'dice'  # ⬇️ กลับไปใช้ Dice (Combo ทำ NaN แม้ LR ต่ำ + Gamma 1.5)
FOCAL_ALPHA = 0.25  # Weight for positive class in Focal Loss
FOCAL_GAMMA = 2.0   # Focusing parameter
DICE_SMOOTH = 1e-6  # Smoothing factor for Dice Loss
COMBO_FOCAL_WEIGHT = 0.3  # น้ำหนัก Focal Loss
COMBO_DICE_WEIGHT = 0.7   # น้ำหนัก Dice Loss


# Early stopping
EARLY_STOPPING_PATIENCE = 100  # ⬇️ ลดลงจาก 40 (ไม่ต้องรอนาน)
EARLY_STOPPING_MIN_DELTA = 1e-4  # การเปลี่ยนแปลงขั้นต่ำที่ถือว่า "ดีขึ้น"

# Checkpointing
SAVE_BEST_ONLY = True  # บันทึกเฉพาะโมเดลที่ดีที่สุด
CHECKPOINT_METRIC = 'val_dice'  # Metric ที่ใช้ในการตัดสินใจ
CHECKPOINT_MODE = 'max'  # 'max' (สูงกว่า = ดีกว่า) or 'min' (ต่ำกว่า = ดีกว่า)

# ==================== Data Augmentation Parameters ====================
AUGMENTATION_ENABLED = True  # ⬆️ คงไว้ (Round 8 ใช้ได้ดี)

# Augmentation - กลับไปใช้ค่า Round 8 ที่ balanced ดี
AUG_HORIZONTAL_FLIP_PROB = 0.3  # ⬇️ กลับเป็น 0.3 (Round 8)
AUG_VERTICAL_FLIP_PROB = 0.0  # ไม่แนะนำสำหรับ medical images
AUG_ROTATE_PROB = 0.25  # ลดลงเล็กน้อย (จาก 0.3)
AUG_ROTATE_LIMIT = 10  # คง ±10°

AUG_ELASTIC_TRANSFORM_PROB = 0.15  # ⬇️ ลดลง (จาก 0.2) แต่ยังเปิดใช้
AUG_ELASTIC_ALPHA = 0.5  # คง 0.5 (อ่อน)
AUG_ELASTIC_SIGMA = 50.0

AUG_BRIGHTNESS_CONTRAST_PROB = 0.2  # คง 0.2
AUG_BRIGHTNESS_LIMIT = 0.08  # คง 0.08
AUG_CONTRAST_LIMIT = 0.08  # คง 0.08

AUG_GAUSSIAN_NOISE_PROB = 0.12  # ⬇️ ลดลงเล็กน้อย (จาก 0.15)
AUG_GAUSSIAN_NOISE_VAR = (5.0, 22.0)  # ลดลงเล็กน้อย (จาก 25)

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

# ==================== MLflow Settings ====================
# MLflow - Experiment Tracking & Model Registry
MLFLOW_ENABLED = True  # เปิด/ปิด MLflow tracking
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")  # Local tracking directory
MLFLOW_EXPERIMENT_NAME = f"DWI-NOV-{MODEL_ARCHITECTURE}"  # ชื่อ experiment
MLFLOW_RUN_NAME = None  # None = auto-generate (e.g., "unet++_resnet34_20250108_143022")

# MLflow Tags (จะถูกเพิ่มใน run automatically)
# - architecture: MODEL_ARCHITECTURE
# - encoder: ENCODER_NAME (for SMP models)
# - pretrained: "yes" / "no"
# - augmentation: "enabled" / "disabled"
# - loss_type: LOSS_TYPE

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
    print(f"   Architecture: {MODEL_ARCHITECTURE.upper()}")
    print(f"   Input Channels: {IN_CHANNELS}")
    print(f"   Output Channels: {OUT_CHANNELS}")
    
    if MODEL_ARCHITECTURE == 'attention_unet':
        print(f"   Encoder Channels: {ENCODER_CHANNELS}")
        print(f"   Bottleneck: {BOTTLENECK_CHANNELS}")
        print(f"   Use Attention: {USE_ATTENTION}")
    else:
        print(f"   Encoder: {ENCODER_NAME}")
        print(f"   Pre-trained: {ENCODER_WEIGHTS or 'None (random init)'}")
    
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
