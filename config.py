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
# Note: PREDICTIONS_DIR will be updated to run-specific subfolder during training
# e.g., predictions/run_a1b2c3d4/ to separate predictions from different training runs
# This prevents mixing predictions from multiple experiments

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
RANDOM_SEED = 5

# Minimum slices per patient (for filtering)
MIN_SLICES_PER_PATIENT = 1  # ตั้งเป็น 1 เพื่อรวมทุก patient (ใช้ zero padding)

# File naming pattern
# Format: Patient_{XXX}_Slice_{YYY}.{extension}
PATIENT_PATTERN = r'Patient_(\d+)_Slice_(\d+)'  # Regex pattern for parsing filenames

# ==================== Preprocessing Parameters ====================

# ==================== N4 Bias Field Correction ====================
# N4 ITK algorithm corrects intensity inhomogeneity in MRI images
# Benefits:
#   - Fixes "bias field" (varying brightness across image)
#   - Improves lesion visibility
#   - Makes normalization more effective
#   - Standard in medical imaging preprocessing
# 
# ⚠️ Requirements: pip install SimpleITK
# ⚠️ Processing time: ~5-30 seconds per image (depends on shrink_factor)
#
# Expected improvement: +3-6% Dice score

N4_ENABLED = True  # ✅ Enable N4 bias field correction (HIGHLY RECOMMENDED)
                   # Set to False to skip N4 correction

# N4 Shrink Factor
# Controls speed vs quality tradeoff
# - shrink_factor=1: Best quality, slowest (~30s per image)
# - shrink_factor=2: Good quality, moderate speed (~15s per image)
# - shrink_factor=4: Good quality, fast (~5-10s per image) ⭐ RECOMMENDED
# - shrink_factor=8: Lower quality, very fast (~2-5s per image)
N4_SHRINK_FACTOR = 4  # Recommended: 4 (good balance)

# N4 Number of Iterations
# More iterations = better correction but slower
# - 50: Fast, good for most cases ⭐ RECOMMENDED
# - 100: Better correction, slower
# - 200: Best correction, very slow (usually unnecessary)
N4_NUM_ITERATIONS = 50  # Recommended: 50

# N4 Multiprocessing
# Number of parallel workers for N4 correction
# - None or 0: Auto-detect (use all available CPUs)
# - 1: Sequential processing (slowest)
# - 4-8: Good balance for most systems ⭐ RECOMMENDED
# - N: Use N parallel workers
N4_NUM_WORKERS = 4  # Recommended: 4-8

# ==================== CLAHE (Contrast Limited Adaptive Histogram Equalization) ====================
CLAHE_ENABLED = False  # ⬇️ ปิด CLAHE เพราะทำให้ผลแย่ลง (55% vs 72%)
                       # Note: N4 correction is usually better than CLAHE for MRI
CLAHE_CLIP_LIMIT = 0.03  # จำกัดการเพิ่ม contrast (ค่าต่ำ = อ่อนโยน, ค่าสูง = แรง)
CLAHE_KERNEL_SIZE = None  # None = auto-calculate based on image size

# ==================== Normalization ====================
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
#   'attention_unet'    - Custom Attention U-Net (current baseline, 17.5M params)
#   'attention_unet_ds' - Attention U-Net with Deep Supervision ⭐ NEW (improves +2-4% Dice)
#   'unet++'            - U-Net++ with nested skip connections (~20M params)
#   'fpn'               - Feature Pyramid Network (~25M params)
#   'deeplabv3+'        - DeepLabV3+ with ASPP (~40M params) ⚠️ NOT compatible with DenseNet!
#   'manet'             - Multi-Attention Network (~22M params)
#   'pspnet'            - Pyramid Scene Parsing Network (~45M params)

MODEL_ARCHITECTURE = 'attention_unet_ds'  # ⭐ Use Deep Supervision for better training
                                          # Change to 'attention_unet' to disable deep supervision
                               # DeepLabV3+ ใช้กับ DenseNet ไม่ได้เพราะต้องการ dilated convolutions

# ⚠️ COMPATIBILITY NOTE:
# DeepLabV3+ requires dilated mode → Only works with: ResNet, ResNeXt, EfficientNet, SE-ResNet
# DenseNet uses pooling for downsampling → Cannot be dilated
# 
# If you want DeepLabV3+, use these encoders:
#   - resnet34, resnet50, resnet101
#   - resnext50_32x4d
#   - efficientnet-b0, efficientnet-b3, efficientnet-b5
#   - se_resnet50, se_resnext50_32x4d
#
# If you want DenseNet, use these architectures:
#   - unet++ ⭐ (แนะนำ - nested skip connections)
#   - fpn (feature pyramid)
#   - manet (multi-attention)
#   - pspnet (pyramid pooling)

# ==================== Encoder Selection (for SMP models) ====================
# Available encoders (when using unet++, fpn, deeplabv3+, manet, pspnet):
#
# 🔥 POPULAR CLASSICS (ดังและนิยม):
#   'vgg16'          - VGG-16 (~15M params, simple & effective, 2014)
#   'vgg19'          - VGG-19 (~20M params, deeper than VGG-16)
#   'densenet121'    - DenseNet-121 (~8M params, dense connections, memory efficient)
#   'densenet169'    - DenseNet-169 (~14M params, deeper)
#   'densenet201'    - DenseNet-201 (~20M params, deepest)
#   'inceptionv4'    - Inception-v4 (~42M params, multi-scale, 2016)
#   'inceptionresnetv2' - Inception-ResNet-v2 (~55M params, hybrid architecture)
#   'xception'       - Xception (~23M params, depthwise separable convolutions)
#
# 🚀 MODERN EFFICIENT (ประสิทธิภาพสูง):
#   'resnet34'       - ResNet-34 (~21M params, balanced) ⭐
#   'resnet50'       - ResNet-50 (~25M params, more capacity) ⭐
#   'resnet101'      - ResNet-101 (~44M params, very deep)
#   'resnext50_32x4d' - ResNeXt-50 (~25M params, strong)
#   'efficientnet-b0' - EfficientNet-B0 (~5M params, most efficient) ⭐
#   'efficientnet-b1' - EfficientNet-B1 (~7M params)
#   'efficientnet-b2' - EfficientNet-B2 (~9M params)
#   'efficientnet-b3' - EfficientNet-B3 (~12M params, powerful)
#   'efficientnet-b4' - EfficientNet-B4 (~19M params)
#   'efficientnet-b5' - EfficientNet-B5 (~30M params)
#
# 🎯 ATTENTION-BASED (มี attention mechanisms):
#   'senet154'       - SENet-154 (~115M params, heavy but powerful)
#   'se_resnet50'    - SE-ResNet-50 (~28M params, ResNet + SE blocks)
#   'se_resnext50_32x4d' - SE-ResNeXt-50 (~27M params)
#   'timm-efficientnet-b5' - EfficientNet-B5 from timm (~30M params)
#
# 📊 SIZE COMPARISON:
#   Small  (<10M):  efficientnet-b0, densenet121, mobilenet_v2
#   Medium (10-30M): resnet34, resnet50, vgg16, densenet169
#   Large  (>30M):   resnet101, inceptionv4, inceptionresnetv2, senet154
#
# 💡 RECOMMENDATIONS:
#   - Small dataset (<1000): efficientnet-b0, densenet121, resnet34
#   - Medium dataset (1000-5000): resnet50, densenet169, efficientnet-b3
#   - Large dataset (>5000): resnet101, efficientnet-b5, inceptionresnetv2
#   - Medical imaging: densenet121 (dense connections), se_resnet50 (attention)

ENCODER_NAME = 'senet154'  # Default encoder for SMP models

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

# ==================== Additional Attention Mechanisms ====================
# These can be applied to ANY architecture (attention_unet, unet++, fpn, deeplabv3+, manet, pspnet)
# Multiple can be enabled simultaneously

# Squeeze-and-Excitation (SE) Block
USE_SE_ATTENTION = True  # ✅ Very lightweight (~0.1M params), channel attention
                          # Best for: Quick improvement with minimal overhead
                          # Expected gain: +1-2% Dice

# Convolutional Block Attention Module (CBAM)
USE_CBAM_ATTENTION = False  # ✅ Channel + Spatial attention (~0.5M params)
                            # Best for: Balanced spatial and channel focus
                            # Expected gain: +2-3% Dice

# Efficient Channel Attention (ECA)
USE_ECA_ATTENTION = False  # ✅ More efficient than SE, no reduction
                           # Best for: Better than SE, still lightweight
                           # Expected gain: +1-2% Dice

# Dual Attention (Position + Channel)
USE_DUAL_ATTENTION = False  # ⚠️ Expensive (~2-3M params), applied at bottleneck only
                            # Best for: Long-range dependencies, medical imaging
                            # Expected gain: +3-5% Dice
                            # Note: Only applied to layers with >=256 channels

# Multi-Scale Attention
USE_MULTISCALE_ATTENTION = False  # ⚠️ Medium cost (~1-2M params)
                                  # Best for: Varying lesion sizes
                                  # Expected gain: +2-4% Dice
                                  # Note: Only applied to layers with >=128 channels

# Attention combination strategy
# 'sequential' - Apply selected attentions one after another
# 'parallel' - Apply selected attentions in parallel and sum outputs
ATTENTION_COMBINATION = 'sequential'  # 'sequential' or 'parallel'

# ==================== Deep Supervision Parameters ====================
# Deep Supervision adds auxiliary outputs at intermediate decoder layers
# Benefits:
#   - Better gradient flow to early layers
#   - Multi-scale supervision
#   - Faster convergence
#   - Improved final performance
#
# Only applies when MODEL_ARCHITECTURE = 'attention_unet_ds'
# Expected improvement: +2-4% Dice score

# Enable/Disable Deep Supervision
USE_DEEP_SUPERVISION = True  # Automatically True if using 'attention_unet_ds'
                             # Set to False to disable loss from auxiliary outputs

# Number of auxiliary supervision levels
# Recommended: 3 (matches number of decoder blocks minus final)
# Range: 1-4 (depending on network depth)
DEEP_SUPERVISION_LEVELS = 3

# Loss weights for deep supervision
# Format: [main_output, aux_level1, aux_level2, aux_level3, ...]
# Default: [1.0, 0.5, 0.25, 0.125] - exponentially decreasing
# These will be normalized internally so their sum doesn't change loss magnitude
DEEP_SUPERVISION_WEIGHTS = [1.0, 0.5, 0.25, 0.125]

# ==================== Training Parameters ====================
# Basic training settings
NUM_EPOCHS = 100  # ⬇️⬇️ ลดจาก 200 → 100 เพื่อให้ LR decay เร็วขึ้น (แก้ plateau)
BATCH_SIZE = 16  # ปรับตาม GPU memory (ถ้า out of memory ให้ลดลง)
NUM_WORKERS = 4  # จำนวน workers สำหรับ DataLoader

# Regularization
DROPOUT = 0.0  # Dropout probability for ConvBlocks (default: 0.0)
               # Increase to 0.1-0.2 if severe overfitting occurs
               # Note: We use other regularization methods (augmentation, weight decay)

# Optimizer
OPTIMIZER = 'adamw'  # 'adam' or 'adamw'
LEARNING_RATE = 8e-5  # ค่าที่ดีอยู่แล้ว
WEIGHT_DECAY = 2e-4  # ⬆️⬆️ เพิ่มจาก 8e-5 → 2e-4 เพื่อลด overfitting (Train 0.90 vs Val 0.66)

# Gradient clipping (ป้องกัน exploding gradients)
GRADIENT_CLIP_VALUE = 0.5  # ⬇️ ลดลงจาก 1.0 → 0.5 เพื่อ stability กับ attention
                            # ถ้ายังเจอ NaN ให้ลดเป็น 0.1

# Learning rate scheduler
# Available: 'cosine', 'reduce_on_plateau', 'warmup_cosine', 'cosine_restarts', 
#            'one_cycle', 'polynomial', 'adaptive', 'exponential'
SCHEDULER = 'polynomial'  # ⭐⭐ เปลี่ยนเป็น polynomial เพื่อ decay เร็วกว่า cosine
                          # Warmup_cosine กับ 200 epochs ทำให้ LR ลดช้าเกิน
                          # → Epoch 71 ยัง LR 0.000059 (ลดแค่ 26%)
                          # Polynomial จะ decay แบบ quadratic เร็วกว่า

# Scheduler parameters (used by different schedulers)
SCHEDULER_PATIENCE = 12  # For 'reduce_on_plateau', 'adaptive'
SCHEDULER_FACTOR = 0.5  # LR reduction factor
SCHEDULER_MIN_LR = 1e-7  # Minimum LR for all schedulers

# Warmup settings (for 'warmup_cosine', 'cosine_restarts')
WARMUP_EPOCHS = 5  # Number of warmup epochs (0 = no warmup)
                   # Recommended: 5-10 for attention, 0-3 for standard models

# Cosine Restarts settings (for 'cosine_restarts')
FIRST_CYCLE_EPOCHS = 50  # Length of first cycle
CYCLE_MULT = 1  # Multiply cycle length: 1=same, 2=double each time

# Polynomial decay settings (for 'polynomial')
POLY_POWER = 2.0  # Polynomial power (1.0=linear, 2.0=quadratic)

# Exponential decay settings (for 'exponential')
EXP_GAMMA = 0.95  # Decay factor per epoch (0.9-0.99)

# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULER STRATEGY GUIDE
# ═══════════════════════════════════════════════════════════════════════════
# 
# PROBLEM: NaN loss at epoch 101 with CBAM attention
# CAUSE: Attention mechanisms amplify gradients → instability with fixed LR
# 
# SOLUTION STRATEGY (3-layer defense):
# ------------------------------------
# 1. NUMERICAL STABILITY (✅ Already implemented):
#    - BatchNorm in all attention modules
#    - Gradient clipping (GRADIENT_CLIP_VALUE=0.5)
#    - Residual scaling (scale=0.1 in CBAM/SE)
#    - Output clamping before sigmoid
# 
# 2. ADVANCED LR SCHEDULING (✅ Just added):
#    - Warmup phase prevents early instability
#    - Adaptive strategies handle loss spikes
#    - Multiple options for different scenarios
# 
# 3. MONITORING & RECOVERY (✅ Available):
#    - nan_detector.py for debugging
#    - AdaptiveScheduler auto-recovers from NaN
#    - Early stopping prevents waste
# 
# RECOMMENDED CONFIGURATIONS:
# ---------------------------
# 
# For Attention Models (Current Setup):
#   SCHEDULER = 'warmup_cosine'
#   WARMUP_EPOCHS = 5
#   LEARNING_RATE = 0.001
#   → Best stability, smooth convergence
# 
# If Still Getting NaN:
#   SCHEDULER = 'adaptive'
#   GRADIENT_CLIP_VALUE = 0.1  # More aggressive
#   → Auto-handles loss spikes
# 
# For Faster Training:
#   SCHEDULER = 'onecycle'
#   LEARNING_RATE = 0.01  # Higher than normal
#   NUM_EPOCHS = 100  # Fewer epochs needed
#   → Super-convergence approach
# 
# If Stuck at Plateau:
#   SCHEDULER = 'cosine_restarts'
#   FIRST_CYCLE_EPOCHS = 30
#   CYCLE_MULT = 2
#   → Periodic LR resets escape local minima
# 
# For Fine-Tuning:
#   SCHEDULER = 'polynomial'
#   POLY_POWER = 2.0
#   LEARNING_RATE = 0.0001  # Lower
#   → Smooth, predictable decay
# 
# TESTING RESULTS:
# ----------------
# Before: NaN at epoch 101 (Val Dice: 0.5479)
# After:  Testing with warmup_cosine + stability fixes
#         Expected: Stable training to epoch 200+
# ═══════════════════════════════════════════════════════════════════════════

# Loss function
# Available options:
#   - 'dice': Standard Dice Loss (stable, recommended baseline)
#   - 'focal': Focal Loss (good for class imbalance)
#   - 'combo': Combo of Focal + Dice (powerful but can be unstable)
#   - 'logcosh_dice': Log-Cosh Dice Loss (smoother than dice) ⭐ NEW
#   - 'combo_logcosh_dice': Combo of Focal + Log-Cosh Dice (stable + powerful) ⭐ NEW RECOMMENDED
#   - 'tversky': Tversky Loss (for recall/precision weighting)
#   - 'bce_dice': BCE + Dice combination
LOSS_TYPE = 'combo_logcosh_dice'  # ⭐ NEW: More stable than 'combo', smoother gradients
FOCAL_ALPHA = 0.25  # Weight for positive class in Focal Loss
FOCAL_GAMMA = 2.0   # Focusing parameter
DICE_SMOOTH = 1e-6  # Smoothing factor for Dice Loss
COMBO_FOCAL_WEIGHT = 0.3  # น้ำหนัก Focal Loss
COMBO_DICE_WEIGHT = 0.7   # น้ำหนัก Dice Loss


# Early stopping
EARLY_STOPPING_PATIENCE = 30  # ⬇️⬇️ ลดจาก 100 → 30 เพราะ epochs ลดลง และไม่อยากเสียเวลารอ
EARLY_STOPPING_MIN_DELTA = 5e-4  # ⬆️ เพิ่มจาก 1e-4 เพื่อให้ต้อง improve จริง ๆ

# Checkpointing
SAVE_BEST_ONLY = True  # บันทึกเฉพาะโมเดลที่ดีที่สุด
CHECKPOINT_METRIC = 'val_dice'  # Metric ที่ใช้ในการตัดสินใจ
CHECKPOINT_MODE = 'max'  # 'max' (สูงกว่า = ดีกว่า) or 'min' (ต่ำกว่า = ดีกว่า)

# ==================== Data Augmentation Parameters ====================
# ⚠️⚠️⚠️ CRITICAL FIX สำหรับ OVERFITTING ⚠️⚠️⚠️
# ปัญหาหลัก: Train Dice 0.90 vs Val Dice 0.66 → Gap 0.26 (overfitting รุนแรง)
# สาเหตุ: Dataset เล็ก (640 train) + Model ใหญ่ (20M params) + ไม่มี augmentation
# วิธีแก้: เปิด augmentation เพื่อเพิ่ม effective dataset size จาก 640 → ~1,920
AUGMENTATION_ENABLED = True  # ⬆️⬆️⬆️ เปลี่ยนเป็น True! (สำคัญมาก)

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

# ==================== Gamma Correction Augmentation ====================
# Gamma correction simulates intensity variations in MRI scans
# Benefits:
#   - Improves model robustness to intensity variations
#   - Helps with images from different scanners/protocols
#   - Simulates different contrast settings
#
# gamma < 1.0: Brightens image (more visible dark regions)
# gamma > 1.0: Darkens image (compress bright regions)
# gamma = 1.0: No change
#
# Expected improvement: +1-2% Dice score

AUG_GAMMA_PROB = 0.25  # 25% chance of applying gamma correction
AUG_GAMMA_LIMIT = (80, 120)  # Gamma range: (0.8, 1.2)
                              # (80, 120) in albumentations = gamma * 100
                              # Reasonable range for medical images

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

# ==================== Test-Time Augmentation (TTA) Parameters ====================
# TTA significantly improves prediction quality without retraining
# Expected improvement: +2-4% Dice score

# Enable/Disable TTA
USE_TTA = True  # Set to False for faster inference (no accuracy gain)

# TTA augmentations to apply
# Options: 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'
# More augmentations = slower but potentially better results
TTA_AUGMENTATIONS = ['hflip', 'vflip']  # Recommended: hflip + vflip (2x slower, good balance)
# TTA_AUGMENTATIONS = ['hflip', 'vflip', 'rot90', 'rot180']  # 5x slower, maximum quality

# ==================== Connected Component Analysis (CCA) Parameters ====================
# CCA removes small false positive regions and improves precision
# Expected improvement: +5-8% Precision, slight Dice improvement

# Enable/Disable CCA cleaning
USE_CCA = True  # Set to False to disable post-processing

# Minimum component size (pixels)
# Components smaller than this will be removed as noise
# Adjust based on minimum expected lesion size
CCA_MIN_SIZE = 10  # pixels (~1.6mm² with 4mm spacing)
                   # Smaller value = keep more small lesions but more noise
                   # Larger value = fewer false positives but may miss small lesions

# Minimum average confidence (probability)
# Components with mean probability below this will be removed
CCA_MIN_CONFIDENCE = 0.3  # 0.0-1.0 (0.3 = 30% confidence)
                          # Lower = keep more uncertain predictions
                          # Higher = only keep high-confidence predictions

# Maximum number of components to keep (None = keep all valid components)
CCA_MAX_COMPONENTS = None  # None or int
                          # Useful if you expect only N lesions per slice

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

# MLflow Run Isolation
# แต่ละ training run จะมี:
# 1. Unique run_id (e.g., a1b2c3d4e5f6g7h8)
# 2. Separate prediction folder: predictions/run_{run_id[:8]}/
# 3. Independent metrics, parameters, and artifacts in MLflow
# 4. No cross-contamination between runs
# → แต่ละ run เป็นอิสระ ไม่มีการเก็บข้อมูลข้าม run กัน

# MLflow Tags (จะถูกเพิ่มใน run automatically)
# - architecture: MODEL_ARCHITECTURE
# - encoder: ENCODER_NAME (for SMP models)
# - pretrained: "yes" / "no"
# - augmentation: "enabled" / "disabled"
# - loss_type: LOSS_TYPE
# - run_id: Unique identifier for this training run

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
