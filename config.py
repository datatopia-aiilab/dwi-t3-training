"""
Configuration file for DWI Ischemic Stroke Segmentation Project
VERSION 3 (Revised Blueprint - Data Minimalist)

Changes:
1.  Respects the data: DISABLED CLAHE and AUGMENTATION (based on empirical results).
2.  Changed LOSS_TYPE to 'focal_tversky' to target False Negatives (faint lesions).
3.  Reverted LEARNING_RATE to a safer 3e-5 to stabilize the new, volatile loss.
4.  Suggests Deep Supervision (to be implemented in model.py and train.py).
"""

import os
import torch
from pathlib import Path

# ==================== Project Paths ====================
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "1_data_raw"
DATA_PROCESSED = PROJECT_ROOT / "2_data_processed"
MODEL_WEIGHTS = PROJECT_ROOT / "3_model_weights"
RESULTS_DIR = PROJECT_ROOT / "4_results"

# ... (Paths for processed data remain the same) ...
PROCESSED_TRAIN_IMG = DATA_PROCESSED / "train" / "images"
PROCESSED_TRAIN_MASK = DATA_PROCESSED / "train" / "masks"
PROCESSED_VAL_IMG = DATA_PROCESSED / "val" / "images"
PROCESSED_VAL_MASK = DATA_PROCESSED / "val" / "masks"
PROCESSED_TEST_IMG = DATA_PROCESSED / "test" / "images"
PROCESSED_TEST_MASK = DATA_PROCESSED / "test" / "masks"

PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# ==================== Data Parameters ====================
IMAGE_SIZE = (256, 256)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.20
TEST_RATIO = 0.0566
RANDOM_SEED = 42
MIN_SLICES_PER_PATIENT = 1
PATIENT_PATTERN = r'Patient_(\d+)_Slice_(\d+)'

# ==================== Preprocessing Parameters ====================
# ⬇️ ACTION: Keep CLAHE DISABLED. Your data rejects it.
CLAHE_ENABLED = False
CLAHE_CLIP_LIMIT = 0.01
CLAHE_KERNEL_SIZE = (8, 8)

NORMALIZE_METHOD = 'zscore'
TRAIN_MEAN = None
TRAIN_STD = None

# ==================== Model Architecture Parameters ====================
IN_CHANNELS = 3
ENCODER_CHANNELS = [64, 128, 256, 512]
DECODER_CHANNELS = [512, 256, 128, 64]
BOTTLENECK_CHANNELS = 1024
OUT_CHANNELS = 1
USE_ATTENTION = True

# ⬇️ ACTION: Implement Deep Supervision in model.py
# This will force the model to return multiple outputs.
USE_DEEP_SUPERVISION = True
DEEP_SUPERVISION_WEIGHTS = [1.0, 0.4, 0.2] # [main, 1/2 size, 1/4 size]

# ==================== Training Parameters ====================
NUM_EPOCHS = 300
BATCH_SIZE = 16
NUM_WORKERS = 4

OPTIMIZER = 'adamw'
# ⬇️ ACTION: Revert to a stable, lower LR. The new loss is volatile.
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-5

GRADIENT_CLIP_VALUE = 1.0
SCHEDULER = 'reduce_on_plateau'
# ⬇️ ACTION: Increase patience. The new loss needs time.
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-7

# ⬇️ ACTION: This is the most critical change.
LOSS_TYPE = 'focal_tversky' # (You must implement this in loss.py)
TVERSKY_ALPHA = 0.3  # Penalize FP (background) less
TVERSKY_BETA = 0.7   # Penalize FN (faint lesion) more
FOCAL_TVERSKY_GAMMA = 0.75 # Focus on hard examples
DICE_SMOOTH = 1e-6

EARLY_STOPPING_PATIENCE = 35
EARLY_STOPPING_MIN_DELTA = 1e-4

CHECKPOINT_METRIC = 'val_dice' # We still monitor Dice
CHECKPOINT_MODE = 'max'
SAVE_BEST_ONLY = True

# ==================== Data Augmentation Parameters ====================
# ⬇️ ACTION: Keep Augmentation DISABLED. Your data rejects it.
AUGMENTATION_ENABLED = False
AUG_HORIZONTAL_FLIP_PROB = 0.5
AUG_ROTATE_PROB = 0.3
AUG_ELASTIC_TRANSFORM_PROB = 0.0
AUG_BRIGHTNESS_CONTRAST_PROB = 0.3

# ==================== Evaluation Parameters ====================
EVAL_METRICS = ['dice', 'iou', 'precision', 'recall', 'f1']
NUM_QUALITATIVE_SAMPLES = 10
VIZ_ALPHA = 0.5
VIZ_GT_COLOR = (1.0, 0.0, 0.0)
VIZ_PRED_COLOR = (0.0, 1.0, 1.0)
PREDICTION_THRESHOLD = 0.5

# ==================== Hardware Settings ====================
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
NUM_GPUS = torch.cuda.device_count() if USE_CUDA else 0
USE_MIXED_PRECISION = True if USE_CUDA else False

# ==================== Logging Settings ====================
LOG_INTERVAL = 10
SAVE_LOG_FILE = True
LOG_FILE = RESULTS_DIR / "training_log_v3.txt" # New log file
USE_TENSORBOARD = False
TENSORBOARD_DIR = RESULTS_DIR / "tensorboard_v3"

# ==================== Helper Functions ====================
# ... (create_directories, print_config, etc. remain the same) ...

def print_config():
    """Prints key configuration settings"""
    print("\n" + "="*60)
    print("🔧 DWI ISCHEMIC STROKE SEGMENTATION - CONFIG (V3 - Data Minimalist)")
    print("="*60)
    print(f"\n🔬 Preprocessing & Augmentation:")
    print(f"    CLAHE Enabled: {CLAHE_ENABLED} (DISABLED)")
    print(f"    Augmentation Enabled: {AUGMENTATION_ENABLED} (DISABLED)")
    print(f"\n🏗️ Model:")
    print(f"    Architecture: Attention U-Net (2.5D)")
    print(f"    Deep Supervision: {USE_DEEP_SUPERVISION}")
    print(f"\n🎓 Training:")
    print(f"    Learning Rate: {LEARNING_RATE} (Stable)")
    print(f"    Loss: {LOSS_TYPE.upper()} (Targeting FNs)")
    print(f"        Tversky Alpha (FP): {TVERSKY_ALPHA}")
    print(f"        Tversky Beta (FN):  {TVERSKY_BETA}")
    print(f"        Focal Gamma: {FOCAL_TVERSKY_GAMMA}")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print_config()
    # create_directories()
    print("✅ V3 Configuration loaded successfully!")