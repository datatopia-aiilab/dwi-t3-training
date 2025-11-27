"""
Configuration file for DWI Artifact Segmentation Training
Simple and minimal - no complexity!
"""

import torch
from pathlib import Path

# ==================== Paths ====================
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

# Data paths - ปรับได้ตามต้องการ
MASKS_DIR = PROJECT_ROOT.parent / "1_data_raw" / "masks"
ORIGINAL_DIR = PROJECT_ROOT.parent / "1_data_raw" / "Original"

# ==================== Data Parameters ====================
IMAGE_SIZE = (384, 384)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.15
TEST_RATIO = 0.05
RANDOM_SEED = 42

# Artifact color in mask (Red: FF0000)
ARTIFACT_COLOR = [255, 0, 0]  # RGB for red

# ==================== Model Architecture ====================
IN_CHANNELS = 3      # RGB input (or grayscale converted to 3 channels)
OUT_CHANNELS = 1     # Binary segmentation
BASE_CHANNELS = 64   # First conv layer channels

# ==================== Training Parameters ====================
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 20

# ==================== Augmentation (Runtime Only) ====================
USE_AUGMENTATION = False  # Master switch: True=Enable, False=Disable all augmentation

# Augmentation parameters (only used if USE_AUGMENTATION=True)
AUG_HFLIP_PROB = 0.3
AUG_ROTATE_LIMIT = 10
AUG_ROTATE_PROB = 0.25
AUG_BRIGHTNESS_LIMIT = 0.1
AUG_BRIGHTNESS_PROB = 0.2

# ==================== MLflow ====================
MLFLOW_EXPERIMENT_NAME = "DWI-Artifact-Baseline"

# ==================== Device ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True if torch.cuda.is_available() else False  # Mixed precision

# ==================== Helper Functions ====================
def create_directories():
    """Create necessary directories"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def print_config():
    """Print configuration summary"""
    print("\n" + "="*60)
    print("📋 DWI ARTIFACT SEGMENTATION - CONFIGURATION")
    print("="*60)
    print(f"Masks Dir: {MASKS_DIR}")
    print(f"Original Dir: {ORIGINAL_DIR}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Split: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")
    print(f"\nTarget: Artifact (Red: RGB{ARTIFACT_COLOR})")
    print(f"\nModel: Attention U-Net")
    print(f"Base Channels: {BASE_CHANNELS}")
    print(f"\nTraining:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Device: {DEVICE}")
    print(f"  Mixed Precision: {USE_AMP}")
    print(f"\nAugmentation: {'ENABLED ✓' if USE_AUGMENTATION else 'DISABLED ✗'}")
    if USE_AUGMENTATION:
        print(f"  Horizontal Flip: {AUG_HFLIP_PROB}")
        print(f"  Rotation ±{AUG_ROTATE_LIMIT}°: {AUG_ROTATE_PROB}")
        print(f"  Brightness: {AUG_BRIGHTNESS_PROB}")
    else:
        print(f"  → No augmentation will be applied (training only)")
    print("="*60 + "\n")

if __name__ == "__main__":
    print_config()
    create_directories()
    print("✅ Configuration loaded successfully!")
