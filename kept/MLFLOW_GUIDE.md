# 🔬 MLflow Experiment Tracking Guide

## Overview

MLflow has been integrated into the DWI Segmentation project to track all experiments systematically. This allows you to:
- ✅ Compare multiple architectures easily
- ✅ Track all hyperparameters automatically
- ✅ Visualize metrics across experiments
- ✅ Manage model versions
- ✅ Reproduce experiments exactly

---

## 🚀 Quick Start

### 1. Install MLflow

```bash
pip install mlflow>=2.9.0
```

### 2. Train with MLflow (Automatic!)

MLflow tracking is **enabled by default**. Just run training as usual:

```bash
python train.py
```

MLflow will automatically:
- Log all hyperparameters from `config.py`
- Track metrics every epoch (train/val loss, dice, learning rate)
- Save best model checkpoint
- Record training time and system info

### 3. View Results in MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open in browser: **http://localhost:5000**

---

## 🎯 What Gets Tracked

### **Parameters (Hyperparameters)**

Automatically logged from `config.py`:

#### Model Architecture
- `model_architecture`: 'attention_unet', 'unet++', 'fpn', etc.
- `encoder_name`: 'resnet34', 'resnet50', 'efficientnet-b0', etc.
- `encoder_weights`: 'imagenet' or 'None'
- `total_parameters`: Total model parameters
- `model_size_mb`: Model size in MB

#### Training Parameters
- `num_epochs`, `batch_size`, `learning_rate`
- `optimizer`, `weight_decay`, `gradient_clip_value`
- `loss_type`, `focal_alpha`, `focal_gamma`
- `scheduler`, `scheduler_patience`, `scheduler_factor`
- `early_stopping_patience`

#### Data Parameters
- `image_size`, `train_ratio`, `val_ratio`, `test_ratio`
- `normalize_method`, `clahe_enabled`

#### Augmentation Parameters
- `augmentation_enabled`
- `aug_horizontal_flip`, `aug_rotate_prob`, `aug_elastic_prob`
- And all other augmentation settings

#### Hardware
- `device` ('cuda' or 'cpu')
- `gpu_name` (e.g., 'NVIDIA GeForce RTX 3080')
- `num_gpus`, `use_mixed_precision`

---

### **Metrics (Per Epoch)**

Tracked every epoch:

- `train_loss`, `train_dice`
- `val_loss`, `val_dice`
- `learning_rate`
- `epoch_time_seconds`
- `overfitting_gap` (train_dice - val_dice)

**Best Metrics** (Final):
- `best_val_dice` ⭐ (Main metric for comparison)
- `best_epoch`
- `final_train_dice`, `final_val_dice`
- `training_time_minutes`, `training_time_hours`
- `total_epochs_trained`

---

### **Artifacts (Files)**

Saved as artifacts:

1. **Models** (`models/`):
   - `best_model.pth` - Best model checkpoint

2. **History** (`history/`):
   - `training_history.json` - All metrics per epoch

3. **Config** (`config/`):
   - `config.py` - Exact configuration used

4. **Plots** (`plots/`) - If generated:
   - Training curves, predictions, etc.

---

### **Tags**

Auto-tagged for easy filtering:

- `architecture`: 'attention_unet', 'unet++', etc.
- `encoder`: 'resnet34', 'resnet50', etc.
- `pretrained`: 'yes' or 'no'
- `augmentation`: 'enabled' or 'disabled'
- `loss_type`: 'dice', 'focal', 'combo'

---

## 📊 Using MLflow UI

### **Experiments Table**

The main view shows all your runs:

| Run Name | best_val_dice | architecture | encoder | pretrained | train_time |
|----------|---------------|--------------|---------|------------|------------|
| attention_unet_20250108... | 70.07% | attention_unet | - | no | 112 min |
| unet++_resnet34_img_... | 72.15% | unet++ | resnet34 | yes | 95 min |
| deeplabv3+_resnet50_... | 74.82% | deeplabv3+ | resnet50 | yes | 145 min |

### **Comparing Runs**

1. **Select multiple runs** (checkbox)
2. Click **"Compare"** button
3. View:
   - Parameter differences
   - Metric charts (line plots)
   - Side-by-side comparison

### **Filtering Runs**

Use filters to find specific experiments:

```
architecture = "unet++"
pretrained = "yes"
best_val_dice > 0.70
```

### **Sorting Runs**

Click column headers to sort:
- Sort by `best_val_dice` ↓ to find best model
- Sort by `training_time_minutes` ↑ to find fastest

### **Viewing Run Details**

Click on any run to see:
- All parameters
- All metrics (with charts)
- Artifacts (download models, history, etc.)
- System info, tags

---

## 🔧 Configuration

### Enable/Disable MLflow

Edit `config.py`:

```python
# Enable MLflow
MLFLOW_ENABLED = True

# Disable MLflow (no tracking)
MLFLOW_ENABLED = False
```

### Change Experiment Name

```python
MLFLOW_EXPERIMENT_NAME = "DWI_Segmentation"  # Default
MLFLOW_EXPERIMENT_NAME = "DWI_Final_Experiments"  # Custom
```

### Custom Run Name

```python
# Auto-generate (default)
MLFLOW_RUN_NAME = None
# Example: "unet++_resnet34_img_20250108_143022"

# Custom name
MLFLOW_RUN_NAME = "my_experiment_v1"
```

### Change Tracking Location

```python
# Local directory (default)
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

# Remote server
MLFLOW_TRACKING_URI = "http://my-mlflow-server:5000"
```

---

## 📈 Common Workflows

### **Workflow 1: Compare All Architectures**

```bash
# Train each architecture
# Edit config.py: MODEL_ARCHITECTURE = 'attention_unet'
python train.py

# Edit config.py: MODEL_ARCHITECTURE = 'unet++'
python train.py

# Edit config.py: MODEL_ARCHITECTURE = 'fpn'
python train.py

# ... repeat for all 6 architectures

# View results
mlflow ui
```

In UI:
1. Filter by experiment: "DWI_Segmentation"
2. Sort by `best_val_dice` descending
3. Select top 3 runs
4. Click "Compare"
5. View metrics charts side-by-side

### **Workflow 2: Test Pre-trained vs Random Init**

```bash
# Pre-trained
# config.py: ENCODER_WEIGHTS = 'imagenet'
python train.py

# Random init
# config.py: ENCODER_WEIGHTS = None
python train.py

# Compare in MLflow UI
```

Filter: `architecture = "unet++" AND encoder = "resnet34"`

### **Workflow 3: Hyperparameter Tuning**

```bash
# Test different learning rates
# config.py: LEARNING_RATE = 1e-5
python train.py

# config.py: LEARNING_RATE = 5e-5
python train.py

# config.py: LEARNING_RATE = 1e-4
python train.py

# Find best in MLflow UI
```

Sort by `best_val_dice`, compare learning_rate parameter

### **Workflow 4: Reproduce Best Run**

1. In MLflow UI, find best run
2. Click on run → "Artifacts" → "config/config.py"
3. Download config file
4. Replace your config.py with downloaded version
5. Run: `python train.py`

---

## 🎨 Visualizations in MLflow UI

### **Metric Charts**

Automatically generated for each run:
- Training/Validation Loss curves
- Training/Validation Dice curves
- Learning Rate schedule
- Overfitting gap over time

### **Parallel Coordinates Plot**

Compare multiple runs:
- X-axis: Different parameters/metrics
- Lines: Individual runs
- Color: best_val_dice (red = high, blue = low)

Useful for finding parameter patterns.

### **Scatter Plots**

Plot any parameter vs metric:
- X: `learning_rate`
- Y: `best_val_dice`
- Color: `architecture`

---

## 💡 Best Practices

### **1. Use Descriptive Run Names**

For important experiments:
```python
MLFLOW_RUN_NAME = "final_deeplabv3_pretrained_heavy_aug"
```

### **2. Tag Important Runs**

In MLflow UI, add custom tags:
- "production"
- "best_model"
- "for_paper"

### **3. Add Notes**

Click run → "Edit" → Add notes describing:
- What you're testing
- Expected outcome
- Observations

### **4. Archive Old Experiments**

Create new experiment for new project phase:
```python
MLFLOW_EXPERIMENT_NAME = "DWI_Segmentation_V2_Final"
```

### **5. Regular Cleanup**

Delete failed/incomplete runs:
```bash
mlflow gc --backend-store-uri ./mlruns
```

---

## 🐛 Troubleshooting

### **"Cannot connect to MLflow server"**

Solution:
```python
# config.py
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")  # Use local
```

### **"MLflow UI shows no experiments"**

Check:
```bash
ls mlruns/  # Should show experiment folders
```

Start UI with correct path:
```bash
mlflow ui --backend-store-uri ./mlruns
```

### **"Metrics not appearing in UI"**

- Refresh browser (F5)
- Check training didn't crash
- Verify `MLFLOW_ENABLED = True` in config.py

### **"Run name is too long"**

Set custom name:
```python
MLFLOW_RUN_NAME = "short_name"
```

### **"Cannot download artifacts"**

Artifacts saved locally in: `mlruns/<experiment_id>/<run_id>/artifacts/`

---

## 📚 Advanced Features

### **Model Registry** (Optional)

Register best model:
```python
import mlflow
mlflow.register_model(
    f"runs:/{run_id}/models/best_model",
    "DWI_Segmentation_Production"
)
```

### **Remote Tracking Server** (Optional)

For team collaboration:
```bash
# Start server
mlflow server --host 0.0.0.0 --port 5000

# In config.py
MLFLOW_TRACKING_URI = "http://your-server:5000"
```

### **Automatic Tagging**

Add custom tags in `mlflow_utils.py`:
```python
mlflow.set_tag("team_member", "John")
mlflow.set_tag("gpu_type", "RTX_3080")
```

### **Log Additional Metrics**

In `train.py`, add:
```python
from mlflow_utils import log_test_metrics

# After evaluation
test_metrics = {'dice': 0.68, 'iou': 0.52}
log_test_metrics(test_metrics)
```

---

## 📊 Example: View Your Experiments

### **Command Line**

```bash
# Start UI
mlflow ui --backend-store-uri ./mlruns

# Open browser
open http://localhost:5000
```

### **In Browser**

1. **Experiments List** (Left sidebar)
   - Click "DWI_Segmentation"

2. **Runs Table** (Main view)
   - Shows all training runs
   - Sort by `best_val_dice`

3. **Select Best Run** (Click row)
   - View all parameters
   - View metric charts
   - Download artifacts

4. **Compare Multiple Runs**
   - Check 3+ runs
   - Click "Compare"
   - See parameter differences
   - View overlaid metric charts

---

## 🎯 Key Metrics to Track

For DWI Segmentation, focus on:

1. **Primary**: `best_val_dice` (validation performance)
2. **Generalization**: `overfitting_gap` (should be small)
3. **Efficiency**: `training_time_minutes`
4. **Final**: `test_dice` (when you add evaluation)

**Target**: Find model with highest `best_val_dice` and lowest `overfitting_gap`

---

## ✅ Verification Checklist

After running your first experiment:

- [ ] MLflow UI starts: `mlflow ui`
- [ ] Experiment "DWI_Segmentation" exists
- [ ] Run appears in experiments table
- [ ] Run has auto-generated name (e.g., "attention_unet_20250108...")
- [ ] All parameters are logged (60+ parameters)
- [ ] Metrics charts show training curves
- [ ] Artifacts include best_model.pth
- [ ] Can download model checkpoint
- [ ] Can compare multiple runs

---

## 🎉 Success!

You now have complete experiment tracking! Every time you run `python train.py`, everything is automatically logged to MLflow.

**Next Steps:**
1. Train multiple architectures
2. Compare results in MLflow UI
3. Find best hyperparameters
4. Reproduce best model
5. Deploy to production

**Questions?** Check:
- MLflow docs: https://mlflow.org/docs/latest/index.html
- Project README.md
- `mlflow_utils.py` source code

---

**Happy Experimenting! 🚀**
