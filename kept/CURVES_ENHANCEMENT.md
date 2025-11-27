# 📊 Training Curves Enhancement - Dual Format

## 🎯 สิ่งที่เพิ่มเข้ามา

เพิ่มฟังก์ชั่นสร้าง **training curves แบบแยกกราฟ** เพื่อให้มีทั้งหมด **2 แบบ**:

### 1. **Combined (Dual Y-axis)** - แบบเดิม ปรับปรุงแล้ว
- Loss และ Dice อยู่ในกราฟเดียว
- ใช้ 2 แกน Y (Loss ซ้าย, Dice ขวา)
- ดูภาพรวมได้ในครั้งเดียว
- ชื่อไฟล์: `training_curves_combined.png`

### 2. **Separated (2 Subplots)** - ✨ ใหม่!
- Loss และ Dice แยกกราฟ
- แต่ละกราฟมีพื้นที่เต็ม
- ดูรายละเอียดได้ชัดเจนขึ้น
- มี markers ทุก 5% ของ epochs
- ชื่อไฟล์: `training_curves_separated.png`

---

## 🔧 การเปลี่ยนแปลง

### 1. **utils.py** - เพิ่มฟังก์ชั่นใหม่

#### เพิ่ม `plot_training_curves_separated()`:
```python
def plot_training_curves_separated(history, best_epoch=None, save_path=None):
    """
    Plot professional training curves with SEPARATED subplots
    - 2 subplots: Loss (left) | Dice (right)
    - Each subplot has full space for details
    - Markers every 5% of epochs
    - Best epoch marked with vertical line
    - Size: 18×6 inches, 300 DPI
    """
```

**Features:**
- ✅ 2 subplots แนวนอน (18×6 นิ้ว)
- ✅ Colors: Blue (Train), Red (Val)
- ✅ Markers: 'o' (Train), 's' (Val)
- ✅ Grid, legend, best epoch marker
- ✅ 300 DPI publication quality

#### ปรับปรุง `plot_training_curves_advanced()`:
```python
# เปลี่ยนชื่อใน title:
- 'Training History: Loss & Dice Score'
+ 'Training History: Loss & Dice Score (Combined)'
```

---

### 2. **train.py** - สร้างทั้ง 2 แบบ

#### เปลี่ยนจาก:
```python
plot_training_curves_advanced(history, best_epoch, save_path)
```

#### เป็น:
```python
# Version 1: Combined
curves_combined_path = cfg.PLOTS_DIR / 'training_curves_combined.png'
plot_training_curves_advanced(history, best_epoch, curves_combined_path)

# Version 2: Separated
curves_separated_path = cfg.PLOTS_DIR / 'training_curves_separated.png'
plot_training_curves_separated(history, best_epoch, curves_separated_path)

print(f"   ✅ Combined version: {curves_combined_path.name}")
print(f"   ✅ Separated version: {curves_separated_path.name}")
```

#### อัปเดตการ log MLflow:
```python
log_training_complete(
    ...,
    best_model_path, history_path,
    curves_combined_path, curves_separated_path  # ⭐ 2 paths
)
```

---

### 3. **mlflow_utils.py** - รองรับ 2 curves

#### เปลี่ยนจาก:
```python
def log_training_complete(..., curves_path=None):
```

#### เป็น:
```python
def log_training_complete(..., 
                         curves_combined_path=None, 
                         curves_separated_path=None):
    # Log both curves
    if curves_combined_path and Path(curves_combined_path).exists():
        mlflow.log_artifact(str(curves_combined_path), artifact_path="plots")
        print(f"      ✅ Combined: {curves_combined_path.name}")
    
    if curves_separated_path and Path(curves_separated_path).exists():
        mlflow.log_artifact(str(curves_separated_path), artifact_path="plots")
        print(f"      ✅ Separated: {curves_separated_path.name}")
```

---

### 4. **evaluate.py** - สร้างทั้ง 2 แบบ

```python
# Version 1: Combined
fig_combined = plot_training_curves_advanced(history, best_epoch=None)
save_path_combined = save_dir / 'training_curves_combined.png'
fig_combined.savefig(save_path_combined, dpi=300, bbox_inches='tight')

# Version 2: Separated
fig_separated = plot_training_curves_separated(history, best_epoch=None)
save_path_separated = save_dir / 'training_curves_separated.png'
fig_separated.savefig(save_path_separated, dpi=300, bbox_inches='tight')
```

---

## 📊 Output เมื่อรัน

### เมื่อรัน `python train.py`:

```bash
📊 Generating training curves...
✅ Combined training curves (combined) saved to training_curves_combined.png
✅ Training curves (separated) saved to training_curves_separated.png
   ✅ Combined version: training_curves_combined.png
   ✅ Separated version: training_curves_separated.png

======================================================================
📦 LOGGING TRAINING ARTIFACTS TO MLFLOW
======================================================================

   📊 Logging training curves...
      ✅ Combined: training_curves_combined.png
      ✅ Separated: training_curves_separated.png

📁 GENERATED FILES:
   Training Curves:
      ✅ Combined (dual y-axis): training_curves_combined.png
      ✅ Separated (2 subplots): training_curves_separated.png
```

---

## 📦 MLflow Artifacts

```
mlruns/
└── [experiment_id]/
    └── [run_id]/
        └── artifacts/
            ├── plots/
            │   ├── training_curves_combined.png    ⭐ NEW - dual y-axis
            │   ├── training_curves_separated.png   ⭐ NEW - 2 subplots
            │   └── test_metrics_distribution.png
            ├── predictions/
            │   └── *.png (62 images, 4-panel)
            └── evaluation/
                └── test_per_sample_results.csv
```

---

## 📋 เปรียบเทียบ 2 แบบ

| Feature | Combined (Dual Y-axis) | Separated (2 Subplots) |
|---------|------------------------|------------------------|
| **Size** | 15×6 inches | 18×6 inches |
| **Panels** | 1 panel | 2 panels |
| **Y-axes** | 2 (Loss left, Dice right) | 1 per panel |
| **Colors** | Blue, Red, Green, Pink | Blue (Train), Red (Val) |
| **Markers** | No markers | Yes ('o' and 's') |
| **Best for** | Quick overview | Detailed analysis |
| **Use case** | Presentations | Publications |

---

## 🎨 Visualization Details

### Combined Version:
- **Loss (Left Y-axis):**
  - Train Loss: Solid blue line
  - Val Loss: Solid red line
- **Dice (Right Y-axis):**
  - Train Dice: Dashed green line
  - Val Dice: Dashed pink line
- **Best epoch:** Gray vertical dotted line

### Separated Version:
- **Left subplot (Loss):**
  - Train: Blue line with 'o' markers
  - Val: Red line with 's' markers
  - Y-axis: Loss values
- **Right subplot (Dice):**
  - Train: Blue line with 'o' markers
  - Val: Red line with 's' markers
  - Y-axis: 0-1.0 (fixed)
- **Best epoch:** Gray vertical dotted line (both panels)

---

## ✅ Summary

**ไฟล์ที่แก้ไข:**
1. ✅ `utils.py` - เพิ่ม `plot_training_curves_separated()`
2. ✅ `train.py` - สร้างทั้ง 2 แบบ + log ทั้งคู่
3. ✅ `mlflow_utils.py` - รองรับ 2 curves paths
4. ✅ `evaluate.py` - สร้างทั้ง 2 แบบ

**ผลลัพธ์:**
- ✅ มี 2 เวอร์ชั่นของ training curves
- ✅ ทั้งคู่บันทึกใน local และ MLflow
- ✅ แต่ละเวอร์ชั่นมีจุดเด่นของตัวเอง
- ✅ ใช้งานได้ทันที: `python train.py`

**Date:** November 9, 2025
**Version:** 2.1 - Dual Format Training Curves
