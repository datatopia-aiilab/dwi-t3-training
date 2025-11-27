# 📋 Changelog - Version 2.0

## 🎯 เปลี่ยนแปลงหลัก

### 1. ✅ Visualize ทุก Test Cases (ไม่จำกัดที่ 10-20 รูป)

**ปัญหาเดิม:**
- สร้างเฉพาะ 10 รูปจาก test set (best 3, worst 3, random 4)
- Log เฉพาะ 20 รูปแรกไป MLflow
- ไม่เห็นภาพรวมทั้งหมดของ test set

**การแก้ไข:**

#### ไฟล์ที่แก้ไข:

**1. `evaluation_module.py`:**
```python
# เดิม:
def generate_qualitative_results(..., num_samples=10):

# ใหม่:
def generate_qualitative_results(..., num_samples=None):
    """
    num_samples: int หรือ None
        - ถ้าเป็น None: visualize ทั้งหมด ✅
        - ถ้าเป็น int: เลือก best/worst/random
    """
    if num_samples is None:
        # Visualize ALL samples
        selected_indices = list(range(len(sample_results)))
```

**2. `train.py`:**
```python
# เดิม:
generate_qualitative_results(..., num_samples=10)

# ใหม่:
generate_qualitative_results(..., num_samples=None)  # ⭐ ทั้งหมด
```

**3. `mlflow_utils.py`:**
```python
# เดิม:
log_qualitative_images(images_dir, max_images=20)

# ใหม่:
log_qualitative_images(images_dir, max_images=None)  # ⭐ ไม่จำกัด
```

**ผลลัพธ์:**
- ✅ ถ้า test set มี 62 samples → ได้ครบ 62 รูป
- ✅ สามารถวิเคราะห์ผลทุก case ได้ละเอียด
- ✅ Log ทุกรูปไป MLflow artifacts

---

### 2. ✅ ลบฟังก์ชั่นเวอร์ชั่นเก่า (ใช้เฉพาะเวอร์ชั่นใหม่)

**ปัญหาเดิม:**
- มีฟังก์ชั่นซ้ำซ้อน 2 เวอร์ชั่น:
  - `visualize_sample()` - 3 panels แบบเก่า
  - `visualize_sample_advanced()` - 4 panels แบบใหม่
  - `plot_training_curves()` - 2 subplots แบบเก่า
  - `plot_training_curves_advanced()` - dual y-axis แบบใหม่

**การแก้ไข:**

#### ไฟล์ที่แก้ไข:

**1. `utils.py` - ลบฟังก์ชั่นเก่า:**
```python
# ลบทิ้ง:
❌ def visualize_sample(...):  # 3-panel old version
❌ def plot_training_curves(...):  # 2-subplot old version

# เก็บเฉพาะ:
✅ def visualize_sample_advanced(...):  # 4-panel NEW
✅ def plot_training_curves_advanced(...):  # dual y-axis NEW
```

**2. `evaluate.py` - อัปเดตให้ใช้เวอร์ชั่นใหม่:**
```python
# เดิม:
from utils import plot_training_curves as plot_curves
fig = plot_curves(history)
save_path = save_dir / 'training_curves.png'

# ใหม่:
from utils import plot_training_curves_advanced
fig = plot_training_curves_advanced(history, best_epoch=None)
save_path = save_dir / 'training_curves_advanced.png'  # ⭐ ชื่อใหม่
```

**3. `utils.py` - อัปเดต test function:**
```python
# เดิม:
fig = visualize_sample(dummy_image, dummy_mask, dummy_pred)

# ใหม่:
fig = visualize_sample_advanced(
    dummy_image, dummy_mask, dummy_pred,
    pixel_spacing=4.0, slice_thickness=4.0
)
```

**ผลลัพธ์:**
- ✅ ใช้เฉพาะเวอร์ชั่นใหม่ (4-panel, dual y-axis)
- ✅ ไม่มีความซ้ำซ้อน
- ✅ Code สะอาด maintainable

---

### 3. ✅ ปรับปรุงข้อความแสดงผล

**การแก้ไข:**

**`train.py` - แสดงข้อมูลละเอียดขึ้น:**
```python
print(f"   Prediction Images (4-panel layout with volumes):")
print(f"      ✅ ALL {len(pred_images)} test samples visualized")
print(f"         - Format: 4 panels (Original | GT+Volume | Pred+Volume | Overlap)")
print(f"         - Resolution: 300 DPI")
```

**`evaluation_module.py` - แสดง progress:**
```python
if num_samples is None:
    print(f"   📊 Visualizing ALL {len(sample_results)} test samples...")
else:
    print(f"   📊 Selecting {num_samples} samples (best/worst/random)...")
```

---

## 📊 เปรียบเทียบ Before/After

| Feature | Before (v1) | After (v2) |
|---------|-------------|------------|
| **Test Visualizations** | 10 รูป (selected) | ทั้งหมด (62 รูป) ✅ |
| **MLflow Images** | จำกัด 20 รูป | ไม่จำกัด (ทั้งหมด) ✅ |
| **Visualization Style** | 3 panels (เก่า/ใหม่ ปนกัน) | 4 panels (ใหม่เท่านั้น) ✅ |
| **Training Curves** | 2 subplots (เก่า/ใหม่ ปนกัน) | Dual y-axis (ใหม่เท่านั้น) ✅ |
| **Volume Display** | มี (ใน advanced) | มี ✅ |
| **Overlap Analysis** | มี (ใน advanced) | มี ✅ |
| **Code Cleanliness** | มีฟังก์ชั่นซ้ำซ้อน | สะอาด ไม่ซ้ำซ้อน ✅ |

---

## 🎯 ผลลัพธ์หลังแก้ไข

### เมื่อรัน `python train.py`:

```bash
🖼️  Generating prediction visualizations...
   📊 Visualizing ALL 62 test samples...
   ✅ Generated 62 prediction images in: 4_results/predictions

📦 LOGGING TEST EVALUATION TO MLFLOW
   🖼️  Logging prediction images...
   ✅ Logged 62 prediction images

📁 GENERATED FILES:
   Prediction Images (4-panel layout with volumes):
      ✅ ALL 62 test samples visualized in 4_results/predictions/
         - Format: 4 panels (Original | GT+Volume | Pred+Volume | Overlap)
         - Resolution: 300 DPI
         - Example: patient001_slice05_dice_0.847.png
```

### MLflow Artifacts:

```
mlruns/
└── [experiment_id]/
    └── [run_id]/
        └── artifacts/
            ├── plots/
            │   ├── training_curves_advanced.png  ✅ NEW version only
            │   └── test_metrics_distribution.png
            ├── predictions/
            │   ├── patient001_slice01_dice_0.823.png  ← 4 panels
            │   ├── patient001_slice02_dice_0.845.png
            │   └── ... (ครบ 62 รูป) ✅ ALL samples
            └── evaluation/
                └── test_per_sample_results.csv
```

---

## 🔧 Migration Guide

### สำหรับผู้ใช้งานเดิม:

**1. ถ้ามี script เรียกใช้ `visualize_sample()` เก่า:**
```python
# เปลี่ยนจาก:
from utils import visualize_sample
fig = visualize_sample(image, mask, pred, title="Test")

# เป็น:
from utils import visualize_sample_advanced
fig = visualize_sample_advanced(
    image, mask, pred, 
    filename="Test",
    pixel_spacing=4.0,
    slice_thickness=4.0
)
```

**2. ถ้ามี script เรียกใช้ `plot_training_curves()` เก่า:**
```python
# เปลี่ยนจาก:
from utils import plot_training_curves
fig = plot_training_curves(history)

# เป็น:
from utils import plot_training_curves_advanced
fig = plot_training_curves_advanced(history, best_epoch=None)
```

**3. ถ้าต้องการจำกัดจำนวนรูป (เช่นสำหรับ test ขนาดใหญ่):**
```python
# ใน train.py เปลี่ยนจาก:
num_samples=None  # ทั้งหมด

# เป็น:
num_samples=20  # จำกัด 20 รูป (best/worst/random)
```

---

## ✅ Summary

**ไฟล์ที่แก้ไข:**
1. ✅ `evaluation_module.py` - รองรับ `num_samples=None`
2. ✅ `train.py` - ใช้ `num_samples=None`
3. ✅ `mlflow_utils.py` - ใช้ `max_images=None`
4. ✅ `utils.py` - ลบฟังก์ชั่นเก่า
5. ✅ `evaluate.py` - ใช้เวอร์ชั่นใหม่

**ผลลัพธ์:**
- ✅ Visualize ทุก test case (62/62)
- ✅ Log ทุกรูปไป MLflow
- ✅ ใช้เฉพาะเวอร์ชั่นใหม่ (4-panel, dual y-axis)
- ✅ Code สะอาด ไม่ซ้ำซ้อน
- ✅ พร้อมใช้งาน `python train.py`

**Date:** November 9, 2025
**Version:** 2.0
