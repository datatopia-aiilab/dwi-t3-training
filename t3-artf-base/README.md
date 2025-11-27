# DWI Artifact Segmentation - Simple & Clean

โปรเจคสำหรับ DWI Artifact Segmentation (Red Color: FF0000)
**เน้นความเรียบง่าย ไม่ซับซ้อน ทำงานได้แน่นอน!**

---

## 📁 โครงสร้าง

```
t3-artf-base/
├── config.py          # การตั้งค่าทั้งหมด
├── model.py           # Attention U-Net (simple)
├── train.py           # All-in-one: preprocess + train + evaluate
├── README.md          # คู่มือนี้
│
├── models/            # (auto-created) - temporary storage
│   └── best_model.pth
│
└── mlruns/            # (auto-created) - MLflow tracking
    └── 0/
        └── <run_id>/
            ├── params/
            ├── metrics/
            └── artifacts/
                ├── training_curve.png
                ├── test_predictions/
                ├── best_model.pth
                └── test_metrics.json
```

---

## 🚀 วิธีใช้งาน

### 1. เตรียม Environment

```bash
# ติดตั้ง dependencies
pip install -r requirements.txt
```

### 2. เตรียมข้อมูล

โครงสร้างข้อมูล:
```
../1_data_raw/
├── masks/             # Mask PNG files (red artifacts: RGB 255,0,0)
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── Original/          # Original PNG images
    ├── image_001.png
    ├── image_002.png
    └── ...
```

**หมายเหตุ:**
- ชื่อไฟล์ใน `masks/` และ `Original/` **ต้องตรงกัน**
- Artifact ใน mask ต้องเป็นสีแดง (RGB: 255, 0, 0)
- ระบบจะอ่านชื่อจาก `masks/` ก่อน แล้วค้นหาใน `Original/`

### 3. รัน Training

```bash
cd t3-artf-base
python train.py
```

**จะทำอะไรบ้าง:**
1. อ่านชื่อไฟล์จาก `masks/` folder
2. หารูปที่ตรงกันจาก `Original/` folder
3. Extract binary mask จาก red artifacts (FF0000)
4. Preprocess (resize, normalize)
5. แบ่งข้อมูล train/val/test (80/15/5)
6. Train Attention U-Net (100 epochs หรือจน early stop)
7. Evaluate บน test set
8. Log ทุกอย่างไปยัง MLflow

### 4. ดูผลลัพธ์

```bash
# เปิด MLflow UI
mlflow ui --port 5000

# เปิดเบราว์เซอร์
http://localhost:5000
```

---

## 📊 สิ่งที่ MLflow จะเก็บ

### Parameters:
- `image_size`: (384, 384)
- `batch_size`: 16
- `learning_rate`: 0.0001
- `artifact_color`: [255, 0, 0]
- `model_params`: ~31M
- และอื่นๆ

### Metrics (ทุก epoch):
- `train_loss`, `train_dice`
- `val_loss`, `val_dice`
- `learning_rate`

### Test Metrics:
- `test_dice_mean`, `test_dice_std`
- `test_iou_mean`, `test_iou_std`

### Artifacts:
- `training_curve.png` - กราฟ loss + dice
- `test_predictions/` - 10 ตัวอย่างการทำนาย
- `best_model.pth` - โมเดลที่ดีที่สุด
- `test_metrics.json` - สรุปผลการประเมิน

---

## ⚙️ การตั้งค่า

แก้ไขใน `config.py`:

```python
# Data
IMAGE_SIZE = (384, 384)
ARTIFACT_COLOR = [255, 0, 0]  # Red artifacts

# Model
BASE_CHANNELS = 64

# Training
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 20

# Augmentation
USE_AUGMENTATION = False  # True/False
```

---

## 🎯 ความแตกต่างจาก t3-training-base

| Feature | t3-training-base | t3-artf-base |
|---------|------------------|--------------|
| **Input** | .npy or .nii.gz (3D medical) | PNG (2D images) |
| **Target** | Ischemic stroke lesion | Red artifacts (FF0000) |
| **Data Loading** | หารูปก่อน → mask | หา mask ก่อน → รูป |
| **Input Channels** | 2.5D (3 slices stacked) | RGB (3 channels) |
| **Mask Extraction** | Direct binary | Extract red color |
| **Normalization** | Z-score per slice | ImageNet stats |
| **Data Source** | `images/` → `masks/` | `masks/` → `Original/` |

---

## 🏗️ Model Architecture

**Attention U-Net** (เหมือนกับ t3-training-base):
- Encoder: 4 levels (64→128→256→512)
- Bottleneck: 1024
- Decoder: 4 levels with Attention Gates
- Total Parameters: ~31M

---

## 🔍 การทำงานของระบบ

### 1. Data Loading
```python
# อ่าน mask files
masks_dir/image_001.png

# หารูปที่ตรงกัน
Original/image_001.png

# Extract red artifacts
binary_mask = extract_red_mask(mask_rgb, color=[255,0,0])
```

### 2. Mask Extraction
```python
# ค้นหาพิกเซลที่เป็นสีแดง (tolerance ±10)
lower = [245, 0, 0]  # R-10, G-10, B-10
upper = [255, 10, 10]  # R+10, G+10, B+10
binary_mask = cv2.inRange(mask_rgb, lower, upper)
```

### 3. Preprocessing
- Resize to 384x384
- Normalize with ImageNet stats
- Convert to tensor

---

## 🔧 Troubleshooting

### ไม่พบไฟล์รูป
```
Warning: Original not found for image_xxx.png, skipping...
```
**แก้ไข:** ตรวจสอบว่าชื่อไฟล์ใน `masks/` และ `Original/` ตรงกันหรือไม่

### ไม่มี artifact
```
Skipped: X
```
**แก้ไข:** ตรวจสอบว่า mask มีสีแดง (FF0000) หรือไม่ อาจต้องปรับ `tolerance` ในฟังก์ชัน `extract_red_mask()`

### Out of Memory
```python
# ใน config.py
BATCH_SIZE = 8  # ลดจาก 16
```

---

## 📝 ตัวอย่างการใช้งาน

```bash
# 1. เตรียมข้อมูล
# วางไฟล์ PNG ใน ../1_data_raw/masks/ และ ../1_data_raw/Original/

# 2. รัน training
cd t3-artf-base
python train.py

# 3. ดูผลใน MLflow
mlflow ui --port 5000
```

---

## ✅ เสร็จแล้ว!

รัน `python train.py` แล้วรอผลครับ! 🚀

ดูผลใน MLflow UI: `mlflow ui --port 5000`
