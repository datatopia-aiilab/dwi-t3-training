# DWI Baseline Training - Simple & Clean

โปรเจคพื้นฐานสำหรับ DWI Ischemic Stroke Segmentation
**เน้นความเรียบง่าย ไม่ซับซ้อน ทำงานได้แน่นอน!**

---

## 📁 โครงสร้าง

```
t3-training-base/
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
# ติดตั้ง dependencies (ถ้ายังไม่มี)
pip install torch torchvision
pip install nibabel opencv-python albumentations
pip install matplotlib tqdm
pip install mlflow
```

### 2. รัน Training (คำสั่งเดียว!)

```bash
cd t3-training-base

# (ตัวเลือก) ตรวจสอบข้อมูลก่อน
python check_data.py

# รัน training
python train.py
```

**จะทำอะไรบ้าง:**
1. โหลดข้อมูลจาก `../1_data_raw/` (รองรับทั้ง .npy และ .nii.gz)
2. Preprocess ในหน่วยความจำ (resize, normalize, 2.5D)
3. แบ่งข้อมูล train/val/test (80/15/5)
4. Train Attention U-Net (100 epochs หรือจน early stop)
5. Evaluate บน test set
6. Log ทุกอย่างไปยัง MLflow
7. เสร็จสิ้น!

### 3. ดูผลลัพธ์

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
- `epochs`: 100
- `base_channels`: 64
- `model_params`: ~31M
- และอื่นๆ ทั้งหมด

### Metrics (ทุก epoch):
- `train_loss`
- `train_dice`
- `val_loss`
- `val_dice`
- `learning_rate`

### Test Metrics:
- `test_dice_mean`
- `test_dice_std`
- `test_iou_mean`
- `test_iou_std`

### Artifacts:
- `training_curve.png` - กราฟ loss + dice
- `test_predictions/` - 10 ตัวอย่างการทำนาย (GT vs Pred)
- `best_model.pth` - โมเดลที่ดีที่สุด
- `test_metrics.json` - สรุปผลการประเมิน

---

## ⚙️ การตั้งค่า

แก้ไขใน `config.py`:

```python
# Data
IMAGE_SIZE = (384, 384)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.15
TEST_RATIO = 0.05

# Model
BASE_CHANNELS = 64  # เพิ่มให้ใหญ่ขึ้น → โมเดลใหญ่ขึ้น

# Training
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 20

# Augmentation
AUG_HFLIP_PROB = 0.3
AUG_ROTATE_PROB = 0.25
AUG_BRIGHTNESS_PROB = 0.2
```

---

## 🎯 ผลลัพธ์ที่คาดหวัง

**Baseline Performance (ไม่มี optimization):**
- Val Dice: **60-70%**
- Test Dice: **55-65%**
- Training Time: ~90-120 minutes (100 epochs, RTX 3080)
- Convergence: ~40-60 epochs

**ทำไมต่ำกว่าโปรเจคเก่า?**
- ไม่มี N4 bias correction
- ไม่มี Deep Supervision
- ไม่มี TTA/CCA
- แต่จะ **stable, simple, debugable**!

---

## 🏗️ Model Architecture

**Attention U-Net:**

```
Encoder:
  Conv Block 1: 3 → 64
  Conv Block 2: 64 → 128
  Conv Block 3: 128 → 256
  Conv Block 4: 256 → 512

Bottleneck:
  Conv Block 5: 512 → 1024

Decoder (with Attention Gates):
  Up Block 1: 1024 → 512 + Attention
  Up Block 2: 512 → 256 + Attention
  Up Block 3: 256 → 128 + Attention
  Up Block 4: 128 → 64 + Attention

Output:
  Conv: 64 → 1 (Sigmoid)

Total Parameters: ~31M
```

**ไม่มี:**
- Deep Supervision
- SE/CBAM/ECA modules
- Multi-scale features
- ซับซ้อนอื่นๆ

**มีแค่:**
- Standard U-Net structure
- Attention Gates (spatial attention)
- BatchNorm + ReLU
- เท่านั้น!

---

## 📝 หมายเหตุ

### ข้อดี:
✅ เรียบง่าย เข้าใจง่าย
✅ รันคำสั่งเดียวทำครบ
✅ Debug ง่าย (แต่ละไฟล์สั้น)
✅ เก็บทุกอย่างใน MLflow
✅ ไม่มีไฟล์กระจัดกระจาย
✅ Reproducible

### ข้อจำกัด:
❌ Performance ไม่สูงสุด (เป็น baseline)
❌ ไม่มี advanced features
❌ ไม่มี N4, Deep Supervision, TTA

### เมื่อไหร่ควรใช้:
- ต้องการ baseline ที่เชื่อถือได้
- ต้องการทดสอบ idea ใหม่อย่างรวดเร็ว
- ต้องการโค้ดที่เข้าใจง่าย debug ง่าย
- ไม่ต้องการความซับซ้อน

---

## 🔧 Troubleshooting

### ถ้า Out of Memory:
```python
# ใน config.py
BATCH_SIZE = 8  # ลดจาก 16
```

### ถ้าต้องการโมเดลเล็กลง:
```python
# ใน config.py
BASE_CHANNELS = 32  # ลดจาก 64
# Parameters จะลดลงเหลือ ~8M
```

### ถ้าต้องการ train เร็วขึ้น:
```python
# ใน config.py
EPOCHS = 50  # ลดจาก 100
EARLY_STOP_PATIENCE = 10  # ลดจาก 20
```

---

## 📚 การพัฒนาต่อ

ถ้าต้องการเพิ่มประสิทธิภาพ ให้เพิ่มทีละอย่าง:

1. **N4 Bias Correction** (+3-5% Dice)
2. **Deep Supervision** (+2-4% Dice)
3. **TTA + CCA** (+2-4% Dice)
4. **เปลี่ยน Loss Function** (+1-2% Dice)

แต่ละอย่างเป็น run แยกใน MLflow เพื่อเปรียบเทียบ!

---

## 🎉 เสร็จแล้ว!

รัน `python train.py` แล้วรอผลครับ! 🚀

ดูผลใน MLflow UI: `mlflow ui --port 5000`
