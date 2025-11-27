# 🚀 Quick Start Guide: Model Improvements
## การเริ่มต้นใช้งานฟีเจอร์ใหม่

**อัปเดต:** 27 พฤศจิกายน 2568

---

## ✅ **สถานะปัจจุบัน**

### เสร็จแล้ว (Phase 1-2):
- ✅ Test-Time Augmentation (TTA)
- ✅ Connected Component Analysis (CCA)
- ✅ N4 Bias Field Correction

### รอดำเนินการ (Phase 3-4):
- 🔄 Gamma Correction Augmentation
- 🔄 Log-Cosh Dice Loss
- 🔄 Deep Supervision

---

## 📦 **ขั้นตอนที่ 1: Install Dependencies**

```bash
# Navigate to project directory
cd /Users/Sribilone/AiiLAB/_datatopia/DWI/NovEdition/dwi-t3-training

# Install/update requirements
pip install -r requirements.txt

# สำคัญ: ตรวจสอบว่า SimpleITK ติดตั้งสำเร็จ
python -c "import SimpleITK as sitk; print(f'SimpleITK version: {sitk.Version.VersionString()}')"
```

---

## 🧪 **ขั้นตอนที่ 2: ทดสอบ TTA + CCA (ไม่ต้อง Retrain)**

### 2.1 ตรวจสอบ Configuration

```bash
# ดู config ปัจจุบัน
python -c "
import config
print('='*60)
print('TTA + CCA Configuration')
print('='*60)
print(f'USE_TTA: {config.USE_TTA}')
print(f'TTA_AUGMENTATIONS: {config.TTA_AUGMENTATIONS}')
print(f'USE_CCA: {config.USE_CCA}')
print(f'CCA_MIN_SIZE: {config.CCA_MIN_SIZE} pixels')
print(f'CCA_MIN_CONFIDENCE: {config.CCA_MIN_CONFIDENCE}')
print('='*60)
"
```

**ผลลัพธ์ที่ควรเห็น:**
```
============================================================
TTA + CCA Configuration
============================================================
USE_TTA: True
TTA_AUGMENTATIONS: ['hflip', 'vflip']
USE_CCA: True
CCA_MIN_SIZE: 10 pixels
CCA_MIN_CONFIDENCE: 0.3
============================================================
```

### 2.2 Run Evaluation กับ Model ปัจจุบัน

```bash
# Evaluate with TTA + CCA (ใช้ best_model.pth ที่มีอยู่)
python evaluate.py --num_samples 48

# หรือ evaluate ทุก test samples
python evaluate.py --num_samples 999
```

**สิ่งที่จะเกิดขึ้น:**
- โหลด `best_model.pth`
- ใช้ TTA ทำ prediction หลายครั้ง (hflip, vflip)
- Average predictions
- Apply CCA cleaning
- แสดงผลลัพธ์

**ตัวอย่าง Output:**
```
🚀 Running evaluation with TTA+CCA...
🔄 TTA enabled with: ['hflip', 'vflip']
   Number of predictions to average: 3
✅ CCA enabled: min_size=10px, min_conf=0.3

Evaluating (TTA+CCA): 100%|███████████| 48/48 [01:23<00:00]

📊 TEST SET RESULTS (with TTA+CCA):
   Dice:      0.6523 ± 0.1234  (+0.0323 vs baseline)
   IoU:       0.5234 ± 0.1123
   Precision: 0.7123 ± 0.1345  (+0.0523 vs baseline)
   Recall:    0.6234 ± 0.1256
```

### 2.3 เปรียบเทียบผลลัพธ์

```bash
# ดูผลลัพธ์ละเอียด
cat 4_results/test_per_sample_results.csv

# หรือใช้ Python
python -c "
import pandas as pd
df = pd.read_csv('4_results/test_per_sample_results.csv')
print(df[['filename', 'dice', 'precision', 'recall']].head(10))
print(f'\nAverage Dice: {df['dice'].mean():.4f}')
print(f'Average Precision: {df['precision'].mean():.4f}')
"
```

### 2.4 ปิด TTA/CCA เพื่อเปรียบเทียบ (Optional)

```python
# แก้ไข config.py
USE_TTA = False
USE_CCA = False

# Run evaluate อีกครั้ง
python evaluate.py
```

---

## 🔬 **ขั้นตอนที่ 3: Re-preprocess ด้วย N4 Correction**

### 3.1 Backup ข้อมูลเดิม

```bash
# Backup processed data
mv 2_data_processed 2_data_processed_backup

# หรือ copy (ปลอดภัยกว่า แต่ใช้พื้นที่มากกว่า)
cp -r 2_data_processed 2_data_processed_backup
```

### 3.2 ตรวจสอบ N4 Configuration

```bash
python -c "
import config
print('='*60)
print('N4 Bias Correction Configuration')
print('='*60)
print(f'N4_ENABLED: {config.N4_ENABLED}')
print(f'N4_SHRINK_FACTOR: {config.N4_SHRINK_FACTOR}')
print(f'N4_NUM_ITERATIONS: {config.N4_NUM_ITERATIONS}')
print(f'N4_NUM_WORKERS: {config.N4_NUM_WORKERS}')
print('='*60)
print('\nEstimated processing time:')
print('  Single-threaded: ~2-4 hours')
print(f'  {config.N4_NUM_WORKERS} workers: ~30-60 minutes')
print('='*60)
"
```

### 3.3 Run Preprocessing

```bash
# Run preprocessing with N4 correction
# ใช้เวลาประมาณ 30-60 นาที (4 workers)
python 01_preprocess.py
```

**ผลลัพธ์ที่คาดหวัง:**
```
============================================================
PREPROCESSING DWI IMAGES
============================================================

Step 1: Creating directories...
✅ Created all necessary directories

Step 2: Building slice mappings...
Found 848 image-mask pairs
Train: 640 slices
Val: 160 slices
Test: 48 slices

Step 3: Computing normalization stats from training set...
Loading images: 100%|███████████| 640/640
Mean: 156.2341
Std:  89.1234

Step 4: Processing splits...
🚀 Processing train set (640 files)...
   N4 correction enabled (shrink=4, workers=4)
   train: 100%|███████████| 640/640 [25:32<00:00, 0.42it/s]
   ✅ TRAIN: 640/640 files processed successfully

🚀 Processing val set (160 files)...
   val: 100%|███████████| 160/160 [06:23<00:00, 0.42it/s]
   ✅ VAL: 160/160 files processed successfully

🚀 Processing test set (48 files)...
   test: 100%|███████████| 48/48 [01:55<00:00, 0.42it/s]
   ✅ TEST: 48/48 files processed successfully

Step 5: Saving preprocessing config...
✅ Saved to: 2_data_processed/preprocess_config.json

============================================================
✅ PREPROCESSING COMPLETED!
============================================================
Total time: 33 minutes 50 seconds
```

### 3.4 ตรวจสอบผลลัพธ์

```bash
# ตรวจสอบ config ที่บันทึก
cat 2_data_processed/preprocess_config.json

# ควรเห็น:
# "n4_enabled": true,
# "n4_shrink_factor": 4,
# "n4_num_iterations": 50,
```

---

## 🎓 **ขั้นตอนที่ 4: Retrain Model**

### 4.1 Run Training

```bash
# Train model ด้วย N4-corrected data
python train.py

# Expected training time: 6-12 hours (depends on GPU)
```

**ผลลัพธ์ที่คาดหวัง:**
```
Epoch 50/100 - 145.2s - LR: 0.000042
  Train Loss: 0.2234 | Train Dice: 0.7766
  Val Loss:   0.2512 | Val Dice:   0.7488
  ✅ New best model! Val Dice: 0.7488 (saved)

...

Epoch 100/100 - 142.8s - LR: 0.000008
  Train Loss: 0.2134 | Train Dice: 0.7866
  Val Loss:   0.2398 | Val Dice:   0.7602
  ✅ New best model! Val Dice: 0.7602 (saved)

============================================================
✅ TRAINING COMPLETED!
============================================================
Total training time: 8.2 hours
Best validation Dice: 0.7602 at epoch 95  (+0.06 vs baseline)
```

### 4.2 Auto-Evaluation

Training จะ run evaluation อัตโนมัติหลังเสร็จ:

```
============================================================
🧪 RUNNING AUTOMATIC TEST EVALUATION
============================================================

📊 Evaluating on 48 test samples...
🔄 TTA enabled: ['hflip', 'vflip']
✅ CCA enabled: min_size=10px, min_conf=0.3

Evaluating (TTA+CCA): 100%|███████████| 48/48

📊 TEST SET RESULTS:
   Dice:      0.6823 ± 0.1123  (+0.06 vs baseline)
   IoU:       0.5523 ± 0.1034
   Precision: 0.7423 ± 0.1234
   Recall:    0.6523 ± 0.1145

✅ TEST EVALUATION COMPLETED!
```

---

## 📊 **ขั้นตอนที่ 5: เปรียบเทียบผลลัพธ์**

### 5.1 เปรียบเทียบ Training Curves

```bash
# ดู training curves
open 4_results/plots/training_curves_separated.png

# หรือใช้ MLflow UI
mlflow ui --backend-store-uri mlruns
# เปิด browser: http://localhost:5000
```

### 5.2 เปรียบเทียบ Metrics

```python
# สร้าง comparison script
python -c "
import pandas as pd
import json

# Load baseline results (ถ้ามี)
baseline = pd.read_csv('4_results/baseline_results.csv')

# Load new results
new = pd.read_csv('4_results/test_per_sample_results.csv')

print('='*60)
print('COMPARISON: Baseline vs N4+TTA+CCA')
print('='*60)
print(f'Baseline Dice: {baseline['dice'].mean():.4f}')
print(f'New Dice:      {new['dice'].mean():.4f}')
print(f'Improvement:   +{(new['dice'].mean() - baseline['dice'].mean()):.4f}')
print('='*60)
"
```

---

## 🔧 **Troubleshooting**

### ปัญหา: SimpleITK ติดตั้งไม่ได้

```bash
# ลอง install ด้วย conda แทน pip
conda install -c simpleitk simpleitk

# หรือ install version เฉพาะ
pip install SimpleITK==2.2.1
```

### ปัญหา: Out of Memory ตอน Training

```python
# แก้ใน config.py
BATCH_SIZE = 8  # ลดจาก 16
USE_MIXED_PRECISION = True  # ต้องเปิด
```

### ปัญหา: N4 Correction ช้ามาก

```python
# แก้ใน config.py
N4_SHRINK_FACTOR = 8  # เพิ่มจาก 4 (เร็วขึ้น 2x แต่คุณภาพลดลงเล็กน้อย)
N4_NUM_ITERATIONS = 25  # ลดจาก 50
N4_NUM_WORKERS = 8  # เพิ่ม workers (ถ้ามี CPU หลายคอร์)
```

### ปัญหา: TTA ช้ามาก

```python
# แก้ใน config.py
TTA_AUGMENTATIONS = ['hflip']  # ใช้แค่ hflip (เร็วขึ้น 2x)
# หรือปิด TTA
USE_TTA = False
```

### ปัญหา: CCA กรองออกมากเกินไป

```python
# แก้ใน config.py
CCA_MIN_SIZE = 5  # ลดจาก 10 (รักษา small lesions ไว้มากขึ้น)
CCA_MIN_CONFIDENCE = 0.2  # ลดจาก 0.3
```

---

## 📈 **Expected Results Timeline**

### วันที่ 1-2: TTA + CCA (ไม่ retrain)
```
Test Dice: 62% → 64-66%  (+2-4%)
```

### วันที่ 3-5: N4 + Retrain
```
Val Dice:  70% → 73-76%  (+3-6%)
Test Dice: 62% → 67-72%  (+5-10%)
```

### วันที่ 6-10: Phase 3-4 (Gamma + Log-Cosh + Deep Supervision)
```
Val Dice:  70% → 75-78%  (+5-8%)
Test Dice: 62% → 71-79%  (+9-17%)
```

---

## ✅ **Checklist**

### Phase 1-2 (ทำแล้ว)
- [x] Install SimpleITK
- [x] ตรวจสอบ TTA/CCA config
- [x] Run evaluate.py ด้วย TTA+CCA
- [x] Backup ข้อมูลเดิม
- [x] Run preprocessing ด้วย N4
- [ ] **Retrain model** ← ขั้นตอนถัดไป
- [ ] เปรียบเทียบผลลัพธ์

### Phase 3-4 (ยังไม่ทำ)
- [ ] เพิ่ม Gamma Correction
- [ ] เพิ่ม Log-Cosh Dice Loss
- [ ] Retrain ด้วย augmentation ใหม่
- [ ] เพิ่ม Deep Supervision
- [ ] Final retrain
- [ ] Final evaluation

---

## 📞 **Next Steps**

1. **ทดสอบ TTA+CCA:**
   ```bash
   python evaluate.py
   ```

2. **Re-preprocess ด้วย N4:**
   ```bash
   pip install SimpleITK
   mv 2_data_processed 2_data_processed_backup
   python 01_preprocess.py
   ```

3. **Retrain Model:**
   ```bash
   python train.py
   ```

4. **เปรียบเทียบผลลัพธ์:**
   ```bash
   mlflow ui
   # Check metrics in browser
   ```

---

**หมายเหตุ:** ทุกขั้นตอนสามารถ rollback ได้โดยใช้ backup ที่สร้างไว้

**เอกสารเพิ่มเติม:** ดูที่ `IMPROVEMENT_PLAN.md` สำหรับรายละเอียดเทคนิค
