# 🧪 Automated Experiment System

ระบบรันการทดลองอัตโนมัติสำหรับ DWI Segmentation Project

## 📁 ไฟล์ที่เกี่ยวข้อง

- `run_experiments.py` - Main script สำหรับรันการทดลอง
- `analyze_results.py` - วิเคราะห์และสรุปผลการทดลอง
- `experiments_example.json` - ตัวอย่าง custom experiment config
- `experiment_results.json` - ผลลัพธ์ทั้งหมด (auto-generated)
- `experiment_logs/` - Log files แต่ละการทดลอง

## 🚀 Quick Start

### Stage 1: Architecture Selection (16 experiments, ~8 hours)
ทดสอบ 6 architectures × 3 encoders เพื่อหา best combination

```bash
python run_experiments.py --stage 1
```

**Experiments:**
- attention_unet (custom)
- unet++ × 3 encoders (efficientnet-b0, efficientnet-b3, resnet34)
- fpn × 3 encoders
- deeplabv3+ × 3 encoders
- manet × 3 encoders
- pspnet × 3 encoders

**Expected time:** ~30 min/experiment × 16 = ~8 hours

### Stage 2: Resolution Optimization (9 experiments, ~5 hours)
ทดสอบ 3 resolutions สำหรับ top-3 models

```bash
# ดูผลลัพธ์ stage 1 ก่อน
python analyze_results.py

# เลือก top-3 models แล้วรัน stage 2
python run_experiments.py --stage 2 --top-models "manet_efficientnet-b3,deeplabv3+_resnet34,fpn_efficientnet-b3"
```

**Experiments:**
- Top 3 models × 3 resolutions (256, 384, 512)

**Expected time:** ~30-40 min/experiment × 9 = ~5 hours

### Stage 3: Preprocessing (4 experiments, ~2 hours)
ทดสอบ CLAHE และ Augmentation

```bash
python run_experiments.py --stage 3
```

**Experiments:**
- Best model × (CLAHE on/off) × (Aug on/off)

**Expected time:** ~30 min/experiment × 4 = ~2 hours

### Stage 4: Fine-tuning (9 experiments, ~4-5 hours)
ทดสอบ loss functions และ learning rates

```bash
python run_experiments.py --stage 4
```

**Experiments:**
- Best model × 3 loss types × 3 learning rates

**Expected time:** ~30 min/experiment × 9 = ~4-5 hours

---

## 📊 การวิเคราะห์ผลลัพธ์

### ดูสรุปผล
```bash
python analyze_results.py
```

Output:
- Summary statistics
- Top 10 results
- Architecture comparison
- Encoder comparison
- Resolution comparison
- Preprocessing impact

### Export เป็น CSV
```bash
python analyze_results.py --export-csv results.csv
```

### Export เป็น HTML Report
```bash
python analyze_results.py --export-html report.html
```

### สร้างกราฟเปรียบเทียบ
```bash
python analyze_results.py --plot
```

ไฟล์กราฟจะถูกสร้างใน `experiment_plots/`:
- `architecture_comparison.png`
- `encoder_comparison.png`
- `time_vs_performance.png`
- `resolution_comparison.png`

---

## 🎨 Custom Experiments

### สร้าง Custom Config

สร้างไฟล์ JSON (เช่น `my_experiments.json`):

```json
{
  "description": "My custom experiments",
  "experiments": [
    {
      "id": "exp1_manet_b3_512",
      "params": {
        "MODEL_ARCHITECTURE": "manet",
        "ENCODER_NAME": "efficientnet-b3",
        "IMAGE_SIZE": [512, 512],
        "BATCH_SIZE": 8,
        "NUM_EPOCHS": 200,
        "AUGMENTATION_ENABLED": true,
        "CLAHE_ENABLED": false,
        "LEARNING_RATE": 8e-5,
        "LOSS_TYPE": "dice"
      }
    },
    {
      "id": "exp2_attention_unet_256",
      "params": {
        "MODEL_ARCHITECTURE": "attention_unet",
        "IMAGE_SIZE": [256, 256],
        "BATCH_SIZE": 32,
        "NUM_EPOCHS": 200,
        "AUGMENTATION_ENABLED": true,
        "CLAHE_ENABLED": true
      }
    }
  ]
}
```

### รัน Custom Config

```bash
python run_experiments.py --config my_experiments.json
```

---

## 🔧 Advanced Options

### Skip Completed Experiments
```bash
python run_experiments.py --stage 1 --skip-existing
```

### Continue Failed Experiments
```bash
# แก้ไขปัญหาแล้วรันใหม่ (จะ skip experiments ที่สำเร็จแล้ว)
python run_experiments.py --stage 1 --skip-existing
```

---

## 📝 Experiment Parameters

### ค่าที่สามารถปรับได้:

**Model Architecture:**
- `MODEL_ARCHITECTURE`: attention_unet, unet++, fpn, deeplabv3+, manet, pspnet

**Encoder (สำหรับ SMP models):**
- `ENCODER_NAME`: efficientnet-b0, efficientnet-b3, resnet34, resnet50, resnext50_32x4d

**Image Processing:**
- `IMAGE_SIZE`: (256,256), (384,384), (512,512)
- `CLAHE_ENABLED`: true/false
- `AUGMENTATION_ENABLED`: true/false

**Training:**
- `NUM_EPOCHS`: 200 (default), 100 (quick test)
- `BATCH_SIZE`: 8, 16, 32
- `LEARNING_RATE`: 5e-5, 8e-5, 1e-4
- `OPTIMIZER`: adam, adamw
- `LOSS_TYPE`: dice, focal, combo

**Regularization:**
- `WEIGHT_DECAY`: 1e-5, 8e-5, 1e-4
- `GRADIENT_CLIP_VALUE`: 1.0

---

## 📈 Expected Results

### Total Experiments: ~38 experiments
- Stage 1: 16 experiments (~8 hours)
- Stage 2: 9 experiments (~5 hours)  
- Stage 3: 4 experiments (~2 hours)
- Stage 4: 9 experiments (~5 hours)

**Total time: ~20 hours** (ถ้ารันทีละตัว)

### Best Practices:

1. **Run Stage 1 first** - หา best architecture
2. **Analyze results** - เลือก top-3 models
3. **Run Stage 2** - optimize resolution
4. **Run Stage 3** - optimize preprocessing
5. **Run Stage 4** - fine-tune hyperparameters

---

## 🐛 Troubleshooting

### Experiment Failed
- ตรวจสอบ log file: `experiment_logs/{experiment_id}.log`
- แก้ไขปัญหาแล้วรันใหม่ (จะ skip experiments ที่สำเร็จแล้ว)

### Out of Memory
- ลด `BATCH_SIZE` ในการทดลอง
- หรือลด `IMAGE_SIZE`

### Preprocessing Error
- ตรวจสอบว่า raw data มีอยู่ใน `1_data_raw/`
- รัน `python 01_preprocess.py` แยกก่อน

### Training Timeout
- เพิ่ม timeout ใน `run_experiments.py` (default: 2 hours)
- หรือลด `NUM_EPOCHS` สำหรับการทดลองเร็ว

---

## 💡 Tips

### Quick Test (5 minutes)
ทดสอบว่า system ทำงานได้ก่อนรัน full experiments:

```bash
# สร้าง quick_test.json
{
  "experiments": [
    {
      "id": "quick_test",
      "params": {
        "MODEL_ARCHITECTURE": "manet",
        "ENCODER_NAME": "efficientnet-b0",
        "IMAGE_SIZE": [256, 256],
        "BATCH_SIZE": 16,
        "NUM_EPOCHS": 2,
        "AUGMENTATION_ENABLED": false
      }
    }
  ]
}

# รัน
python run_experiments.py --config quick_test.json
```

### Parallel Experiments
ถ้ามี multiple GPUs สามารถรันหลาย experiments พร้อมกันได้:
- แยกไฟล์ config
- รันแต่ละไฟล์บน GPU ต่างกัน
- ใช้ `CUDA_VISIBLE_DEVICES=0 python run_experiments.py ...`

---

## 📦 Output Files

```
dwi-t3-training/
├── experiment_results.json      # ผลลัพธ์ทั้งหมด (JSON)
├── experiment_logs/             # Log แต่ละการทดลอง
│   ├── s1_manet_efficientnet-b3.log
│   ├── s1_deeplabv3+_resnet34.log
│   └── ...
├── experiment_plots/            # กราฟเปรียบเทียบ
│   ├── architecture_comparison.png
│   ├── encoder_comparison.png
│   └── ...
└── mlruns/                      # MLflow tracking
    └── {experiment_id}/
```

---

## 🎯 Goal

หา best configuration ที่ให้:
- **Highest Dice Score** (validation + test)
- **Reasonable Training Time** (<1 hour)
- **Stable Performance** (low variance)
- **Good Generalization** (val-test gap < 5%)

---

## 📚 Additional Resources

- MLflow UI: `mlflow ui --backend-store-uri ./mlruns`
- Config reference: `config.py`
- Model architectures: `models.py`
- Dataset info: `dataset.py`

---

**Happy Experimenting! 🚀**
