# 🔧 การแก้ไขปัญหา Visualization & MLflow Logging

## ปัญหาที่พบ

1. ❌ **Training curves ไม่ถูก log ไป MLflow**
   - สาเหตุ: `plot_training_curves_advanced()` ถูกเรียกหลังจาก `log_training_complete()`
   - ผลลัพธ์: กราฟถูกสร้างแต่ไม่ถูก upload ไป MLflow

2. ❌ **ไม่แสดงรายละเอียดว่าอะไรถูก log บ้าง**
   - ไม่มีการแสดงผลว่าไฟล์ใดถูกบันทึก
   - ไม่มีการยืนยันว่า volume metrics ถูก log

3. ❌ **ไม่มีวิธีตรวจสอบ MLflow artifacts**
   - ไม่มี script สำหรับดู runs และ artifacts

## การแก้ไข

### 1. แก้ไข train.py ✅

**เปลี่ยน:** สร้าง training curves **ก่อน** log ไป MLflow

```python
# Before (ผิด):
save_training_history(history, history_path)
log_training_complete(...)  # ← log ก่อน
plot_training_curves_advanced(...)  # ← สร้างกราฟหลัง (ไม่ถูก log!)

# After (ถูก):
save_training_history(history, history_path)
plot_training_curves_advanced(...)  # ← สร้างกราฟก่อน
log_training_complete(..., curves_path)  # ← log ทั้งหมดรวมกราฟ
```

**เพิ่ม:** แสดงรายละเอียดไฟล์ทั้งหมดหลัง test evaluation

```python
print(f"\n📁 GENERATED FILES:")
print(f"   Training Curves: ✅ {curves_path}")
print(f"   Test Results: ✅ {csv_path} ({len(df)} samples)")
print(f"   Prediction Images: ✅ {len(pred_images)} images")
print(f"   Test Plots: ✅ {test_plot}")
print(f"   Model Checkpoints: ✅ best_model.pth, final_model.pth")
```

### 2. แก้ไข mlflow_utils.py ✅

**เปลี่ยน:** `log_training_complete()` รับ `curves_path` parameter

```python
def log_training_complete(cfg, ..., curves_path=None):  # ← เพิ่ม parameter
    ...
    # Log training curves explicitly
    if curves_path and Path(curves_path).exists():
        print(f"   📊 Logging training curves...")
        mlflow.log_artifact(str(curves_path), artifact_path="plots")
        print(f"      ✅ Logged: {Path(curves_path).name} → mlflow artifacts/plots/")
```

**เพิ่ม:** แสดงรายละเอียด MLflow logging

```python
# ใน log_training_complete():
print("   📊 Logging best metrics...")
print(f"      ✅ Best val dice: {best_val_dice:.4f}")
print("   💾 Logging model checkpoint...")
print("   📈 Logging training history...")
print("   📊 Logging training curves...")
print(f"      ✅ Logged: training_curves_advanced.png → mlflow artifacts/plots/")

# แสดง MLflow UI URL
print(f"\n   🌐 View results in MLflow UI:")
print(f"   http://localhost:5000")
print(f"\n   💡 To open MLflow UI, run:")
print(f"   mlflow ui --port 5000")
```

**ปรับปรุง:** `log_complete_evaluation()` แสดงผลละเอียดขึ้น

```python
print(f"   📊 Logging aggregated test metrics...")
print(f"   💾 Logging per-sample CSV...")
print(f"   🖼️  Logging prediction images...")
print(f"   📈 Logging test plots...")

# Check volume metrics
if 'gt_volume_ml' in results['per_sample'][0]:
    print(f"   - Volume metrics: ✅ Included")
else:
    print(f"   - Volume metrics: ❌ Not found")
```

### 3. สร้าง check_mlflow.py ✅

**ใหม่:** Script สำหรับตรวจสอบ MLflow runs และ artifacts

```bash
python check_mlflow.py
```

**จะแสดง:**
- รายการ runs ล่าสุด (5 runs)
- Artifacts ในแต่ละ run
- รายละเอียด latest run:
  - ✅/❌ training_curves_advanced.png
  - ✅/❌ test_per_sample_results.csv
  - ✅/❌ prediction images
  - ✅/❌ volume metrics
- คำแนะนำเปิด MLflow UI

## ผลลัพธ์หลังแก้ไข

### เมื่อรัน `python train.py` จะเห็น:

```bash
✅ TRAINING COMPLETED!

📊 Generating advanced training curves...
   ✅ Saved to: 4_results/plots/training_curves_advanced.png

======================================================================
📦 LOGGING TRAINING ARTIFACTS TO MLFLOW
======================================================================

   📊 Logging best metrics...
      ✅ Best val dice: 0.7855 at epoch 2
   💾 Logging model checkpoint...
      ✅ Logged: best_model.pth
   📈 Logging training history...
      ✅ Logged: training_history.json
   ⚙️  Logging config file...
      ✅ Logged: config.py
   📊 Logging training curves...
      ✅ Logged: training_curves_advanced.png → mlflow artifacts/plots/

======================================================================
✅ ALL ARTIFACTS LOGGED SUCCESSFULLY

📊 MLflow Run Information:
   Run ID: abc123...
   Run Name: unet_20241109_143022

   🌐 View results in MLflow UI:
   http://mlflow.example.com/#/experiments/1/runs/abc123

   💡 To open MLflow UI, run:
   mlflow ui --port 5000
   Then open: http://localhost:5000
======================================================================

🧪 RUNNING AUTOMATIC TEST EVALUATION
...

======================================================================
📦 LOGGING TEST EVALUATION TO MLFLOW
======================================================================

   📊 Logging aggregated test metrics...
      ✅ Logged aggregated test metrics
      ✅ Logged volume metrics
   💾 Logging per-sample CSV...
      ✅ Logged per-sample CSV: test_per_sample_results.csv
   🖼️  Logging prediction images...
      ✅ Logged 10 prediction images
   📈 Logging test plots...
      ✅ Logged test plots

======================================================================
✅ TEST EVALUATION LOGGED TO MLFLOW
   - Aggregated metrics: 9 metrics
   - Per-sample results: 62 samples
   - Volume metrics: ✅ Included (gt_volume_ml, pred_volume_ml, volume_error_percent)
   - Prediction images: logged to artifacts/predictions/
   - Test plots: logged to artifacts/plots/
======================================================================

✅ TEST EVALUATION COMPLETED!

📁 GENERATED FILES:
   Training Curves:
      ✅ 4_results/plots/training_curves_advanced.png

   Test Results:
      ✅ 4_results/test_per_sample_results.csv
         - 62 samples
         - Columns: filename, dice, iou, precision, recall, f1, gt_volume_ml, pred_volume_ml, volume_error_percent

   Prediction Images:
      ✅ 10 images in 4_results/predictions
         - Example: patient001_slice05_dice_0.847.png

   Test Plots:
      ✅ 4_results/plots/test_metrics_distribution.png

   Model Checkpoints:
      ✅ 3_model_weights/best_model.pth
      ✅ 3_model_weights/final_model.pth

   Training History:
      ✅ 4_results/training_history.json
```

### ตรวจสอบ MLflow artifacts:

```bash
python check_mlflow.py
```

จะแสดง:
- ✅ plots/training_curves_advanced.png
- ✅ plots/test_metrics_distribution.png
- ✅ evaluation/test_per_sample_results.csv
- ✅ predictions/ (10 images)
- ✅ models/best_model/
- ✅ Volume metrics (test_mean_gt_volume_ml, test_mean_pred_volume_ml, test_mean_volume_error_percent)

## วิธีใช้งาน

### 1. Training (จะได้ทั้งหมดอัตโนมัติ):
```bash
python train.py
```

### 2. ตรวจสอบ MLflow:
```bash
python check_mlflow.py
```

### 3. เปิด MLflow UI:
```bash
mlflow ui --port 5000
# Open: http://localhost:5000
```

### 4. ดูไฟล์ที่สร้าง:
```bash
ls -R 4_results/
# plots/training_curves_advanced.png
# plots/test_metrics_distribution.png
# predictions/*.png (4-panel images)
# test_per_sample_results.csv (with volume columns)
# training_history.json
```

## สรุป

✅ **แก้ไขแล้ว:**
1. Training curves ถูก log ไป MLflow
2. แสดงรายละเอียดว่าอะไรถูก log
3. แสดง MLflow UI URL
4. ยืนยัน volume metrics ถูก log
5. มี script ตรวจสอบ MLflow artifacts

✅ **ได้ทุกอย่างที่ต้องการ:**
- 📊 Training curves (dual y-axis, 300 DPI)
- 🖼️ Test images (4-panel, volume info)
- 💾 CSV with volumes
- 📦 MLflow logging ครบถ้วน
- 🌐 MLflow UI พร้อมใช้งาน
