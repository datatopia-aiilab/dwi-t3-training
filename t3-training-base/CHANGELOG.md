# Changelog - DWI T3 Training Base

## [Update] 2024-11-28

### 🔧 Fixed
- **รองรับไฟล์ .npy**: แก้ไขฟังก์ชัน `load_and_preprocess_data()` ให้รองรับทั้งไฟล์ `.npy` และ `.nii.gz`
  - ระบบจะตรวจสอบไฟล์ `.npy` ก่อน ถ้าไม่มีจึงค้นหา `.nii.gz`
  - รองรับข้อมูลทั้ง 2D และ 3D
  - แสดงประเภทไฟล์ที่กำลังโหลด

### 📝 Changes in `train.py`

#### Before:
```python
# หาเฉพาะไฟล์ .nii.gz
image_files = sorted(glob.glob(str(images_dir / "*.nii.gz")))
print(f"Found {len(image_files)} image files")

if len(image_files) == 0:
    raise FileNotFoundError(f"No .nii.gz files found in {images_dir}")
```

#### After:
```python
# ลองหา .npy ก่อน ถ้าไม่มีค่อยหา .nii.gz
image_files = sorted(glob.glob(str(images_dir / "*.npy")))
if len(image_files) == 0:
    image_files = sorted(glob.glob(str(images_dir / "*.nii.gz")))
    file_type = "nii.gz"
else:
    file_type = "npy"

print(f"Found {len(image_files)} {file_type} image files")

if len(image_files) == 0:
    raise FileNotFoundError(f"No .npy or .nii.gz files found in {images_dir}")
```

#### การโหลดข้อมูล:
```python
# Load data based on file type
if file_type == "npy":
    # Load .npy files
    img_data = np.load(img_path)
    mask_data = np.load(str(mask_path))
else:
    # Load NIfTI files
    img_nii = nib.load(img_path)
    mask_nii = nib.load(str(mask_path))
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

# Handle different data shapes
# If 2D (H, W), add a dummy slice dimension
if img_data.ndim == 2:
    img_data = img_data[:, :, np.newaxis]
    mask_data = mask_data[:, :, np.newaxis]
```

### ✅ Features
- ✓ รองรับไฟล์ `.npy` (NumPy array format)
- ✓ รองรับไฟล์ `.nii.gz` (NIfTI format)
- ✓ รองรับข้อมูล 2D และ 3D
- ✓ Auto-detection ประเภทไฟล์
- ✓ Error message ชัดเจนกว่าเดิม

### 📊 โครงสร้างข้อมูลที่รองรับ

#### ไฟล์ .npy:
```
1_data_raw/
├── images/
│   ├── patient_001.npy  # shape: (H, W) หรือ (H, W, D)
│   ├── patient_002.npy
│   └── ...
└── masks/
    ├── patient_001.npy  # shape: (H, W) หรือ (H, W, D)
    ├── patient_002.npy
    └── ...
```

#### ไฟล์ .nii.gz:
```
1_data_raw/
├── images/
│   ├── patient_001.nii.gz
│   ├── patient_002.nii.gz
│   └── ...
└── masks/
    ├── patient_001.nii.gz
    ├── patient_002.nii.gz
    └── ...
```

### 🚀 วิธีใช้งาน

ไม่มีการเปลี่ยนแปลง! ใช้งานเหมือนเดิม:

```bash
cd t3-training-base
python train.py
```

ระบบจะ:
1. ตรวจสอบไฟล์ `.npy` ในโฟลเดอร์ `images/` ก่อน
2. ถ้าไม่มี จึงค้นหา `.nii.gz`
3. โหลดและประมวลผลตามประเภทไฟล์ที่พบ
4. ดำเนินการ training ตามปกติ

### 🔍 การแก้ปัญหา

#### ปัญหาเดิม:
```
FileNotFoundError: No .nii.gz files found in /path/to/1_data_raw/images
```

#### ปัญหาแก้แล้ว:
- ระบบจะค้นหาทั้งสองประเภทไฟล์
- แสดง error message ที่ชัดเจนขึ้น
- รองรับข้อมูลหลายรูปแบบ

### 📦 Dependencies
ไม่มีการเพิ่ม dependencies ใหม่ - ใช้ `numpy` ที่มีอยู่แล้ว

### ⚠️ หมายเหตุ
- ชื่อไฟล์ของ image และ mask **ต้องตรงกัน**
- รองรับ mixed format ไม่ได้ (ต้องเป็น .npy ทั้งหมด หรือ .nii.gz ทั้งหมด)
- ข้อมูล 2D จะถูกแปลงเป็น 3D อัตโนมัติ (เพิ่ม dimension)

---

## Previous Versions

### [Initial] 2024-11-27
- สร้าง baseline training pipeline
- Attention U-Net model
- MLflow integration
- In-memory preprocessing
