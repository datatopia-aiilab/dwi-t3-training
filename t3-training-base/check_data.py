"""
Quick Data Check Script
ตรวจสอบข้อมูล .npy หรือ .nii.gz ว่าสามารถโหลดได้หรือไม่
"""

import glob
import numpy as np
from pathlib import Path

def check_data_directory(data_path="../1_data_raw"):
    """ตรวจสอบโครงสร้างและข้อมูล"""
    
    print("=" * 80)
    print("🔍 ตรวจสอบข้อมูล DWI")
    print("=" * 80)
    
    data_path = Path(data_path)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    
    print(f"\n📂 ตำแหน่งข้อมูล:")
    print(f"   Images: {images_dir}")
    print(f"   Masks: {masks_dir}")
    
    # ตรวจสอบว่าโฟลเดอร์มีอยู่หรือไม่
    if not images_dir.exists():
        print(f"\n❌ ไม่พบโฟลเดอร์ images: {images_dir}")
        return False
    
    if not masks_dir.exists():
        print(f"\n❌ ไม่พบโฟลเดอร์ masks: {masks_dir}")
        return False
    
    print("\n✅ โฟลเดอร์พบแล้ว")
    
    # ค้นหาไฟล์
    npy_images = sorted(glob.glob(str(images_dir / "*.npy")))
    nii_images = sorted(glob.glob(str(images_dir / "*.nii.gz")))
    
    npy_masks = sorted(glob.glob(str(masks_dir / "*.npy")))
    nii_masks = sorted(glob.glob(str(masks_dir / "*.nii.gz")))
    
    print(f"\n📊 ไฟล์ที่พบ:")
    print(f"   Images (.npy): {len(npy_images)}")
    print(f"   Images (.nii.gz): {len(nii_images)}")
    print(f"   Masks (.npy): {len(npy_masks)}")
    print(f"   Masks (.nii.gz): {len(nii_masks)}")
    
    # กำหนดประเภทไฟล์ที่จะใช้
    if len(npy_images) > 0:
        print(f"\n✓ จะใช้ไฟล์ .npy ({len(npy_images)} ไฟล์)")
        image_files = npy_images
        mask_files = npy_masks
        file_type = "npy"
    elif len(nii_images) > 0:
        print(f"\n✓ จะใช้ไฟล์ .nii.gz ({len(nii_images)} ไฟล์)")
        image_files = nii_images
        mask_files = nii_masks
        file_type = "nii.gz"
    else:
        print("\n❌ ไม่พบไฟล์ข้อมูล (.npy หรือ .nii.gz)")
        return False
    
    # ตรวจสอบไฟล์ตัวอย่าง
    print(f"\n🔬 ตรวจสอบไฟล์ตัวอย่าง (5 ไฟล์แรก):")
    
    for idx, img_path in enumerate(image_files[:5]):
        img_name = Path(img_path).name
        
        # หา mask ที่ตรงกัน
        mask_path = masks_dir / img_name
        
        print(f"\n   [{idx+1}] {img_name}")
        
        # ตรวจสอบว่ามี mask หรือไม่
        if not mask_path.exists():
            print(f"       ❌ ไม่พบ mask: {mask_path.name}")
            continue
        else:
            print(f"       ✓ พบ mask")
        
        # โหลดและตรวจสอบ shape
        try:
            if file_type == "npy":
                img_data = np.load(img_path)
                mask_data = np.load(mask_path)
            else:
                import nibabel as nib
                img_data = nib.load(img_path).get_fdata()
                mask_data = nib.load(mask_path).get_fdata()
            
            print(f"       Image shape: {img_data.shape}")
            print(f"       Mask shape: {mask_data.shape}")
            print(f"       Image range: [{img_data.min():.2f}, {img_data.max():.2f}]")
            print(f"       Mask unique values: {np.unique(mask_data)}")
            print(f"       Mask coverage: {(mask_data > 0).sum() / mask_data.size * 100:.2f}%")
            
        except Exception as e:
            print(f"       ❌ Error loading: {e}")
    
    # สรุป
    print("\n" + "=" * 80)
    print("📋 สรุป:")
    print("=" * 80)
    print(f"✓ ประเภทไฟล์: {file_type}")
    print(f"✓ จำนวนไฟล์ images: {len(image_files)}")
    print(f"✓ จำนวนไฟล์ masks: {len(mask_files)}")
    
    # ตรวจสอบว่า image และ mask มีจำนวนเท่ากันหรือไม่
    if len(image_files) != len(mask_files):
        print(f"\n⚠️  คำเตือน: จำนวน images และ masks ไม่เท่ากัน!")
    else:
        print(f"\n✅ จำนวน images และ masks ตรงกัน")
    
    print("\n💡 พร้อมสำหรับการ training!")
    print("   รันคำสั่ง: python train.py")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    import sys
    
    # รับ path จาก command line argument หรือใช้ default
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../1_data_raw"
    
    success = check_data_directory(data_path)
    
    sys.exit(0 if success else 1)
