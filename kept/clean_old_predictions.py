"""
Clean Old Prediction Images
ลบภาพ prediction เก่าที่ถูกสร้างด้วยฟังก์ชัน visualize_sample() แบบเก่า
"""

import shutil
from pathlib import Path
import config

def clean_old_predictions():
    """ลบไฟล์ prediction images เก่าทั้งหมด"""
    
    print("\n" + "="*70)
    print("🧹 CLEANING OLD PREDICTION IMAGES")
    print("="*70)
    
    # Directories to clean
    dirs_to_clean = [
        config.PREDICTIONS_DIR,
        config.PLOTS_DIR
    ]
    
    total_deleted = 0
    
    for dir_path in dirs_to_clean:
        if not dir_path.exists():
            print(f"\n   ⚠️  Directory not found: {dir_path}")
            continue
            
        # Count files before deletion
        png_files = list(dir_path.glob("*.png"))
        jpg_files = list(dir_path.glob("*.jpg"))
        all_image_files = png_files + jpg_files
        
        if not all_image_files:
            print(f"\n   ✅ {dir_path.name}/ - Already empty")
            continue
        
        print(f"\n   📁 {dir_path.name}/")
        print(f"      Found {len(all_image_files)} image files")
        
        # Delete all image files
        for img_file in all_image_files:
            try:
                img_file.unlink()
                total_deleted += 1
            except Exception as e:
                print(f"      ❌ Error deleting {img_file.name}: {e}")
        
        print(f"      ✅ Deleted {len(all_image_files)} files")
    
    print("\n" + "="*70)
    print(f"✅ TOTAL: Deleted {total_deleted} old image files")
    print("="*70)
    
    print("\n💡 Next steps:")
    print("   1. Run: python train.py")
    print("      → Will generate new 4-panel training curves (combined & separated)")
    print("   2. Run: python evaluate.py --model [best_model.pth]")
    print("      → Will generate new 4-panel test predictions with volumes")
    print()


if __name__ == "__main__":
    import sys
    
    # Safety check
    response = input("\n⚠️  This will DELETE all images in predictions/ and plots/\n   Continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        clean_old_predictions()
        print("\n✅ Done! Ready for fresh predictions.\n")
    else:
        print("\n❌ Cancelled.\n")
        sys.exit(0)
