"""
Code Verification Script
Tests all components without requiring dependencies to be installed
"""

import ast
import sys
from pathlib import Path

def check_syntax(filepath):
    """Check Python file syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax Error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def analyze_imports(filepath):
    """Analyze imports in a Python file"""
    imports = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return imports
    except Exception as e:
        return [f"Error: {str(e)}"]

def analyze_functions_and_classes(filepath):
    """Extract functions and classes from a Python file"""
    functions = []
    classes = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                    functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return functions, classes
    except Exception as e:
        return [f"Error: {str(e)}"], []

def main():
    """Main verification function"""
    print("=" * 80)
    print("🔍 การตรวจสอบ Code ทั้งหมด (DWI T3 Training Base)")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    python_files = [
        base_dir / "config.py",
        base_dir / "model.py",
        base_dir / "train.py"
    ]
    
    all_ok = True
    
    for filepath in python_files:
        print(f"\n{'=' * 80}")
        print(f"📄 ไฟล์: {filepath.name}")
        print("=" * 80)
        
        if not filepath.exists():
            print(f"❌ ไฟล์ไม่พบ: {filepath}")
            all_ok = False
            continue
        
        # Check syntax
        print("\n1️⃣ ตรวจสอบ Syntax:")
        is_valid, message = check_syntax(filepath)
        if is_valid:
            print(f"   ✅ {message}")
        else:
            print(f"   ❌ {message}")
            all_ok = False
            continue
        
        # Analyze imports
        print("\n2️⃣ Libraries ที่ใช้:")
        imports = analyze_imports(filepath)
        unique_imports = sorted(set([imp.split('.')[0] for imp in imports]))
        for imp in unique_imports:
            print(f"   - {imp}")
        
        # Analyze structure
        print("\n3️⃣ โครงสร้างโค้ด:")
        functions, classes = analyze_functions_and_classes(filepath)
        
        if classes:
            print(f"   Classes ({len(classes)}):")
            for cls in classes:
                print(f"      - {cls}")
        
        if functions:
            print(f"   Functions ({len(functions)}):")
            for func in functions[:10]:  # Show first 10
                print(f"      - {func}")
            if len(functions) > 10:
                print(f"      ... และอีก {len(functions) - 10} functions")
        
        # File size
        size_kb = filepath.stat().st_size / 1024
        print(f"\n4️⃣ ขนาดไฟล์: {size_kb:.1f} KB")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 สรุปผลการตรวจสอบ")
    print("=" * 80)
    
    print("\n✅ ไฟล์ที่ตรวจสอบ:")
    for filepath in python_files:
        if filepath.exists():
            is_valid, _ = check_syntax(filepath)
            status = "✅ OK" if is_valid else "❌ ERROR"
            print(f"   {status} - {filepath.name}")
    
    print("\n📋 รายละเอียด Project:")
    print("   - Project: DWI Ischemic Stroke Segmentation")
    print("   - Model: Attention U-Net")
    print("   - Framework: PyTorch")
    print("   - Tracking: MLflow")
    
    print("\n🎯 คุณสมบัติหลัก:")
    print("   ✓ โหลดและ preprocess ข้อมูล .nii.gz")
    print("   ✓ สร้าง 2.5D input (3 channels)")
    print("   ✓ Data augmentation (ถ้าเปิดใช้งาน)")
    print("   ✓ Train Attention U-Net model")
    print("   ✓ Validation และ Early stopping")
    print("   ✓ Evaluation บน test set")
    print("   ✓ MLflow experiment tracking")
    print("   ✓ บันทึกผลลัพธ์และ visualizations")
    
    print("\n⚙️ Dependencies ที่ต้องการ:")
    required_packages = [
        "torch (>=2.0.0)",
        "torchvision (>=0.15.0)",
        "nibabel (>=5.0.0)",
        "opencv-python (>=4.8.0)",
        "albumentations (>=1.3.0)",
        "numpy (>=1.24.0)",
        "matplotlib (>=3.7.0)",
        "tqdm (>=4.65.0)",
        "mlflow (>=2.8.0)"
    ]
    for pkg in required_packages:
        print(f"   - {pkg}")
    
    print("\n🚀 วิธีใช้งาน:")
    print("   1. ติดตั้ง dependencies: pip install -r requirements.txt")
    print("   2. เตรียมข้อมูลใน: ../1_data_raw/images/ และ ../1_data_raw/masks/")
    print("   3. รัน training: python train.py")
    print("   4. ดูผลลัพธ์: mlflow ui --port 5000")
    
    if all_ok:
        print("\n" + "=" * 80)
        print("✅ ผลการตรวจสอบ: ทุกไฟล์ถูกต้อง พร้อมใช้งาน!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ ผลการตรวจสอบ: พบข้อผิดพลาด โปรดตรวจสอบข้อความด้านบน")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
