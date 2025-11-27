#!/bin/bash

# Quick Start Script for DWI Baseline Training
# This script will check environment and run training

echo "======================================"
echo "DWI Baseline Training - Quick Start"
echo "======================================"

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found!"
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "❌ train.py not found! Please run this script from t3-training-base/ directory"
    exit 1
fi

# Check if data exists
if [ ! -d "../dwi-t3-training/1_data_raw" ]; then
    echo "❌ Data not found at ../dwi-t3-training/1_data_raw/"
    echo "Please make sure the data directory exists"
    exit 1
fi

echo "✅ Data directory found"

# Check dependencies
echo ""
echo "Checking dependencies..."

MISSING_DEPS=0

python -c "import torch" 2>/dev/null || { echo "❌ PyTorch not installed"; MISSING_DEPS=1; }
python -c "import nibabel" 2>/dev/null || { echo "❌ nibabel not installed"; MISSING_DEPS=1; }
python -c "import cv2" 2>/dev/null || { echo "❌ opencv-python not installed"; MISSING_DEPS=1; }
python -c "import albumentations" 2>/dev/null || { echo "❌ albumentations not installed"; MISSING_DEPS=1; }
python -c "import mlflow" 2>/dev/null || { echo "❌ mlflow not installed"; MISSING_DEPS=1; }

if [ $MISSING_DEPS -eq 1 ]; then
    echo ""
    echo "Some dependencies are missing. Install them with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "✅ All dependencies installed"

# Check GPU
echo ""
echo "Checking GPU..."
python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Ready to start
echo ""
echo "======================================"
echo "Everything looks good!"
echo "======================================"
echo ""
echo "Starting training in 3 seconds..."
echo "Press Ctrl+C to cancel"
echo ""

sleep 3

# Run training
python train.py

echo ""
echo "======================================"
echo "Training finished!"
echo "======================================"
echo ""
echo "View results with:"
echo "  mlflow ui --port 5000"
echo "  Then open: http://localhost:5000"
echo ""
