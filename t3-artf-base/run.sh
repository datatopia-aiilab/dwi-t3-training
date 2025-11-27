#!/bin/bash

# DWI Artifact Segmentation Training
# Simple run script

echo "======================================"
echo "DWI Artifact Segmentation Training"
echo "======================================"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run training
echo "Starting training..."
python train.py

echo ""
echo "======================================"
echo "Training completed!"
echo "View results: mlflow ui --port 5000"
echo "======================================"
