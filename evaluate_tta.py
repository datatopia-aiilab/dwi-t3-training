"""
Test-Time Augmentation (TTA) Evaluation
Applies multiple augmentations during inference and averages predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import *
from model import get_attention_unet
from dataset import DWIDataset25D
from torch.utils.data import DataLoader


def tta_predict(model, image, device):
    """
    Apply Test-Time Augmentation
    
    Augmentations:
    1. Original
    2. Horizontal flip
    3. Rotate +5°
    4. Rotate -5°
    5. Rotate +10°
    6. Rotate -10°
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 1. Original
        pred = torch.sigmoid(model(image))
        predictions.append(pred)
        
        # 2. Horizontal flip
        image_hflip = torch.flip(image, dims=[3])
        pred_hflip = torch.sigmoid(model(image_hflip))
        pred_hflip = torch.flip(pred_hflip, dims=[3])
        predictions.append(pred_hflip)
        
        # 3-6. Rotations
        angles = [5, -5, 10, -10]
        for angle in angles:
            # Rotate image
            angle_rad = np.deg2rad(angle)
            theta = torch.tensor([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0]
            ], dtype=torch.float32, device=device).unsqueeze(0)
            
            grid = F.affine_grid(theta, image.size(), align_corners=False)
            image_rot = F.grid_sample(image, grid, align_corners=False)
            
            # Predict
            pred_rot = torch.sigmoid(model(image_rot))
            
            # Rotate back
            theta_inv = torch.tensor([
                [np.cos(-angle_rad), -np.sin(-angle_rad), 0],
                [np.sin(-angle_rad), np.cos(-angle_rad), 0]
            ], dtype=torch.float32, device=device).unsqueeze(0)
            
            grid_inv = F.affine_grid(theta_inv, pred_rot.size(), align_corners=False)
            pred_rot_back = F.grid_sample(pred_rot, grid_inv, align_corners=False)
            
            predictions.append(pred_rot_back)
    
    # Average all predictions
    tta_pred = torch.stack(predictions).mean(dim=0)
    
    return tta_pred


def evaluate_with_tta(model_path, test_loader, device):
    """Evaluate model with TTA"""
    
    print(f"\n{'='*70}")
    print("🔮 TEST-TIME AUGMENTATION EVALUATION")
    print(f"{'='*70}\n")
    
    # Load model
    print(f"📦 Loading model from: {model_path}")
    model = get_attention_unet(cfg=None).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   ✅ Model loaded\n")
    
    # Evaluate
    all_dice_scores = []
    
    print("🔬 Evaluating with TTA (6 augmentations)...")
    for batch in tqdm(test_loader, desc="Processing"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Get TTA prediction
        pred_probs = tta_predict(model, images, device)
        pred_binary = (pred_probs > 0.5).float()
        
        # Calculate Dice
        for i in range(pred_binary.shape[0]):
            pred = pred_binary[i].view(-1)
            target = masks[i].view(-1)
            
            intersection = (pred * target).sum()
            dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
            all_dice_scores.append(dice.item())
    
    # Statistics
    dice_mean = np.mean(all_dice_scores)
    dice_std = np.std(all_dice_scores)
    dice_min = np.min(all_dice_scores)
    dice_max = np.max(all_dice_scores)
    
    # Print results
    print(f"\n{'='*70}")
    print("📊 TTA RESULTS")
    print(f"{'='*70}\n")
    print(f"DICE (MEAN ± STD): {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"DICE (MIN/MAX)   : {dice_min:.4f} / {dice_max:.4f}\n")
    
    # Compare with baseline
    baseline_dice = 0.6231  # Round 10 without TTA
    improvement = dice_mean - baseline_dice
    improvement_pct = (improvement / baseline_dice) * 100
    
    print(f"📈 Improvement:")
    print(f"   Baseline (no TTA): {baseline_dice:.4f}")
    print(f"   With TTA:          {dice_mean:.4f}")
    print(f"   Gain:              {improvement:+.4f} ({improvement_pct:+.2f}%)\n")
    
    return dice_mean


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test loader
    test_dataset = DWIDataset25D(split='test', augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # TTA works best with batch_size=1
        shuffle=False,
        num_workers=2
    )
    
    # Evaluate
    model_path = MODEL_WEIGHTS / "best_model.pth"
    evaluate_with_tta(model_path, test_loader, device)


if __name__ == "__main__":
    main()
