"""
Ensemble Evaluation Script
Combines predictions from multiple models to improve performance
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from config import *
from model import get_attention_unet
from dataset import DWIDataset25D
from torch.utils.data import DataLoader


def load_model(checkpoint_path, device):
    """Load a trained model"""
    model = get_attention_unet(cfg=None).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def ensemble_predict(models, batch, device, method='average'):
    """
    Generate ensemble prediction
    
    Args:
        models: List of models
        batch: Input batch
        device: Device to use
        method: 'average', 'max', or 'vote'
    """
    images = batch['image'].to(device)
    predictions = []
    
    with torch.no_grad():
        for model in models:
            output = model(images)
            pred_prob = torch.sigmoid(output)
            predictions.append(pred_prob)
    
    # Stack predictions: (num_models, batch, 1, H, W)
    predictions = torch.stack(predictions)
    
    if method == 'average':
        # Average probabilities
        ensemble_pred = predictions.mean(dim=0)
    elif method == 'max':
        # Max probability
        ensemble_pred = predictions.max(dim=0)[0]
    elif method == 'vote':
        # Majority voting
        binary_preds = (predictions > 0.5).float()
        ensemble_pred = (binary_preds.mean(dim=0) > 0.5).float()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ensemble_pred


def dice_score(pred, target, smooth=1e-6):
    """Calculate Dice score"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def evaluate_ensemble(model_paths, test_loader, device, method='average'):
    """Evaluate ensemble of models"""
    
    print(f"\n{'='*70}")
    print(f"🎯 ENSEMBLE EVALUATION - Method: {method.upper()}")
    print(f"{'='*70}\n")
    
    # Load all models
    print("📦 Loading models...")
    models = []
    for i, path in enumerate(model_paths, 1):
        print(f"   Model {i}: {Path(path).name}")
        model = load_model(path, device)
        models.append(model)
    
    print(f"   ✅ Loaded {len(models)} models\n")
    
    # Evaluate
    all_dice_scores = []
    all_iou_scores = []
    all_precision_scores = []
    all_recall_scores = []
    
    print("🔬 Evaluating ensemble...")
    for batch in tqdm(test_loader, desc="Processing"):
        masks = batch['mask'].to(device)
        
        # Get ensemble prediction
        pred_probs = ensemble_predict(models, batch, device, method)
        pred_binary = (pred_probs > 0.5).float()
        
        # Calculate metrics
        for i in range(pred_binary.shape[0]):
            pred = pred_binary[i]
            target = masks[i]
            
            # Dice
            dice = dice_score(pred, target)
            all_dice_scores.append(dice)
            
            # IoU
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            all_iou_scores.append(iou.item())
            
            # Precision & Recall
            tp = (pred * target).sum()
            fp = (pred * (1 - target)).sum()
            fn = ((1 - pred) * target).sum()
            
            precision = (tp + 1e-6) / (tp + fp + 1e-6)
            recall = (tp + 1e-6) / (tp + fn + 1e-6)
            
            all_precision_scores.append(precision.item())
            all_recall_scores.append(recall.item())
    
    # Calculate statistics
    dice_mean = np.mean(all_dice_scores)
    dice_std = np.std(all_dice_scores)
    dice_min = np.min(all_dice_scores)
    dice_max = np.max(all_dice_scores)
    
    iou_mean = np.mean(all_iou_scores)
    iou_std = np.std(all_iou_scores)
    
    precision_mean = np.mean(all_precision_scores)
    precision_std = np.std(all_precision_scores)
    
    recall_mean = np.mean(all_recall_scores)
    recall_std = np.std(all_recall_scores)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"📊 ENSEMBLE RESULTS")
    print(f"{'='*70}\n")
    print(f"{'='*50}")
    print(f"📊 Ensemble Test Set Metrics ({method.upper()})")
    print(f"{'='*50}")
    print(f"DICE (MEAN ± STD): {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"DICE (MIN/MAX)   : {dice_min:.4f} / {dice_max:.4f}")
    print(f"IOU (MEAN ± STD) : {iou_mean:.4f} ± {iou_std:.4f}")
    print(f"PRECISION (MEAN ± STD): {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"RECALL (MEAN ± STD)   : {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"{'='*50}\n")
    
    # Calculate improvement
    baseline_dice = 0.6231  # Round 10 single model
    improvement = dice_mean - baseline_dice
    improvement_pct = (improvement / baseline_dice) * 100
    
    print(f"🎯 Key Results:")
    print(f"   Dice Score:  {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"   IoU Score:   {iou_mean:.4f} ± {iou_std:.4f}")
    print(f"   Precision:   {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"   Recall:      {recall_mean:.4f} ± {recall_std:.4f}\n")
    
    print(f"📈 Improvement over single model:")
    print(f"   Baseline (Round 10): {baseline_dice:.4f}")
    print(f"   Ensemble:            {dice_mean:.4f}")
    print(f"   Gain:                {improvement:+.4f} ({improvement_pct:+.2f}%)\n")
    
    target_dice = 0.95
    gap = target_dice - dice_mean
    gap_pct = (gap / target_dice) * 100
    
    if dice_mean >= target_dice:
        print(f"✅ Target reached! Dice: {dice_mean:.4f} >= {target_dice}")
    else:
        print(f"⚠️  Target not reached. Gap: {gap:.4f} (need {gap_pct:.1f}% improvement)")
    
    print(f"{'='*70}\n")
    
    # Save results
    results = {
        'method': method,
        'num_models': len(models),
        'model_paths': [str(p) for p in model_paths],
        'dice_mean': float(dice_mean),
        'dice_std': float(dice_std),
        'dice_min': float(dice_min),
        'dice_max': float(dice_max),
        'iou_mean': float(iou_mean),
        'iou_std': float(iou_std),
        'precision_mean': float(precision_mean),
        'precision_std': float(precision_std),
        'recall_mean': float(recall_mean),
        'recall_std': float(recall_std),
        'improvement_over_baseline': float(improvement),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = RESULTS_DIR / f'ensemble_results_{method}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {results_path}")
    
    return results


def main():
    """Main ensemble evaluation"""
    
    print(f"\n{'='*70}")
    print("🎯 ENSEMBLE MODEL EVALUATION")
    print(f"{'='*70}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}\n")
    
    # Define model checkpoints
    # จะใช้ best models จาก rounds ที่ดีที่สุด
    model_paths = [
        MODEL_WEIGHTS / "best_model.pth",  # Round 10 (Test 62%)
        # เพิ่ม models อื่น ๆ ถ้ามี
    ]
    
    # ตรวจสอบว่า model มีจริง
    existing_models = []
    for path in model_paths:
        if path.exists():
            existing_models.append(path)
        else:
            print(f"⚠️  Model not found: {path}")
    
    if len(existing_models) < 2:
        print(f"\n❌ Need at least 2 models for ensemble (found {len(existing_models)})")
        print("   Please train more models with different configurations first.")
        print("\n💡 Suggestions:")
        print("   1. Train with different random seeds")
        print("   2. Train with different augmentation settings")
        print("   3. Train with different model architectures")
        return
    
    print(f"✅ Found {len(existing_models)} models\n")
    
    # Create test dataset
    print("📦 Creating test dataloader...")
    test_dataset = DWIDataset25D(
        split='test',
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"   Test samples: {len(test_dataset)}\n")
    
    # Evaluate different ensemble methods
    methods = ['average', 'max']
    
    all_results = {}
    for method in methods:
        results = evaluate_ensemble(existing_models, test_loader, device, method)
        all_results[method] = results
    
    # Compare methods
    print(f"\n{'='*70}")
    print("🏆 ENSEMBLE METHOD COMPARISON")
    print(f"{'='*70}\n")
    
    for method, results in all_results.items():
        print(f"{method.upper():10s}: Dice = {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    
    # Find best method
    best_method = max(all_results.items(), key=lambda x: x[1]['dice_mean'])
    print(f"\n🥇 Best method: {best_method[0].upper()} (Dice = {best_method[1]['dice_mean']:.4f})")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
