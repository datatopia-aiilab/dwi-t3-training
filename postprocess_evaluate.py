"""
Post-processing for Segmentation Predictions
Applies morphological operations and CRF to refine predictions
"""

import torch
import numpy as np
from scipy import ndimage
from skimage import morphology
import cv2


def remove_small_objects(mask, min_size=50):
    """Remove small disconnected objects"""
    mask_np = mask.cpu().numpy().astype(bool)
    cleaned = morphology.remove_small_objects(mask_np, min_size=min_size)
    return torch.from_numpy(cleaned.astype(np.float32)).to(mask.device)


def fill_holes(mask):
    """Fill holes in segmentation"""
    mask_np = mask.cpu().numpy().astype(bool)
    filled = ndimage.binary_fill_holes(mask_np)
    return torch.from_numpy(filled.astype(np.float32)).to(mask.device)


def morphological_closing(mask, kernel_size=3):
    """Apply morphological closing (dilation + erosion)"""
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    return torch.from_numpy((closed / 255.0).astype(np.float32)).to(mask.device)


def morphological_opening(mask, kernel_size=3):
    """Apply morphological opening (erosion + dilation)"""
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    return torch.from_numpy((opened / 255.0).astype(np.float32)).to(mask.device)


def postprocess_prediction(pred_mask, 
                          remove_small=True, 
                          min_object_size=50,
                          fill_holes_flag=True,
                          apply_closing=True,
                          closing_kernel=3):
    """
    Apply full post-processing pipeline
    
    Args:
        pred_mask: Binary prediction mask (B, 1, H, W)
        remove_small: Remove small objects
        min_object_size: Minimum object size in pixels
        fill_holes_flag: Fill holes in objects
        apply_closing: Apply morphological closing
        closing_kernel: Kernel size for closing
    
    Returns:
        Refined binary mask
    """
    refined_masks = []
    
    for i in range(pred_mask.shape[0]):
        mask = pred_mask[i, 0]
        
        # 1. Morphological closing (smooth boundaries)
        if apply_closing:
            mask = morphological_closing(mask, closing_kernel)
        
        # 2. Fill holes
        if fill_holes_flag:
            mask = fill_holes(mask)
        
        # 3. Remove small objects
        if remove_small:
            mask = remove_small_objects(mask, min_size=min_object_size)
        
        refined_masks.append(mask)
    
    return torch.stack(refined_masks).unsqueeze(1)


def evaluate_with_postprocessing(model, test_loader, device, 
                                 min_object_size=50,
                                 closing_kernel=3):
    """Evaluate model with post-processing"""
    
    from tqdm import tqdm
    
    all_dice_before = []
    all_dice_after = []
    
    model.eval()
    
    print(f"\n{'='*70}")
    print("🔧 POST-PROCESSING EVALUATION")
    print(f"{'='*70}\n")
    print(f"Settings:")
    print(f"  - Remove small objects: min_size={min_object_size}")
    print(f"  - Fill holes: True")
    print(f"  - Morphological closing: kernel={closing_kernel}x{closing_kernel}\n")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Predict
            outputs = model(images)
            pred_probs = torch.sigmoid(outputs)
            pred_binary = (pred_probs > 0.5).float()
            
            # Before post-processing
            for i in range(pred_binary.shape[0]):
                pred = pred_binary[i].view(-1)
                target = masks[i].view(-1)
                
                intersection = (pred * target).sum()
                dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
                all_dice_before.append(dice.item())
            
            # Apply post-processing
            pred_refined = postprocess_prediction(
                pred_binary,
                remove_small=True,
                min_object_size=min_object_size,
                fill_holes_flag=True,
                apply_closing=True,
                closing_kernel=closing_kernel
            )
            
            # After post-processing
            for i in range(pred_refined.shape[0]):
                pred = pred_refined[i].view(-1)
                target = masks[i].view(-1)
                
                intersection = (pred * target).sum()
                dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
                all_dice_after.append(dice.item())
    
    # Results
    dice_before_mean = np.mean(all_dice_before)
    dice_after_mean = np.mean(all_dice_after)
    improvement = dice_after_mean - dice_before_mean
    
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print(f"{'='*70}\n")
    print(f"Before post-processing: {dice_before_mean:.4f}")
    print(f"After post-processing:  {dice_after_mean:.4f}")
    print(f"Improvement:            {improvement:+.4f} ({(improvement/dice_before_mean)*100:+.2f}%)\n")
    
    return dice_after_mean


if __name__ == "__main__":
    from config import *
    from model import get_attention_unet
    from dataset import DWIDataset25D
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_attention_unet(cfg=None).to(device)
    checkpoint = torch.load(MODEL_WEIGHTS / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test loader
    test_dataset = DWIDataset25D(split='test', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Evaluate
    evaluate_with_postprocessing(model, test_loader, device)
