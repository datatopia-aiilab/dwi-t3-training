"""
Evaluation Module for DWI Segmentation
Reusable evaluation functions that can be called from train.py or evaluate.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from utils import (
    calculate_all_metrics, 
    visualize_sample_advanced,
    calculate_infarction_volume
)


def run_evaluation(model, test_loader, device, config, show_progress=True):
    """
    Run evaluation on test set
    
    Args:
        model: Trained model (already in eval mode)
        test_loader: Test dataloader
        device: Device (cuda/cpu)
        config: Configuration module
        show_progress: Show progress bar
    
    Returns:
        dict: {
            'aggregated': aggregated metrics,
            'per_sample': list of per-sample metrics with filenames,
            'sample_results': list of sample results for visualization
        }
    """
    model.eval()
    
    all_metrics = []
    sample_results = []
    
    iterator = tqdm(test_loader, desc="Evaluating on test set") if show_progress else test_loader
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(iterator):
            # Unpack batch (now includes filenames)
            if len(batch_data) == 3:
                images, masks, filenames = batch_data
            else:
                # Fallback for old format
                images, masks = batch_data
                filenames = [f"sample_{batch_idx:03d}"] * images.shape[0]
            
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Convert to binary predictions
            preds = (outputs > config.PREDICTION_THRESHOLD).float()
            
            # Move to CPU for metrics calculation
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
            # Process each sample in batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                # Calculate metrics for this sample
                sample_pred = preds_np[i:i+1]
                sample_mask = masks_np[i:i+1]
                sample_metrics = calculate_all_metrics(sample_pred, sample_mask)
                
                # Add filename to metrics
                filename = filenames[i] if isinstance(filenames, (list, tuple)) else filenames
                sample_metrics['filename'] = filename
                
                # Calculate volumes (using 4mm pixel spacing)
                gt_volume = calculate_infarction_volume(sample_mask[0], pixel_spacing=4.0, slice_thickness=4.0)
                pred_volume = calculate_infarction_volume(sample_pred[0], pixel_spacing=4.0, slice_thickness=4.0)
                volume_error = abs(pred_volume - gt_volume) / (gt_volume + 1e-6) * 100
                
                sample_metrics['gt_volume_ml'] = gt_volume
                sample_metrics['pred_volume_ml'] = pred_volume
                sample_metrics['volume_error_percent'] = volume_error
                
                all_metrics.append(sample_metrics)
                
                # Store results for visualization
                sample_results.append({
                    'filename': filename,
                    'image': images[i:i+1].cpu().numpy(),
                    'mask': sample_mask,
                    'prediction': sample_pred,
                    'output_prob': outputs_np[i:i+1],
                    'metrics': sample_metrics
                })
    
    # Aggregate metrics
    aggregated_metrics = {}
    metric_keys = [k for k in all_metrics[0].keys() if k != 'filename']
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        aggregated_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    return {
        'aggregated': aggregated_metrics,
        'per_sample': all_metrics,
        'sample_results': sample_results
    }


def generate_qualitative_results(sample_results, save_dir, config, num_samples=10):
    """
    Generate qualitative visualization of predictions
    
    Args:
        sample_results: List of sample result dicts
        save_dir: Directory to save visualizations
        config: Configuration module
        num_samples: Number of samples to visualize
    
    Returns:
        list: Paths to generated images
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Select samples: best, worst, and random
    dice_scores = [r['metrics']['dice'] for r in sample_results]
    
    # Best samples
    best_indices = np.argsort(dice_scores)[-num_samples//3:][::-1]
    
    # Worst samples  
    worst_indices = np.argsort(dice_scores)[:num_samples//3]
    
    # Random samples
    remaining = list(set(range(len(sample_results))) - set(best_indices) - set(worst_indices))
    if len(remaining) > 0:
        n_random = min(num_samples - len(best_indices) - len(worst_indices), len(remaining))
        random_indices = np.random.choice(remaining, size=n_random, replace=False)
    else:
        random_indices = []
    
    selected_indices = list(best_indices) + list(random_indices) + list(worst_indices)
    
    saved_paths = []
    
    for i, idx in enumerate(selected_indices):
        result = sample_results[idx]
        
        # Get data
        image = result['image'][0]  # (3, H, W)
        mask = result['mask'][0, 0]  # (H, W)
        pred = result['prediction'][0, 0]  # (H, W)
        metrics = result['metrics']
        filename = result['filename']
        
        # Remove .npy extension if present
        clean_filename = filename.replace('.npy', '')
        
        # Visualize with advanced 4-panel layout
        fig = visualize_sample_advanced(
            image, mask, pred,
            filename=f"{clean_filename} | Dice: {metrics['dice']:.3f} | IoU: {metrics['iou']:.3f}",
            pixel_spacing=4.0,
            slice_thickness=4.0
        )
        
        # Save with filename in path
        save_path = save_dir / f'{clean_filename}_dice_{metrics["dice"]:.3f}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        saved_paths.append(save_path)
    
    return saved_paths


def print_evaluation_summary(results, show_per_sample_top=5):
    """
    Print beautiful evaluation summary with tables
    
    Args:
        results: Results dict from run_evaluation
        show_per_sample_top: Number of top/worst samples to show
    """
    print("\n" + "="*80)
    print("📊 TEST SET EVALUATION RESULTS")
    print("="*80)
    
    aggregated = results['aggregated']
    per_sample = results['per_sample']
    
    # Main metrics table
    print("\n🎯 Aggregated Metrics:")
    print(f"{'─'*80}")
    print(f"{'Metric':<20} {'Mean ± Std':<25} {'Min':<12} {'Max':<12} {'Median':<12}")
    print(f"{'─'*80}")
    
    for metric_name in ['dice', 'iou', 'precision', 'recall', 'f1']:
        if metric_name in aggregated:
            values = aggregated[metric_name]
            print(f"{metric_name.upper():<20} "
                  f"{values['mean']:.4f} ± {values['std']:.4f}      "
                  f"{values['min']:.4f}      "
                  f"{values['max']:.4f}      "
                  f"{values['median']:.4f}")
    print(f"{'─'*80}")
    
    # Top 5 Best Samples
    if show_per_sample_top > 0:
        sorted_samples = sorted(per_sample, key=lambda x: x['dice'], reverse=True)
        
        print(f"\n🏆 TOP {show_per_sample_top} BEST SAMPLES:")
        print(f"{'─'*80}")
        print(f"{'Rank':<6} {'Filename':<35} {'Dice':<10} {'IoU':<10} {'F1':<10}")
        print(f"{'─'*80}")
        
        for rank, sample in enumerate(sorted_samples[:show_per_sample_top], 1):
            filename = sample['filename'].replace('.npy', '')
            print(f"{rank:<6} {filename:<35} "
                  f"{sample['dice']:.4f}     "
                  f"{sample['iou']:.4f}     "
                  f"{sample['f1']:.4f}")
        print(f"{'─'*80}")
        
        # Top 5 Worst Samples
        print(f"\n⚠️  TOP {show_per_sample_top} WORST SAMPLES:")
        print(f"{'─'*80}")
        print(f"{'Rank':<6} {'Filename':<35} {'Dice':<10} {'IoU':<10} {'F1':<10}")
        print(f"{'─'*80}")
        
        for rank, sample in enumerate(sorted_samples[-show_per_sample_top:][::-1], 1):
            filename = sample['filename'].replace('.npy', '')
            print(f"{rank:<6} {filename:<35} "
                  f"{sample['dice']:.4f}     "
                  f"{sample['iou']:.4f}     "
                  f"{sample['f1']:.4f}")
        print(f"{'─'*80}")
    
    # Target check
    target_dice = 0.95
    mean_dice = aggregated['dice']['mean']
    
    print(f"\n🎯 Target Analysis:")
    if mean_dice >= target_dice:
        print(f"   ✅ TARGET ACHIEVED! Test Dice ({mean_dice:.4f}) >= {target_dice}")
    else:
        gap = target_dice - mean_dice
        improvement_pct = (gap / mean_dice) * 100
        print(f"   ⚠️  Target: {target_dice:.4f}")
        print(f"   📊 Current: {mean_dice:.4f}")
        print(f"   📈 Gap: {gap:.4f} ({improvement_pct:.1f}% improvement needed)")
    
    print("="*80 + "\n")


def save_per_sample_results_csv(per_sample_metrics, save_path):
    """
    Save per-sample metrics to CSV file
    
    Args:
        per_sample_metrics: List of metric dicts with filenames
        save_path: Path to save CSV
    
    Returns:
        Path: Path to saved CSV
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(per_sample_metrics)
    
    # Reorder columns to put filename first
    cols = ['filename'] + [col for col in df.columns if col != 'filename']
    df = df[cols]
    
    # Sort by dice score descending
    df = df.sort_values('dice', ascending=False)
    
    # Save
    df.to_csv(save_path, index=False, float_format='%.6f')
    
    print(f"💾 Saved per-sample results to: {save_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Columns: {', '.join(df.columns)}")
    
    return save_path


def plot_metrics_distribution(per_sample_metrics, save_dir):
    """
    Plot distribution of metrics across test set
    
    Args:
        per_sample_metrics: List of metric dicts
        save_dir: Directory to save plots
    
    Returns:
        Path: Path to saved plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metrics_to_plot):
        values = [m[metric_name] for m in per_sample_metrics]
        
        ax = axes[idx]
        ax.hist(values, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        
        mean_val = np.mean(values)
        median_val = np.median(values)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_val:.3f}')
        
        ax.set_xlabel(metric_name.upper(), fontsize=13, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{metric_name.upper()} Distribution\n(n={len(values)} samples)', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    save_path = save_dir / 'test_metrics_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved metrics distribution plot to: {save_path}")
    
    return save_path
