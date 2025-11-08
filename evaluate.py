"""
Evaluation Script for DWI Ischemic Stroke Segmentation
Evaluates model on test set and generates visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import argparse
import json

# Import our modules
import config
from models import get_model  # ⬆️ Updated to support multiple architectures
from dataset import create_dataloaders
from utils import (
    calculate_all_metrics,
    visualize_sample,
    plot_training_curves,
    load_training_history,
    print_metrics_table
)


def load_best_model(model_path, device):
    """
    Load the best trained model
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
        checkpoint: Checkpoint dict with metadata
    """
    print(f"\n📂 Loading model from: {model_path}")
    
    # Load checkpoint (weights_only=False for PyTorch 2.6+)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model using get_model() to support all architectures
    model = get_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Val Dice: {checkpoint['val_dice']:.4f}")
    
    return model, checkpoint


def evaluate_on_test_set(model, test_loader, device):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device (cuda/cpu)
    
    Returns:
        dict: Aggregated metrics and per-sample results
    """
    print("\n" + "="*70)
    print("🧪 EVALUATING ON TEST SET")
    print("="*70)
    
    model.eval()
    
    all_metrics = []
    sample_results = []
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
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
            
            # Calculate metrics for this sample
            sample_metrics = calculate_all_metrics(preds_np, masks_np)
            all_metrics.append(sample_metrics)
            
            # Store results for visualization
            sample_results.append({
                'image': images.cpu().numpy(),
                'mask': masks_np,
                'prediction': preds_np,
                'output_prob': outputs.cpu().numpy(),
                'metrics': sample_metrics
            })
    
    # Aggregate metrics
    aggregated_metrics = {}
    for key in all_metrics[0].keys():
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


def plot_metrics_distribution(all_metrics, save_dir):
    """
    Plot distribution of metrics across test set
    
    Args:
        all_metrics: List of metric dicts
        save_dir: Directory to save plots
    """
    print("\n📊 Plotting metrics distribution...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metrics_to_plot):
        values = [m[metric_name] for m in all_metrics]
        
        ax = axes[idx]
        ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(values), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
        ax.axvline(np.median(values), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.3f}')
        ax.set_xlabel(metric_name.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{metric_name.upper()} Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    save_path = save_dir / 'metrics_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved to: {save_path}")


def generate_qualitative_results(sample_results, num_samples, save_dir):
    """
    Generate qualitative visualization of predictions
    
    Args:
        sample_results: List of sample result dicts
        num_samples: Number of samples to visualize
        save_dir: Directory to save plots
    """
    print(f"\n🎨 Generating qualitative results ({num_samples} samples)...")
    
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
    random_indices = np.random.choice(remaining, size=num_samples - len(best_indices) - len(worst_indices), replace=False)
    
    selected_indices = list(best_indices) + list(random_indices) + list(worst_indices)
    
    for i, idx in enumerate(tqdm(selected_indices, desc="Generating visualizations")):
        result = sample_results[idx]
        
        # Get data (take first sample from batch if batch_size > 1)
        image = result['image'][0]  # (3, H, W)
        mask = result['mask'][0, 0]  # (H, W)
        pred = result['prediction'][0, 0]  # (H, W)
        metrics = result['metrics']
        
        # Create title with metrics
        title = f"Sample {idx} - Dice: {metrics['dice']:.3f}, IoU: {metrics['iou']:.3f}"
        
        # Visualize
        fig = visualize_sample(
            image, mask, pred,
            title=title,
            alpha=config.VIZ_ALPHA,
            gt_color=config.VIZ_GT_COLOR,
            pred_color=config.VIZ_PRED_COLOR
        )
        
        # Save
        save_path = save_dir / f'sample_{idx:03d}_dice_{metrics["dice"]:.3f}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"   ✅ Saved {len(selected_indices)} visualizations to: {save_dir}")


def plot_training_history(history_path, save_dir):
    """
    Plot training curves from saved history
    
    Args:
        history_path: Path to training_history.json
        save_dir: Directory to save plots
    """
    print("\n📈 Plotting training curves...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load history
    history = load_training_history(history_path)
    
    # Plot curves
    fig = plot_training_curves(history)
    
    save_path = save_dir / 'training_curves.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   ✅ Saved to: {save_path}")


def save_evaluation_results(results, save_path):
    """
    Save evaluation results to JSON
    
    Args:
        results: Results dict
        save_path: Path to save JSON
    """
    save_path = Path(save_path)
    
    # Prepare data for JSON (convert numpy to python types)
    results_json = {
        'aggregated_metrics': {},
        'per_sample_metrics': []
    }
    
    # Aggregated metrics
    for metric_name, values in results['aggregated'].items():
        results_json['aggregated_metrics'][metric_name] = {
            k: float(v) for k, v in values.items()
        }
    
    # Per-sample metrics
    for sample_metrics in results['per_sample']:
        results_json['per_sample_metrics'].append({
            k: float(v) for k, v in sample_metrics.items()
        })
    
    # Save
    with open(save_path, 'w') as f:
        json.dump(results_json, f, indent=4)
    
    print(f"\n💾 Evaluation results saved to: {save_path}")


def print_evaluation_summary(results):
    """
    Print summary of evaluation results
    
    Args:
        results: Results dict from evaluate_on_test_set
    """
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    
    aggregated = results['aggregated']
    
    # Print metrics table
    summary = {}
    for metric_name, values in aggregated.items():
        summary[f"{metric_name.upper()} (mean ± std)"] = f"{values['mean']:.4f} ± {values['std']:.4f}"
        summary[f"{metric_name.upper()} (min/max)"] = f"{values['min']:.4f} / {values['max']:.4f}"
    
    print_metrics_table(summary, "Test Set Metrics")
    
    # Highlight key metrics
    print("\n🎯 Key Results:")
    print(f"   Dice Score:  {aggregated['dice']['mean']:.4f} ± {aggregated['dice']['std']:.4f}")
    print(f"   IoU Score:   {aggregated['iou']['mean']:.4f} ± {aggregated['iou']['std']:.4f}")
    print(f"   Precision:   {aggregated['precision']['mean']:.4f} ± {aggregated['precision']['std']:.4f}")
    print(f"   Recall:      {aggregated['recall']['mean']:.4f} ± {aggregated['recall']['std']:.4f}")
    
    # Check if target achieved
    target_dice = 0.95
    if aggregated['dice']['mean'] >= target_dice:
        print(f"\n✅ TARGET ACHIEVED! Dice Score ({aggregated['dice']['mean']:.4f}) >= {target_dice}")
    else:
        gap = target_dice - aggregated['dice']['mean']
        print(f"\n⚠️  Target not reached. Gap: {gap:.4f} (need {gap*100:.1f}% improvement)")
    
    print("="*70)


def main(args):
    """
    Main evaluation function
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("📊 DWI SEGMENTATION - MODEL EVALUATION")
    print("="*70)
    
    # Set device
    device = config.DEVICE
    print(f"\n💻 Using device: {device}")
    
    # Display current architecture configuration
    print(f"\n🏗️  Model Configuration:")
    print(f"   Architecture: {config.MODEL_ARCHITECTURE}")
    if config.MODEL_ARCHITECTURE != 'attention_unet':
        print(f"   Encoder: {config.ENCODER_NAME}")
        print(f"   Pre-trained: {config.ENCODER_WEIGHTS or 'None (random init)'}")
    
    # Plot only mode (skip evaluation)
    if args.plot_only:
        print("\n📈 Plot-only mode: Generating plots from existing results...")
        
        history_path = config.RESULTS_DIR / "training_history.json"
        if history_path.exists():
            plot_training_history(history_path, config.PLOTS_DIR)
        else:
            print(f"❌ Training history not found: {history_path}")
        
        print("\n✅ Done!")
        return
    
    # Check if model exists
    model_path = config.get_model_save_path('best_model')
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print(f"   Please train the model first: python train.py")
        return
    
    # Load model
    model, checkpoint = load_best_model(model_path, device)
    
    # Create dataloaders
    print("\n📦 Creating test dataloader...")
    dataloaders = create_dataloaders(config)
    test_loader = dataloaders['test']
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Evaluate on test set
    results = evaluate_on_test_set(model, test_loader, device)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results
    results_path = config.RESULTS_DIR / "test_results.json"
    save_evaluation_results(results, results_path)
    
    # Plot metrics distribution
    plot_metrics_distribution(results['per_sample'], config.PLOTS_DIR)
    
    # Generate qualitative results
    num_samples = min(args.num_samples, len(results['sample_results']))
    generate_qualitative_results(
        results['sample_results'],
        num_samples,
        config.PREDICTIONS_DIR
    )
    
    # Plot training curves
    history_path = config.RESULTS_DIR / "training_history.json"
    if history_path.exists():
        plot_training_history(history_path, config.PLOTS_DIR)
    else:
        print(f"\n⚠️  Training history not found: {history_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETED!")
    print("="*70)
    print(f"\n📁 Results saved to:")
    print(f"   Metrics: {results_path}")
    print(f"   Plots: {config.PLOTS_DIR}")
    print(f"   Predictions: {config.PREDICTIONS_DIR}")
    print("\n🎉 Check the visualization files to see the model's predictions!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DWI Segmentation Model")
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to visualize (default: 10)'
    )
    
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only generate plots from existing results (skip evaluation)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model checkpoint (default: use best_model.pth)'
    )
    
    args = parser.parse_args()
    
    # Override model path if provided
    if args.model_path:
        config.MODEL_WEIGHTS = Path(args.model_path).parent
    
    # Check if test data exists
    if not args.plot_only and not config.PROCESSED_TEST_IMG.exists():
        print("❌ Test data not found!")
        print(f"   Expected location: {config.PROCESSED_TEST_IMG}")
        print(f"\n   Please run preprocessing first:")
        print(f"   python 01_preprocess.py")
        exit(1)
    
    # Run evaluation
    main(args)
