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
    load_training_history,
    print_metrics_table
)

# Import evaluation functions from shared module
from evaluation_module import (
    run_evaluation,
    generate_qualitative_results,
    print_evaluation_summary,
    save_per_sample_results_csv,
    plot_metrics_distribution
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


def plot_training_curves(history_path, save_dir):
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
    
    # Import plot function from utils
    from utils import plot_training_curves as plot_curves
    
    # Plot curves
    fig = plot_curves(history)
    
    save_path = save_dir / 'training_curves.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   ✅ Saved to: {save_path}")


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
            plot_training_curves(history_path, config.PLOTS_DIR)
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
    
    # Evaluate on test set using shared evaluation module
    print("\n" + "="*70)
    print("🧪 EVALUATING ON TEST SET")
    print("="*70)
    
    results = run_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
        config=config,
        show_progress=True
    )
    
    # Print summary using formatted output
    print_evaluation_summary(results, show_per_sample_top=5)
    
    # Save per-sample results to CSV
    csv_path = config.RESULTS_DIR / "test_per_sample_results.csv"
    save_per_sample_results_csv(results['per_sample'], csv_path)
    print(f"\n💾 Saved per-sample results to: {csv_path}")
    
    # Save aggregated results to JSON for backward compatibility
    results_json_path = config.RESULTS_DIR / "test_results.json"
    save_evaluation_results_json(results, results_json_path)
    
    # Plot metrics distribution
    plot_metrics_distribution(results['per_sample'], config.PLOTS_DIR)
    
    # Generate qualitative results
    num_samples = min(args.num_samples, len(results['sample_results']))
    print(f"\n🖼️  Generating prediction visualizations...")
    saved_images = generate_qualitative_results(
        sample_results=results['sample_results'],
        save_dir=config.PREDICTIONS_DIR,
        config=config,
        num_samples=num_samples
    )
    print(f"   ✅ Generated {len(saved_images)} prediction images in: {config.PREDICTIONS_DIR}")
    
    # Plot training curves
    history_path = config.RESULTS_DIR / "training_history.json"
    if history_path.exists():
        plot_training_curves(history_path, config.PLOTS_DIR)
    else:
        print(f"\n⚠️  Training history not found: {history_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETED!")
    print("="*70)
    print(f"\n📁 Results saved to:")
    print(f"   CSV: {csv_path}")
    print(f"   JSON: {results_json_path}")
    print(f"   Plots: {config.PLOTS_DIR}")
    print(f"   Predictions: {config.PREDICTIONS_DIR}")
    print("\n🎉 Check the visualization files to see the model's predictions!")
    print("="*70 + "\n")


def save_evaluation_results_json(results, save_path):
    """
    Save evaluation results to JSON (for backward compatibility)
    
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
        sample_dict = {k: float(v) for k, v in sample_metrics.items() if k != 'filename'}
        if 'filename' in sample_metrics:
            sample_dict['filename'] = sample_metrics['filename']
        results_json['per_sample_metrics'].append(sample_dict)
    
    # Save
    with open(save_path, 'w') as f:
        json.dump(results_json, f, indent=4)
    
    print(f"   Metrics JSON: {save_path}")


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
