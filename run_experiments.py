"""
Automated Experiment Runner for DWI Segmentation
Runs multiple training experiments with different configurations

Usage:
    python run_experiments.py --stage 1
    python run_experiments.py --stage 2 --top-models "manet,deeplabv3+,fpn"
    python run_experiments.py --config experiments_custom.json
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class ExperimentRunner:
    """Manages and runs multiple training experiments"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.config_file = base_dir / "config.py"
        self.results_file = base_dir / "experiment_results.json"
        self.log_dir = base_dir / "experiment_logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Load existing results if any
        self.results = self.load_results()
    
    def load_results(self) -> Dict[str, Any]:
        """Load previous experiment results"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {'experiments': [], 'metadata': {'created': datetime.now().isoformat()}}
    
    def save_results(self):
        """Save experiment results"""
        self.results['metadata']['last_updated'] = datetime.now().isoformat()
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def update_config(self, params: Dict[str, Any]):
        """Update config.py with new parameters"""
        # Read current config
        with open(self.config_file, 'r') as f:
            lines = f.readlines()
        
        # Update parameters
        new_lines = []
        for line in lines:
            modified = False
            for key, value in params.items():
                if line.strip().startswith(f"{key} ="):
                    # Format value based on type
                    if isinstance(value, str):
                        new_line = f"{key} = '{value}'\n"
                    elif isinstance(value, bool):
                        new_line = f"{key} = {value}\n"
                    elif isinstance(value, tuple):
                        new_line = f"{key} = {value}\n"
                    else:
                        new_line = f"{key} = {value}\n"
                    new_lines.append(new_line)
                    modified = True
                    break
            if not modified:
                new_lines.append(line)
        
        # Write updated config
        with open(self.config_file, 'w') as f:
            f.writelines(new_lines)
    
    def run_preprocessing(self, image_size: tuple) -> bool:
        """Run preprocessing with specified image size"""
        print(f"\n{'='*70}")
        print(f"🔧 RUNNING PREPROCESSING (Image Size: {image_size})")
        print(f"{'='*70}\n")
        
        try:
            # Update image size in config
            self.update_config({'IMAGE_SIZE': image_size})
            
            # Run preprocessing
            result = subprocess.run(
                ['python', '01_preprocess.py'],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )
            
            if result.returncode == 0:
                print("✅ Preprocessing completed successfully")
                return True
            else:
                print(f"❌ Preprocessing failed:\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Preprocessing timeout (>10 min)")
            return False
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return False
    
    def run_training(self, exp_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run single training experiment"""
        print(f"\n{'='*70}")
        print(f"🚀 RUNNING EXPERIMENT: {exp_id}")
        print(f"{'='*70}")
        print(f"Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*70}\n")
        
        # Update config with experiment parameters
        self.update_config(params)
        
        # Create log file
        log_file = self.log_dir / f"{exp_id}.log"
        
        try:
            start_time = time.time()
            
            # Run training
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    ['python', 'train.py'],
                    cwd=self.base_dir,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=7200  # 2 hours max
                )
            
            elapsed_time = time.time() - start_time
            
            # Parse results from log
            results = self.parse_training_log(log_file)
            results.update({
                'experiment_id': exp_id,
                'parameters': params,
                'success': result.returncode == 0,
                'training_time_seconds': elapsed_time,
                'timestamp': datetime.now().isoformat(),
                'log_file': str(log_file)
            })
            
            if result.returncode == 0:
                print(f"\n✅ Experiment {exp_id} completed successfully!")
                print(f"   Training time: {elapsed_time/60:.1f} minutes")
                if 'best_val_dice' in results:
                    print(f"   Best Val Dice: {results['best_val_dice']:.4f}")
                if 'test_dice' in results:
                    print(f"   Test Dice: {results['test_dice']:.4f}")
            else:
                print(f"\n❌ Experiment {exp_id} failed!")
            
            return results
            
        except subprocess.TimeoutExpired:
            return {
                'experiment_id': exp_id,
                'parameters': params,
                'success': False,
                'error': 'Training timeout (>2 hours)',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'experiment_id': exp_id,
                'parameters': params,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def parse_training_log(self, log_file: Path) -> Dict[str, Any]:
        """Parse training log to extract key metrics"""
        results = {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract best validation dice
            if 'Best validation Dice:' in content:
                for line in content.split('\n'):
                    if 'Best validation Dice:' in line:
                        try:
                            val_dice = float(line.split(':')[-1].split()[0])
                            results['best_val_dice'] = val_dice
                        except:
                            pass
            
            # Extract test metrics
            if 'test_dice' in content.lower():
                for line in content.split('\n'):
                    if 'DICE' in line and 'Mean ± Std' in content[max(0, content.find(line)-200):content.find(line)+200]:
                        try:
                            # Try to extract dice score from aggregated metrics
                            parts = line.split()
                            if len(parts) >= 3:
                                dice_str = parts[1] if parts[1].replace('.','').isdigit() else parts[2]
                                results['test_dice'] = float(dice_str.split('±')[0])
                        except:
                            pass
            
            # Extract training time
            if 'Total training time:' in content:
                for line in content.split('\n'):
                    if 'Total training time:' in line:
                        try:
                            time_min = float(line.split(':')[-1].split()[0])
                            results['training_time_minutes'] = time_min
                        except:
                            pass
            
        except Exception as e:
            print(f"⚠️  Warning: Could not parse log file: {e}")
        
        return results
    
    def run_experiments(self, experiments: List[Dict[str, Any]], skip_existing: bool = True):
        """Run multiple experiments"""
        print(f"\n{'='*70}")
        print(f"🧪 STARTING EXPERIMENT BATCH")
        print(f"{'='*70}")
        print(f"Total experiments: {len(experiments)}")
        print(f"Skip existing: {skip_existing}")
        print(f"{'='*70}\n")
        
        completed = 0
        failed = 0
        skipped = 0
        
        for i, exp_config in enumerate(experiments, 1):
            exp_id = exp_config['id']
            params = exp_config['params']
            
            # Check if already completed
            if skip_existing:
                existing = [r for r in self.results['experiments'] if r.get('experiment_id') == exp_id]
                if existing and existing[0].get('success'):
                    print(f"\n⏭️  Skipping experiment {exp_id} (already completed)")
                    skipped += 1
                    continue
            
            print(f"\n{'='*70}")
            print(f"Progress: {i}/{len(experiments)}")
            print(f"Completed: {completed} | Failed: {failed} | Skipped: {skipped}")
            print(f"{'='*70}")
            
            # Check if preprocessing needed
            current_image_size = params.get('IMAGE_SIZE')
            if current_image_size:
                # Check if we need to preprocess
                needs_preprocess = True
                if self.results['experiments']:
                    last_exp = self.results['experiments'][-1]
                    if last_exp.get('parameters', {}).get('IMAGE_SIZE') == current_image_size:
                        needs_preprocess = False
                
                if needs_preprocess:
                    print(f"\n📦 Image size changed to {current_image_size}, running preprocessing...")
                    if not self.run_preprocessing(current_image_size):
                        print(f"❌ Preprocessing failed, skipping experiment {exp_id}")
                        failed += 1
                        continue
            
            # Run experiment
            result = self.run_training(exp_id, params)
            
            # Save results
            self.results['experiments'].append(result)
            self.save_results()
            
            if result['success']:
                completed += 1
            else:
                failed += 1
            
            # Brief pause between experiments
            time.sleep(5)
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"🎉 EXPERIMENT BATCH COMPLETED")
        print(f"{'='*70}")
        print(f"Total: {len(experiments)}")
        print(f"✅ Completed: {completed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"{'='*70}\n")
        
        # Show top results
        self.show_top_results(n=5)
    
    def show_top_results(self, n: int = 5):
        """Show top N results"""
        successful = [r for r in self.results['experiments'] if r.get('success') and 'best_val_dice' in r]
        if not successful:
            print("No successful experiments yet.")
            return
        
        # Sort by validation dice
        sorted_results = sorted(successful, key=lambda x: x.get('best_val_dice', 0), reverse=True)
        
        print(f"\n{'='*70}")
        print(f"🏆 TOP {n} EXPERIMENTS (by Val Dice)")
        print(f"{'='*70}\n")
        
        for i, result in enumerate(sorted_results[:n], 1):
            print(f"{i}. {result['experiment_id']}")
            print(f"   Val Dice: {result.get('best_val_dice', 0):.4f}")
            if 'test_dice' in result:
                print(f"   Test Dice: {result.get('test_dice', 0):.4f}")
            print(f"   Architecture: {result['parameters'].get('MODEL_ARCHITECTURE', 'N/A')}")
            print(f"   Encoder: {result['parameters'].get('ENCODER_NAME', 'N/A')}")
            print(f"   Image Size: {result['parameters'].get('IMAGE_SIZE', 'N/A')}")
            print(f"   Time: {result.get('training_time_minutes', 0):.1f} min")
            print()


def generate_stage1_experiments() -> List[Dict[str, Any]]:
    """Stage 1: Architecture + Encoder Selection"""
    architectures = ['attention_unet', 'unet++', 'fpn', 'deeplabv3+', 'manet', 'pspnet']
    encoders = ['efficientnet-b0', 'efficientnet-b3', 'resnet34']  # 3 best options
    
    experiments = []
    for arch in architectures:
        for enc in encoders:
            # Skip encoder for attention_unet (it has custom architecture)
            if arch == 'attention_unet':
                exp_id = f"s1_{arch}_custom"
                params = {
                    'MODEL_ARCHITECTURE': arch,
                    'IMAGE_SIZE': (384, 384),
                    'BATCH_SIZE': 16,
                    'NUM_EPOCHS': 200,
                    'AUGMENTATION_ENABLED': True,
                    'CLAHE_ENABLED': False
                }
                experiments.append({'id': exp_id, 'params': params})
                break  # Only one config for attention_unet
            else:
                exp_id = f"s1_{arch}_{enc}"
                params = {
                    'MODEL_ARCHITECTURE': arch,
                    'ENCODER_NAME': enc,
                    'IMAGE_SIZE': (384, 384),
                    'BATCH_SIZE': 16,
                    'NUM_EPOCHS': 200,
                    'AUGMENTATION_ENABLED': True,
                    'CLAHE_ENABLED': False
                }
                experiments.append({'id': exp_id, 'params': params})
    
    return experiments


def generate_stage2_experiments(top_models: List[str]) -> List[Dict[str, Any]]:
    """Stage 2: Resolution Optimization"""
    resolutions = [(256, 256), (384, 384), (512, 512)]
    batch_sizes = {(256, 256): 32, (384, 384): 16, (512, 512): 8}
    
    experiments = []
    for model in top_models:
        # Parse model string (format: "architecture_encoder" or just "architecture")
        parts = model.split('_')
        arch = parts[0]
        enc = parts[1] if len(parts) > 1 and parts[0] != 'attention' else 'efficientnet-b3'
        
        for res in resolutions:
            exp_id = f"s2_{arch}_{enc}_res{res[0]}"
            params = {
                'MODEL_ARCHITECTURE': arch,
                'IMAGE_SIZE': res,
                'BATCH_SIZE': batch_sizes[res],
                'NUM_EPOCHS': 200,
                'AUGMENTATION_ENABLED': True,
                'CLAHE_ENABLED': False
            }
            if arch != 'attention_unet':
                params['ENCODER_NAME'] = enc
            experiments.append({'id': exp_id, 'params': params})
    
    return experiments


def generate_stage3_experiments(best_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Stage 3: Preprocessing Optimization"""
    clahe_options = [False, True]
    aug_options = [False, True]
    
    experiments = []
    for clahe in clahe_options:
        for aug in aug_options:
            exp_id = f"s3_clahe{int(clahe)}_aug{int(aug)}"
            params = best_model.copy()
            params['CLAHE_ENABLED'] = clahe
            params['AUGMENTATION_ENABLED'] = aug
            experiments.append({'id': exp_id, 'params': params})
    
    return experiments


def generate_stage4_experiments(best_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Stage 4: Fine-tuning"""
    loss_types = ['dice', 'focal', 'combo']
    learning_rates = [5e-5, 8e-5, 1e-4]
    
    experiments = []
    for loss in loss_types:
        for lr in learning_rates:
            exp_id = f"s4_{loss}_lr{lr:.0e}"
            params = best_model.copy()
            params['LOSS_TYPE'] = loss
            params['LEARNING_RATE'] = lr
            experiments.append({'id': exp_id, 'params': params})
    
    return experiments


def main():
    parser = argparse.ArgumentParser(description='Run automated experiments')
    parser.add_argument('--stage', type=int, help='Run specific stage (1-4)')
    parser.add_argument('--config', type=str, help='Load custom experiment config JSON')
    parser.add_argument('--top-models', type=str, help='Comma-separated top models for stage 2')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip completed experiments')
    
    args = parser.parse_args()
    
    # Initialize runner
    base_dir = Path(__file__).parent
    runner = ExperimentRunner(base_dir)
    
    # Load custom config
    if args.config:
        with open(args.config, 'r') as f:
            experiments = json.load(f)
    # Generate stage experiments
    elif args.stage == 1:
        print("Generating Stage 1 experiments: Architecture + Encoder Selection")
        experiments = generate_stage1_experiments()
    elif args.stage == 2:
        if not args.top_models:
            print("Error: --top-models required for stage 2")
            print("Example: --top-models 'manet_efficientnet-b3,deeplabv3+_resnet34'")
            sys.exit(1)
        top_models = args.top_models.split(',')
        print(f"Generating Stage 2 experiments for: {top_models}")
        experiments = generate_stage2_experiments(top_models)
    elif args.stage == 3:
        print("Generating Stage 3 experiments: Preprocessing Optimization")
        # Need to get best model from stage 2
        successful = [r for r in runner.results['experiments'] if r.get('success') and 'best_val_dice' in r]
        if not successful:
            print("Error: No successful experiments found. Run stage 1 and 2 first.")
            sys.exit(1)
        best = max(successful, key=lambda x: x['best_val_dice'])
        experiments = generate_stage3_experiments(best['parameters'])
    elif args.stage == 4:
        print("Generating Stage 4 experiments: Fine-tuning")
        successful = [r for r in runner.results['experiments'] if r.get('success') and 'best_val_dice' in r]
        if not successful:
            print("Error: No successful experiments found. Run previous stages first.")
            sys.exit(1)
        best = max(successful, key=lambda x: x['best_val_dice'])
        experiments = generate_stage4_experiments(best['parameters'])
    else:
        print("Error: Must specify --stage (1-4) or --config")
        sys.exit(1)
    
    # Run experiments
    print(f"\nGenerated {len(experiments)} experiments")
    confirm = input("Start experiments? (yes/no): ")
    if confirm.lower() == 'yes':
        runner.run_experiments(experiments, skip_existing=args.skip_existing)
    else:
        print("Cancelled.")


if __name__ == '__main__':
    main()
