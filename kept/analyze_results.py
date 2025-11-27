"""
Experiment Result Analyzer
Analyze and visualize experiment results from automated runs

Usage:
    python analyze_results.py
    python analyze_results.py --export report.html
    python analyze_results.py --compare exp1,exp2,exp3
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any


class ResultAnalyzer:
    """Analyze experiment results and generate reports"""
    
    def __init__(self, results_file: Path):
        self.results_file = results_file
        self.results = self.load_results()
        self.df = self.create_dataframe()
    
    def load_results(self) -> Dict[str, Any]:
        """Load experiment results"""
        if not self.results_file.exists():
            print(f"❌ Results file not found: {self.results_file}")
            return {'experiments': []}
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from results"""
        experiments = []
        for exp in self.results.get('experiments', []):
            if not exp.get('success'):
                continue
            
            row = {
                'experiment_id': exp.get('experiment_id'),
                'architecture': exp.get('parameters', {}).get('MODEL_ARCHITECTURE'),
                'encoder': exp.get('parameters', {}).get('ENCODER_NAME', 'custom'),
                'image_size': str(exp.get('parameters', {}).get('IMAGE_SIZE')),
                'batch_size': exp.get('parameters', {}).get('BATCH_SIZE'),
                'augmentation': exp.get('parameters', {}).get('AUGMENTATION_ENABLED'),
                'clahe': exp.get('parameters', {}).get('CLAHE_ENABLED'),
                'learning_rate': exp.get('parameters', {}).get('LEARNING_RATE'),
                'loss_type': exp.get('parameters', {}).get('LOSS_TYPE'),
                'val_dice': exp.get('best_val_dice'),
                'test_dice': exp.get('test_dice'),
                'training_time_min': exp.get('training_time_minutes'),
                'timestamp': exp.get('timestamp')
            }
            experiments.append(row)
        
        return pd.DataFrame(experiments)
    
    def print_summary(self):
        """Print summary statistics"""
        if self.df.empty:
            print("No successful experiments to analyze.")
            return
        
        print("\n" + "="*70)
        print("📊 EXPERIMENT SUMMARY")
        print("="*70 + "\n")
        
        print(f"Total experiments: {len(self.df)}")
        print(f"Architectures tested: {self.df['architecture'].nunique()}")
        print(f"Encoders tested: {self.df['encoder'].nunique()}")
        print(f"\nValidation Dice Score:")
        print(f"  Mean: {self.df['val_dice'].mean():.4f}")
        print(f"  Std:  {self.df['val_dice'].std():.4f}")
        print(f"  Min:  {self.df['val_dice'].min():.4f}")
        print(f"  Max:  {self.df['val_dice'].max():.4f}")
        
        if 'test_dice' in self.df.columns and self.df['test_dice'].notna().any():
            print(f"\nTest Dice Score:")
            print(f"  Mean: {self.df['test_dice'].mean():.4f}")
            print(f"  Std:  {self.df['test_dice'].std():.4f}")
            print(f"  Min:  {self.df['test_dice'].min():.4f}")
            print(f"  Max:  {self.df['test_dice'].max():.4f}")
        
        print(f"\nTraining Time:")
        print(f"  Mean: {self.df['training_time_min'].mean():.1f} min")
        print(f"  Total: {self.df['training_time_min'].sum():.1f} min ({self.df['training_time_min'].sum()/60:.1f} hours)")
        
        print("\n" + "="*70 + "\n")
    
    def show_top_results(self, n: int = 10):
        """Show top N results"""
        if self.df.empty:
            return
        
        print(f"\n🏆 TOP {n} EXPERIMENTS (by Val Dice)\n")
        top = self.df.nlargest(n, 'val_dice')
        
        for i, (idx, row) in enumerate(top.iterrows(), 1):
            print(f"{i}. {row['experiment_id']}")
            print(f"   Val Dice: {row['val_dice']:.4f}", end="")
            if pd.notna(row['test_dice']):
                print(f" | Test Dice: {row['test_dice']:.4f}", end="")
            print(f" | Time: {row['training_time_min']:.1f} min")
            print(f"   {row['architecture']} + {row['encoder']} @ {row['image_size']}")
            print(f"   Aug: {row['augmentation']} | CLAHE: {row['clahe']} | Loss: {row['loss_type']}")
            print()
    
    def compare_architectures(self):
        """Compare performance across architectures"""
        if self.df.empty:
            return
        
        print("\n📊 ARCHITECTURE COMPARISON\n")
        arch_stats = self.df.groupby('architecture').agg({
            'val_dice': ['mean', 'std', 'max'],
            'test_dice': ['mean', 'max'],
            'training_time_min': 'mean'
        }).round(4)
        
        print(arch_stats.to_string())
        print()
    
    def compare_encoders(self):
        """Compare performance across encoders"""
        if self.df.empty:
            return
        
        print("\n📊 ENCODER COMPARISON\n")
        enc_stats = self.df.groupby('encoder').agg({
            'val_dice': ['mean', 'std', 'max'],
            'test_dice': ['mean', 'max'],
            'training_time_min': 'mean'
        }).round(4)
        
        print(enc_stats.to_string())
        print()
    
    def compare_resolutions(self):
        """Compare performance across image resolutions"""
        if self.df.empty:
            return
        
        print("\n📊 RESOLUTION COMPARISON\n")
        res_stats = self.df.groupby('image_size').agg({
            'val_dice': ['mean', 'std', 'max'],
            'test_dice': ['mean', 'max'],
            'training_time_min': 'mean'
        }).round(4)
        
        print(res_stats.to_string())
        print()
    
    def analyze_preprocessing(self):
        """Analyze impact of preprocessing"""
        if self.df.empty:
            return
        
        print("\n📊 PREPROCESSING IMPACT\n")
        
        # Augmentation impact
        print("Augmentation:")
        aug_stats = self.df.groupby('augmentation')['val_dice'].agg(['mean', 'std', 'max']).round(4)
        print(aug_stats.to_string())
        print()
        
        # CLAHE impact
        print("CLAHE:")
        clahe_stats = self.df.groupby('clahe')['val_dice'].agg(['mean', 'std', 'max']).round(4)
        print(clahe_stats.to_string())
        print()
    
    def plot_results(self, output_dir: Path):
        """Generate visualization plots"""
        if self.df.empty:
            return
        
        output_dir.mkdir(exist_ok=True)
        
        # Plot 1: Architecture comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        arch_data = self.df.groupby('architecture')['val_dice'].apply(list)
        ax.boxplot(arch_data.values, labels=arch_data.index)
        ax.set_xlabel('Architecture', fontsize=12)
        ax.set_ylabel('Validation Dice Score', fontsize=12)
        ax.set_title('Architecture Performance Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'architecture_comparison.png', dpi=300)
        plt.close()
        
        # Plot 2: Encoder comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        enc_data = self.df.groupby('encoder')['val_dice'].apply(list)
        ax.boxplot(enc_data.values, labels=enc_data.index)
        ax.set_xlabel('Encoder', fontsize=12)
        ax.set_ylabel('Validation Dice Score', fontsize=12)
        ax.set_title('Encoder Performance Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'encoder_comparison.png', dpi=300)
        plt.close()
        
        # Plot 3: Training time vs performance
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(self.df['training_time_min'], self.df['val_dice'], 
                           c=self.df['val_dice'], cmap='viridis', s=100, alpha=0.6)
        ax.set_xlabel('Training Time (minutes)', fontsize=12)
        ax.set_ylabel('Validation Dice Score', fontsize=12)
        ax.set_title('Training Time vs Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Val Dice')
        plt.tight_layout()
        plt.savefig(output_dir / 'time_vs_performance.png', dpi=300)
        plt.close()
        
        # Plot 4: Resolution impact
        if self.df['image_size'].nunique() > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            res_data = self.df.groupby('image_size')['val_dice'].apply(list)
            ax.boxplot(res_data.values, labels=res_data.index)
            ax.set_xlabel('Image Size', fontsize=12)
            ax.set_ylabel('Validation Dice Score', fontsize=12)
            ax.set_title('Image Resolution Impact', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'resolution_comparison.png', dpi=300)
            plt.close()
        
        print(f"✅ Plots saved to: {output_dir}")
    
    def export_csv(self, output_file: Path):
        """Export results to CSV"""
        if self.df.empty:
            return
        
        self.df.to_csv(output_file, index=False)
        print(f"✅ Results exported to: {output_file}")
    
    def generate_report(self, output_file: Path):
        """Generate HTML report"""
        if self.df.empty:
            return
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #4CAF50; }}
            </style>
        </head>
        <body>
            <h1>🧪 Experiment Results Report</h1>
            <p>Generated: {pd.Timestamp.now()}</p>
            
            <h2>Summary Statistics</h2>
            <p>Total experiments: {len(self.df)}</p>
            <p>Best Val Dice: <span class="metric">{self.df['val_dice'].max():.4f}</span></p>
            <p>Best Test Dice: <span class="metric">{self.df['test_dice'].max():.4f}</span></p>
            
            <h2>Top 10 Results</h2>
            {self.df.nlargest(10, 'val_dice').to_html(index=False)}
            
            <h2>Architecture Comparison</h2>
            {self.df.groupby('architecture')['val_dice'].describe().to_html()}
            
            <h2>All Results</h2>
            {self.df.sort_values('val_dice', ascending=False).to_html(index=False)}
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✅ HTML report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results', type=str, default='experiment_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--export-csv', type=str, help='Export to CSV file')
    parser.add_argument('--export-html', type=str, help='Export to HTML report')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--compare', type=str, help='Compare specific experiments (comma-separated IDs)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    base_dir = Path(__file__).parent
    results_file = base_dir / args.results
    analyzer = ResultAnalyzer(results_file)
    
    # Print summary
    analyzer.print_summary()
    analyzer.show_top_results(n=10)
    
    # Comparisons
    analyzer.compare_architectures()
    analyzer.compare_encoders()
    analyzer.compare_resolutions()
    analyzer.analyze_preprocessing()
    
    # Export CSV
    if args.export_csv:
        analyzer.export_csv(Path(args.export_csv))
    
    # Export HTML
    if args.export_html:
        analyzer.generate_report(Path(args.export_html))
    
    # Generate plots
    if args.plot:
        plot_dir = base_dir / "experiment_plots"
        analyzer.plot_results(plot_dir)


if __name__ == '__main__':
    main()
