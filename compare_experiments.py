#!/usr/bin/env python3
"""
Compare results across different probe training experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_experiment_results(experiment_dir):
    """Load all results from an experiment directory."""
    experiment_dir = Path(experiment_dir)
    
    # Load config
    with open(experiment_dir / 'experiment_config.json') as f:
        config = json.load(f)
    
    # Load all layer results
    results = []
    for result_file in sorted(experiment_dir.glob('layer_*_results.json')):
        with open(result_file) as f:
            layer_result = json.load(f)
            results.append(layer_result)
    
    # Sort by layer
    results.sort(key=lambda x: x['layer'])
    
    return config, results


def plot_comparison(experiments, output_path='experiments_comparison.png'):
    """Plot comparison of multiple experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for (exp_name, config, results), color in zip(experiments, colors):
        layers = [r['layer'] for r in results]
        
        # Regression metrics
        train_rmse = [r['regression']['train_rmse'] for r in results]
        test_rmse = [r['regression']['test_rmse'] for r in results]
        test_r2 = [r['regression']['test_r2'] for r in results]
        
        # Classification metrics
        train_acc = [r['classification']['train_accuracy'] for r in results]
        test_acc = [r['classification']['test_accuracy'] for r in results]
        
        # Plot
        axes[0, 0].plot(layers, train_rmse, 'o-', color=color, alpha=0.7, label=exp_name)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Train RMSE')
        axes[0, 0].set_title('Regression: Train RMSE by Layer')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(layers, test_rmse, 'o-', color=color, alpha=0.7, label=exp_name)
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Test RMSE')
        axes[0, 1].set_title('Regression: Test RMSE by Layer')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(layers, test_r2, 'o-', color=color, alpha=0.7, label=exp_name)
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Test R²')
        axes[1, 0].set_title('Regression: Test R² by Layer')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(layers, test_acc, 'o-', color=color, alpha=0.7, label=exp_name)
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Test Accuracy')
        axes[1, 1].set_title('Classification: Test Accuracy by Layer')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def print_summary(experiments):
    """Print summary statistics for each experiment."""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*80)
    
    for exp_name, config, results in experiments:
        print(f"\n{exp_name}:")
        print(f"  Config:")
        print(f"    - Max files per temp: {config.get('max_files_per_temp', 'all')}")
        print(f"    - Token sampling: every {config['token_every_n']}")
        print(f"    - Layers: {len(results)}")
        
        # Find best layers
        test_rmse = [r['regression']['test_rmse'] for r in results]
        test_r2 = [r['regression']['test_r2'] for r in results]
        test_acc = [r['classification']['test_accuracy'] for r in results]
        
        best_rmse_idx = np.argmin(test_rmse)
        best_r2_idx = np.argmax(test_r2)
        best_acc_idx = np.argmax(test_acc)
        
        print(f"  Best Regression Layer (RMSE): {results[best_rmse_idx]['layer']} (RMSE={test_rmse[best_rmse_idx]:.4f}, R²={test_r2[best_rmse_idx]:.4f})")
        print(f"  Best Regression Layer (R²): {results[best_r2_idx]['layer']} (RMSE={test_rmse[best_r2_idx]:.4f}, R²={test_r2[best_r2_idx]:.4f})")
        print(f"  Best Classification Layer: {results[best_acc_idx]['layer']} (Acc={test_acc[best_acc_idx]:.4f})")
        print(f"  Averages:")
        print(f"    - Test RMSE: {np.mean(test_rmse):.4f} ± {np.std(test_rmse):.4f}")
        print(f"    - Test R²: {np.mean(test_r2):.4f} ± {np.std(test_r2):.4f}")
        print(f"    - Test Acc: {np.mean(test_acc):.4f} ± {np.std(test_acc):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Compare probe training experiments')
    parser.add_argument('experiment_dirs', nargs='+',
                        help='Paths to experiment directories to compare')
    parser.add_argument('--output', type=str, default='experiments_comparison.png',
                        help='Output plot path (default: experiments_comparison.png)')
    
    args = parser.parse_args()
    
    # Load all experiments
    experiments = []
    for exp_dir in args.experiment_dirs:
        exp_path = Path(exp_dir)
        if not exp_path.exists():
            print(f"Warning: {exp_dir} does not exist, skipping")
            continue
        
        try:
            config, results = load_experiment_results(exp_dir)
            exp_name = exp_path.name
            experiments.append((exp_name, config, results))
            print(f"Loaded: {exp_name} ({len(results)} layers)")
        except Exception as e:
            print(f"Error loading {exp_dir}: {e}")
    
    if not experiments:
        print("No valid experiments found!")
        return
    
    # Print summary
    print_summary(experiments)
    
    # Plot comparison
    plot_comparison(experiments, args.output)
    
    print(f"\n{'='*80}")
    print("Comparison complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

