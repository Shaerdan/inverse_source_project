#!/usr/bin/env python3
"""
Aggregate Statistical Validation Results
=========================================

Loads results from all seeds, computes summary statistics, generates plots.

Implements three analyses from spec:
1. Validate actual-noise bound (PRIMARY)
2. Statistical distribution of n*_actual (68/32 split)
3. Predictive power comparison

Usage:
    python aggregate_statistical_results.py --results-dir stat_results/
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any


# =============================================================================
# DATA LOADING
# =============================================================================

def find_result_folders(base_dir: str) -> List[str]:
    """Find all result folders containing results.json"""
    folders = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and 'results.json' in os.listdir(path):
            folders.append(path)
    return sorted(folders)


def load_all_results(base_dir: str) -> List[dict]:
    """Load results.json from all seed folders."""
    folders = find_result_folders(base_dir)
    print(f"Found {len(folders)} result folders")
    
    results = []
    for folder in folders:
        results_path = os.path.join(folder, 'results.json')
        config_path = os.path.join(folder, 'config.json')
        
        with open(results_path, 'r') as f:
            result = json.load(f)
        
        # Also load config for reference values
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                result['n_star_predicted'] = config.get('n_star_predicted')
                result['N_max_predicted'] = config.get('N_max_predicted')
        
        results.append(result)
    
    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_summary_statistics(results: List[dict]) -> dict:
    """
    Compute all summary statistics per spec.
    
    Analysis 1: Validate actual-noise bound
    Analysis 2: Distribution of n*_actual
    Analysis 3: Predictive power comparison
    """
    n_seeds = len(results)
    
    # Extract arrays
    n_star_actual_arr = np.array([r['n_star_actual'] for r in results])
    N_max_actual_arr = np.array([r['N_max_actual'] for r in results])
    N_transition_arr = np.array([r['N_transition'] for r in results])
    
    # Reference values (should be same for all, take from first)
    n_star_predicted = results[0].get('n_star_predicted', 25.8)
    N_max_predicted = results[0].get('N_max_predicted', 17.2)
    
    # =========================================================================
    # Analysis 1: Validate actual-noise bound
    # =========================================================================
    
    # Correlation between N_max_actual and N_transition
    corr_actual = np.corrcoef(N_max_actual_arr, N_transition_arr)[0, 1]
    
    # Error: N_transition - N_max_actual
    error_actual = N_transition_arr - N_max_actual_arr
    mean_error_actual = np.mean(error_actual)
    std_error_actual = np.std(error_actual)
    
    # =========================================================================
    # Analysis 2: Distribution of n*_actual
    # =========================================================================
    
    n_star_actual_mean = np.mean(n_star_actual_arr)
    n_star_actual_std = np.std(n_star_actual_arr)
    
    # 68/32 split verification
    fraction_above = np.mean(n_star_actual_arr >= n_star_predicted)
    fraction_below = np.mean(n_star_actual_arr < n_star_predicted)
    
    # =========================================================================
    # Analysis 3: Predictive power comparison
    # =========================================================================
    
    # N_max_predicted is constant, so correlation is technically undefined
    # But we can still compute error statistics
    error_predicted = N_transition_arr - N_max_predicted
    mean_error_predicted = np.mean(error_predicted)
    std_error_predicted = np.std(error_predicted)
    
    # For correlation, use variance of N_transition
    # If N_max_predicted is constant, correlation = 0
    corr_predicted = 0.0  # By definition (constant predictor)
    
    # Also compute N_transition statistics
    N_transition_mean = np.mean(N_transition_arr)
    N_transition_std = np.std(N_transition_arr)
    
    summary = {
        # Analysis 1: Actual-noise bound
        'correlation_Nmax_actual_vs_Ntransition': float(corr_actual),
        'mean_error_actual_bound': float(mean_error_actual),
        'std_error_actual_bound': float(std_error_actual),
        
        # Analysis 2: n*_actual distribution
        'n_star_predicted': float(n_star_predicted),
        'n_star_actual_mean': float(n_star_actual_mean),
        'n_star_actual_std': float(n_star_actual_std),
        'fraction_n_star_above_predicted': float(fraction_above),
        'fraction_n_star_below_predicted': float(fraction_below),
        
        # Analysis 3: Predictive power comparison
        'N_max_predicted': float(N_max_predicted),
        'correlation_Nmax_predicted_vs_Ntransition': float(corr_predicted),
        'mean_error_predicted_bound': float(mean_error_predicted),
        'std_error_predicted_bound': float(std_error_predicted),
        
        # Additional statistics
        'N_transition_mean': float(N_transition_mean),
        'N_transition_std': float(N_transition_std),
        'n_seeds': n_seeds,
    }
    
    return summary


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots(results: List[dict], summary: dict, output_dir: str):
    """Generate all 4 plots per spec."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract arrays
    n_star_actual_arr = np.array([r['n_star_actual'] for r in results])
    N_max_actual_arr = np.array([r['N_max_actual'] for r in results])
    N_transition_arr = np.array([r['N_transition'] for r in results])
    n_star_predicted = summary['n_star_predicted']
    N_max_predicted = summary['N_max_predicted']
    
    # =========================================================================
    # Plot 1: N_transition vs N_max_actual (KEY PLOT)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(N_max_actual_arr, N_transition_arr, alpha=0.6, s=50, c='blue', edgecolors='black')
    
    # y=x line
    min_val = min(N_max_actual_arr.min(), N_transition_arr.min()) - 1
    max_val = max(N_max_actual_arr.max(), N_transition_arr.max()) + 1
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x')
    
    ax.set_xlabel('N_max_actual (from actual |η̂_n|)', fontsize=12)
    ax.set_ylabel('N_transition (observed)', fontsize=12)
    ax.set_title(f'PRIMARY VALIDATION: N_transition vs N_max_actual\n'
                 f'Correlation = {summary["correlation_Nmax_actual_vs_Ntransition"]:.3f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Ntransition_vs_Nmax_actual.png'), dpi=150)
    plt.close()
    print(f"  Saved: Ntransition_vs_Nmax_actual.png")
    
    # =========================================================================
    # Plot 2: Histogram of n*_actual
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(n_star_actual_arr, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=n_star_predicted, color='red', linestyle='--', linewidth=2, 
               label=f'n*_predicted = {n_star_predicted:.1f}')
    
    # Annotate 68/32 split
    frac_above = summary['fraction_n_star_above_predicted']
    frac_below = summary['fraction_n_star_below_predicted']
    ax.text(0.95, 0.95, f'Above: {frac_above:.1%}\nBelow: {frac_below:.1%}\n(Expected: 68%/32%)',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('n*_actual', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of n*_actual\n'
                 f'Mean = {summary["n_star_actual_mean"]:.2f}, Std = {summary["n_star_actual_std"]:.2f}', 
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'n_star_histogram.png'), dpi=150)
    plt.close()
    print(f"  Saved: n_star_histogram.png")
    
    # =========================================================================
    # Plot 3: RMSE vs N, all seeds overlaid (Position and Intensity)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get N values from first result
    first_rmse = results[0].get('rmse_position', results[0].get('rmse', {}))
    N_values = sorted([int(k) for k in first_rmse.keys()])
    
    # --- Left: Position RMSE ---
    ax = axes[0]
    for result in results:
        rmse_dict = result.get('rmse_position', result.get('rmse', {}))
        N_vals = sorted([int(k) for k in rmse_dict.keys()])
        rmse_vals = [rmse_dict[str(N)] if str(N) in rmse_dict else rmse_dict.get(N, 0) for N in N_vals]
        ax.plot(N_vals, rmse_vals, 'b-', alpha=0.15, linewidth=1)
    
    # Compute and plot median
    rmse_by_N = defaultdict(list)
    for result in results:
        rmse_dict = result.get('rmse_position', result.get('rmse', {}))
        for N_str, rmse in rmse_dict.items():
            rmse_by_N[int(N_str)].append(rmse)
    
    N_sorted = sorted(rmse_by_N.keys())
    median_rmse = [np.median(rmse_by_N[N]) for N in N_sorted]
    ax.plot(N_sorted, median_rmse, 'r-', linewidth=3, label='Median', zorder=10)
    
    ax.axvline(x=N_max_predicted, color='green', linestyle='--', linewidth=2, 
               label=f'N_max_predicted = {N_max_predicted:.1f}')
    
    ax.set_xlabel('Number of Sources (N)', fontsize=12)
    ax.set_ylabel('Position RMSE', fontsize=12)
    ax.set_title(f'Position RMSE vs N: All {len(results)} Seeds', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # --- Right: Intensity RMSE ---
    ax = axes[1]
    has_intensity = 'rmse_intensity' in results[0]
    
    if has_intensity:
        for result in results:
            rmse_dict = result.get('rmse_intensity', {})
            N_vals = sorted([int(k) for k in rmse_dict.keys()])
            rmse_vals = [rmse_dict[str(N)] if str(N) in rmse_dict else rmse_dict.get(N, 0) for N in N_vals]
            ax.plot(N_vals, rmse_vals, 'b-', alpha=0.15, linewidth=1)
        
        # Compute and plot median
        rmse_by_N = defaultdict(list)
        for result in results:
            for N_str, rmse in result.get('rmse_intensity', {}).items():
                rmse_by_N[int(N_str)].append(rmse)
        
        N_sorted = sorted(rmse_by_N.keys())
        median_rmse = [np.median(rmse_by_N[N]) for N in N_sorted]
        ax.plot(N_sorted, median_rmse, 'r-', linewidth=3, label='Median', zorder=10)
        
        ax.axvline(x=N_max_predicted, color='green', linestyle='--', linewidth=2, 
                   label=f'N_max_predicted = {N_max_predicted:.1f}')
        
        ax.set_xlabel('Number of Sources (N)', fontsize=12)
        ax.set_ylabel('Intensity RMSE', fontsize=12)
        ax.set_title(f'Intensity RMSE vs N: All {len(results)} Seeds', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No intensity RMSE data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Intensity RMSE (not available)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_all_seeds.png'), dpi=150)
    plt.close()
    print(f"  Saved: rmse_all_seeds.png")
    
    # =========================================================================
    # Plot 4: Error histogram (N_transition - N_max_actual)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Error from actual bound
    error_actual = N_transition_arr - N_max_actual_arr
    ax1 = axes[0]
    ax1.hist(error_actual, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('N_transition - N_max_actual', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Error: Actual-Noise Bound\n'
                  f'Mean = {summary["mean_error_actual_bound"]:.2f}, '
                  f'Std = {summary["std_error_actual_bound"]:.2f}', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Right: Error from predicted bound
    error_predicted = N_transition_arr - N_max_predicted
    ax2 = axes[1]
    ax2.hist(error_predicted, bins=15, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('N_transition - N_max_predicted', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Error: σ_Four-Based Bound\n'
                  f'Mean = {summary["mean_error_predicted_bound"]:.2f}, '
                  f'Std = {summary["std_error_predicted_bound"]:.2f}', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_histogram.png'), dpi=150)
    plt.close()
    print(f"  Saved: error_histogram.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Aggregate statistical validation results")
    parser.add_argument('--results-dir', type=str, default='stat_results',
                       help='Directory containing seed result folders')
    parser.add_argument('--output', type=str, default='statistical_validation_summary.json',
                       help='Output JSON file')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # Load all results
    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded {len(results)} seed results")
    
    # Get config from first result
    first_folder = find_result_folders(args.results_dir)[0]
    config_path = os.path.join(first_folder, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Compute summary statistics
    print("\nComputing summary statistics...")
    summary = compute_summary_statistics(results)
    
    # Build full output per spec
    output_data = {
        'config': {
            'domain': config.get('domain', 'disk'),
            'rho': config.get('rho', 0.7),
            'sigma_noise': config.get('sigma_noise', 0.001),
            'n_sensors': config.get('n_sensors', 100),
            'sigma_four': config.get('sigma_four'),
            'n_star_predicted': config.get('n_star_predicted'),
            'N_max_predicted': config.get('N_max_predicted'),
            'n_seeds': len(results),
            'N_values_tested': config.get('N_values_tested', [10, 12, 14, 16, 18, 20, 22, 24]),
        },
        'results': results,
        'summary': summary,
    }
    
    # Save JSON
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {args.output}")
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_dir = 'statistical_plots'
        generate_plots(results, summary, plot_dir)
    
    # Print summary
    print(f"\n{'='*70}")
    print("STATISTICAL VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nAnalysis 1: Actual-Noise Bound Validation (PRIMARY)")
    print(f"  Correlation(N_max_actual, N_transition) = {summary['correlation_Nmax_actual_vs_Ntransition']:.3f}")
    print(f"  Mean error (N_transition - N_max_actual) = {summary['mean_error_actual_bound']:.2f}")
    print(f"  Std error = {summary['std_error_actual_bound']:.2f}")
    
    print(f"\nAnalysis 2: n*_actual Distribution")
    print(f"  n*_predicted = {summary['n_star_predicted']:.2f}")
    print(f"  n*_actual: mean = {summary['n_star_actual_mean']:.2f}, std = {summary['n_star_actual_std']:.2f}")
    print(f"  Fraction ≥ n*_predicted: {summary['fraction_n_star_above_predicted']:.1%} (expected ~68%)")
    print(f"  Fraction < n*_predicted: {summary['fraction_n_star_below_predicted']:.1%} (expected ~32%)")
    
    print(f"\nAnalysis 3: Predictive Power Comparison")
    print(f"  N_max_predicted (constant) = {summary['N_max_predicted']:.2f}")
    print(f"  Mean error (σ_Four bound) = {summary['mean_error_predicted_bound']:.2f}")
    print(f"  Std error (σ_Four bound) = {summary['std_error_predicted_bound']:.2f}")
    
    print(f"\n{'='*70}")
    
    # Interpretation
    if summary['correlation_Nmax_actual_vs_Ntransition'] > 0.8:
        print("✓ HIGH correlation: Actual-noise bound accurately predicts transition!")
    elif summary['correlation_Nmax_actual_vs_Ntransition'] > 0.5:
        print("~ MODERATE correlation: Actual-noise bound has predictive power")
    else:
        print("✗ LOW correlation: Actual-noise bound does not predict transition well")
    
    frac_above = summary['fraction_n_star_above_predicted']
    if 0.60 <= frac_above <= 0.76:
        print(f"✓ 68/32 split verified: {frac_above:.1%} above (expected ~68%)")
    else:
        print(f"~ 68/32 split deviation: {frac_above:.1%} above (expected ~68%)")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
