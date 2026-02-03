#!/usr/bin/env python3
"""
Aggregate Results from Bound Theory Validation
===============================================

Combines results from multiple runs to compute:
- Mean RMSE across seeds
- Std RMSE (variance indicator)
- Success rate
- Aggregated statistics per (domain, rho, N) combination

Usage:
    python aggregate_results.py --results-dir ../results/
    python aggregate_results.py --results-dir ../results/ --output aggregate_summary.json

Output:
    - aggregate_summary.json: Full aggregated statistics
    - aggregate_summary.csv: Simplified CSV for easy viewing
    - aggregate_plots/: Summary plots
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any


def find_result_folders(base_dir: str) -> List[str]:
    """Find all result folders containing results.json"""
    folders = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and 'results.json' in os.listdir(path):
            folders.append(path)
    return sorted(folders)


def parse_folder_name(folder_name: str) -> Dict[str, Any]:
    """
    Parse folder name to extract domain, rho, seed, hash.
    
    Expected pattern: {domain}_rho{rho}_seed{seed}_{hash}
    Example: disk_rho0.70_seed42_abc12345
    """
    parts = folder_name.split('_')
    
    # parts = ['disk', 'rho0.70', 'seed42', 'abc12345']
    domain = parts[0]
    rho = float(parts[1].replace('rho', ''))
    seed = int(parts[2].replace('seed', ''))
    run_hash = parts[3] if len(parts) > 3 else 'unknown'
    
    return {'domain': domain, 'rho': rho, 'seed': seed, 'hash': run_hash}


def load_results(folder_path: str) -> List[dict]:
    """Load results.json from a folder."""
    results_path = os.path.join(folder_path, 'results.json')
    with open(results_path, 'r') as f:
        return json.load(f)


def aggregate_results(base_dir: str) -> Dict[str, Any]:
    """Aggregate all results by (domain, rho, N)."""
    
    folders = find_result_folders(base_dir)
    print(f"Found {len(folders)} result folders")
    
    # Group by (domain, rho, N)
    # Key: (domain, rho, N) -> List of results from different seeds
    grouped = defaultdict(list)
    
    for folder_path in folders:
        folder_name = os.path.basename(folder_path)
        try:
            info = parse_folder_name(folder_name)
            results = load_results(folder_path)
            
            for r in results:
                key = (info['domain'], info['rho'], r['N_tested'])
                grouped[key].append({
                    'seed': info['seed'],
                    'position_rmse': r['position_rmse'],
                    'actual_success': r['actual_success'],
                    'expected_success': r['expected_success'],
                    'N_max_theory': r['N_max_theory'],
                    'time_seconds': r['time_seconds'],
                })
        except Exception as e:
            print(f"Warning: Could not process {folder_name}: {e}")
    
    # Compute aggregate statistics
    aggregated = []
    
    for (domain, rho, N), results_list in sorted(grouped.items()):
        rmse_values = [r['position_rmse'] for r in results_list]
        success_values = [r['actual_success'] for r in results_list]
        times = [r['time_seconds'] for r in results_list]
        
        n_seeds = len(results_list)
        N_max_theory = results_list[0]['N_max_theory']
        expected_success = results_list[0]['expected_success']
        
        agg = {
            'domain': domain,
            'rho': rho,
            'N': N,
            'N_max_theory': N_max_theory,
            'expected_success': expected_success,
            'n_seeds': n_seeds,
            'rmse_mean': float(np.mean(rmse_values)),
            'rmse_std': float(np.std(rmse_values)),
            'rmse_min': float(np.min(rmse_values)),
            'rmse_max': float(np.max(rmse_values)),
            'success_rate': float(np.mean(success_values)),
            'n_success': int(np.sum(success_values)),
            'time_mean': float(np.mean(times)),
        }
        aggregated.append(agg)
    
    return {
        'generated': datetime.now().isoformat(),
        'n_folders': len(folders),
        'n_combinations': len(aggregated),
        'results': aggregated,
    }


def save_csv(data: Dict[str, Any], output_path: str):
    """Save aggregated results to CSV."""
    results = data['results']
    
    with open(output_path, 'w') as f:
        # Header
        f.write("domain,rho,N,N_max,n_seeds,rmse_mean,rmse_std,rmse_min,rmse_max,success_rate,expected\n")
        
        for r in results:
            expected = "below" if r['expected_success'] else "above"
            f.write(f"{r['domain']},{r['rho']:.2f},{r['N']},{r['N_max_theory']:.2f},"
                   f"{r['n_seeds']},{r['rmse_mean']:.6f},{r['rmse_std']:.6f},"
                   f"{r['rmse_min']:.6f},{r['rmse_max']:.6f},{r['success_rate']:.2f},{expected}\n")


def plot_aggregated_results(data: Dict[str, Any], output_dir: str):
    """Create summary plots from aggregated data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = data['results']
    
    # Group by domain
    domains = sorted(set(r['domain'] for r in results))
    rho_values = sorted(set(r['rho'] for r in results))
    
    # Plot 1: RMSE vs N for each domain (one plot per rho)
    for domain in domains:
        domain_results = [r for r in results if r['domain'] == domain]
        
        n_rho = min(6, len(rho_values))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, rho in enumerate(rho_values[:n_rho]):
            ax = axes[idx]
            rho_results = sorted([r for r in domain_results if abs(r['rho'] - rho) < 0.01],
                                key=lambda x: x['N'])
            
            if not rho_results:
                continue
            
            N_values = [r['N'] for r in rho_results]
            rmse_mean = [r['rmse_mean'] for r in rho_results]
            rmse_std = [r['rmse_std'] for r in rho_results]
            N_max = rho_results[0]['N_max_theory']
            
            # Bar plot with error bars
            colors = ['green' if r['success_rate'] > 0.5 else 'red' for r in rho_results]
            ax.bar(N_values, rmse_mean, yerr=rmse_std, color=colors, alpha=0.7, 
                   edgecolor='black', capsize=3)
            ax.axvline(x=N_max, color='blue', linestyle='--', linewidth=2, label=f'N_max={N_max:.1f}')
            ax.axhline(y=0.05, color='orange', linestyle=':', linewidth=2)
            
            ax.set_xlabel('N')
            ax.set_ylabel('RMSE')
            ax.set_title(f'ρ = {rho:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for idx in range(n_rho, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{domain.upper()}: RMSE vs N (mean ± std across seeds)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{domain}_rmse_vs_N.png'), dpi=150)
        plt.close()
    
    # Plot 2: Success rate heatmap per domain
    for domain in domains:
        domain_results = [r for r in results if r['domain'] == domain]
        
        # Get unique N and rho values
        N_values = sorted(set(r['N'] for r in domain_results))
        rho_vals = sorted(set(r['rho'] for r in domain_results))
        
        if not N_values or not rho_vals:
            continue
        
        # Create heatmap data
        success_matrix = np.full((len(rho_vals), len(N_values)), np.nan)
        
        for r in domain_results:
            i = rho_vals.index(r['rho'])
            j = N_values.index(r['N'])
            success_matrix[i, j] = r['success_rate']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(success_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        ax.set_xticks(range(len(N_values)))
        ax.set_xticklabels(N_values)
        ax.set_yticks(range(len(rho_vals)))
        ax.set_yticklabels([f'{r:.2f}' for r in rho_vals])
        
        ax.set_xlabel('N (number of sources)')
        ax.set_ylabel('ρ (conformal radius)')
        ax.set_title(f'{domain.upper()}: Success Rate Heatmap')
        
        plt.colorbar(im, label='Success Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{domain}_success_heatmap.png'), dpi=150)
        plt.close()
    
    # Plot 3: Variance (std) vs N - key indicator of ill-posedness
    for domain in domains:
        domain_results = [r for r in results if r['domain'] == domain]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for rho in rho_values[::2]:  # Every other rho for clarity
            rho_results = sorted([r for r in domain_results if abs(r['rho'] - rho) < 0.01],
                                key=lambda x: x['N'])
            if not rho_results:
                continue
            
            N_vals = [r['N'] for r in rho_results]
            std_vals = [r['rmse_std'] for r in rho_results]
            N_max = rho_results[0]['N_max_theory']
            
            ax.plot(N_vals, std_vals, 'o-', label=f'ρ={rho:.2f}, N_max={N_max:.1f}')
        
        ax.set_xlabel('N (number of sources)')
        ax.set_ylabel('RMSE Std Dev (across seeds)')
        ax.set_title(f'{domain.upper()}: RMSE Variance vs N\n(High variance indicates ill-posedness)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{domain}_variance_vs_N.png'), dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}/")


def print_summary(data: Dict[str, Any]):
    """Print summary to console."""
    results = data['results']
    
    print("\n" + "="*80)
    print("AGGREGATE SUMMARY")
    print("="*80)
    
    domains = sorted(set(r['domain'] for r in results))
    
    for domain in domains:
        domain_results = [r for r in results if r['domain'] == domain]
        
        print(f"\n--- {domain.upper()} ---")
        print(f"{'rho':>6} | {'N':>4} | {'N_max':>6} | {'RMSE mean':>10} | {'RMSE std':>10} | {'Success':>8}")
        print("-" * 70)
        
        for r in sorted(domain_results, key=lambda x: (x['rho'], x['N'])):
            print(f"{r['rho']:>6.2f} | {r['N']:>4} | {r['N_max_theory']:>6.2f} | "
                  f"{r['rmse_mean']:>10.6f} | {r['rmse_std']:>10.6f} | {r['success_rate']:>7.0%}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate bound validation results")
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing result folders')
    parser.add_argument('--output', type=str, default='aggregate_summary.json',
                       help='Output JSON file')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # Aggregate results
    data = aggregate_results(args.results_dir)
    
    if not data['results']:
        print("No results found!")
        return
    
    # Save JSON
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {args.output}")
    
    # Save CSV
    csv_path = args.output.replace('.json', '.csv')
    save_csv(data, csv_path)
    print(f"Saved: {csv_path}")
    
    # Generate plots
    if not args.no_plots:
        plot_dir = 'aggregate_plots'
        plot_aggregated_results(data, plot_dir)
    
    # Print summary
    print_summary(data)


if __name__ == "__main__":
    main()
