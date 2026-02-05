#!/usr/bin/env python3
"""
Aggregate Truncated RMSE Test Results (Test B) - Pooled Analysis
==================================================================

Matches the format of aggregate_statistical_results.py for direct comparison.

Key outputs:
  - Plot A: Scatter of RMSE vs (N - N_max), log y-axis
  - Plot A2: Same as A but with linear y-axis
  - Plot B: Box plots of RMSE for below vs above N_max, log y-axis
  - Plot B2: Same as B but with linear y-axis
  - Plot C: Binned median RMSE with IQR error bars, log y-axis
  - Plot C2: Same as C but with linear y-axis
  - Plot D: RMSE vs N overlay (light lines per seed + median), log y-axis
  - Plot D2: Same as D but with linear y-axis
  - Plot E: n* histogram
  - Summary statistics: median ratio, Mann-Whitney U test

Usage:
    python aggregate_rmse_truncated_results.py --results-dir rmse_truncated_results_general_random_intensity_random_rho
    python aggregate_rmse_truncated_results.py --all-cases
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_CASES = ['same_radius', 'same_angle', 'general', 'general_random_intensity']

EXPECTED_FORMULAS = {
    'same_radius': 'N_max = n*',
    'same_angle': 'N_max = (1/2)n*',
    'general': 'N_max = (2/3)n*',
    'general_random_intensity': 'N_max = (2/3)n*',
}

RMSE_TYPES = ['position', 'intensity', 'total']


# =============================================================================
# DATA LOADING
# =============================================================================

def find_result_folders(base_dir: str) -> List[str]:
    """Find all result folders containing results.json"""
    if not os.path.exists(base_dir):
        return []
    folders = []
    for name in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, name)
        if os.path.isdir(folder_path):
            if os.path.exists(os.path.join(folder_path, 'results.json')):
                folders.append(folder_path)
    return folders


def load_results(base_dir: str) -> List[dict]:
    """Load all results from a directory."""
    folders = find_result_folders(base_dir)
    results = []
    for folder in folders:
        result_file = os.path.join(folder, 'results.json')
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            if 'error' not in result:
                results.append(result)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Failed to load {result_file}: {e}")
    return results


# =============================================================================
# POOLED DATA CONSTRUCTION
# =============================================================================

def build_pooled_table(results: List[dict], test_case: str = None) -> Dict[str, np.ndarray]:
    """
    Pool all (seed, N) data points into arrays.
    
    Returns dict of numpy arrays matching the format used by
    aggregate_statistical_results.py.
    """
    if test_case is None:
        test_case = results[0].get('test_case', 'unknown')
    
    rows = {
        'seed': [],
        'N': [],
        'N_max_predicted': [],
        'N_minus_Nmax': [],
        'N_minus_Nmax_rounded': [],
        'rho_min': [],
        'n_star_max': [],
        'rmse_position': [],
        'rmse_intensity': [],
        'rmse_total': [],
        'residual': [],
        'below_bound': [],  # 1 if N <= N_max, 0 otherwise
    }
    
    for result in results:
        seed = result.get('seed', 0)
        rho_min = result.get('rho_min', 0)
        n_star = result.get('n_star', 0)
        N_max = result.get('N_max', 0)
        
        for nr in result.get('results_by_N', []):
            N = nr['N']
            diff = N - N_max
            
            rows['seed'].append(seed)
            rows['N'].append(N)
            rows['N_max_predicted'].append(N_max)
            rows['N_minus_Nmax'].append(diff)
            rows['N_minus_Nmax_rounded'].append(int(np.floor(diff)))
            rows['rho_min'].append(rho_min)
            rows['n_star_max'].append(n_star)
            rows['rmse_position'].append(float(nr.get('rmse_position', np.nan)))
            rows['rmse_intensity'].append(float(nr.get('rmse_intensity', np.nan)))
            rows['rmse_total'].append(float(nr.get('rmse_total', np.nan)))
            rows['residual'].append(float(nr.get('residual', np.nan)))
            rows['below_bound'].append(1 if N <= N_max else 0)
    
    return {k: np.array(v) for k, v in rows.items()}


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_pooled_statistics(table: Dict[str, np.ndarray]) -> Dict:
    """
    Compute summary statistics from pooled table.
    For each RMSE type: median below/above, ratio, Mann-Whitney U test.
    """
    below = table['below_bound'] == 1
    above = table['below_bound'] == 0
    
    stats = {
        'n_total': len(table['seed']),
        'n_seeds': int(len(np.unique(table['seed']))),
        'n_below': int(np.sum(below)),
        'n_above': int(np.sum(above)),
    }
    
    for rmse_type in RMSE_TYPES:
        key = f'rmse_{rmse_type}'
        values = table[key]
        
        valid = ~np.isnan(values)
        below_vals = values[below & valid]
        above_vals = values[above & valid]
        
        if len(below_vals) == 0 or len(above_vals) == 0:
            stats[f'{rmse_type}_median_below'] = np.nan
            stats[f'{rmse_type}_median_above'] = np.nan
            stats[f'{rmse_type}_ratio'] = np.nan
            stats[f'{rmse_type}_mannwhitney_U'] = np.nan
            stats[f'{rmse_type}_mannwhitney_p'] = np.nan
            continue
        
        med_below = float(np.median(below_vals))
        med_above = float(np.median(above_vals))
        ratio = med_above / med_below if med_below > 0 else float('inf')
        
        stats[f'{rmse_type}_median_below'] = med_below
        stats[f'{rmse_type}_median_above'] = med_above
        stats[f'{rmse_type}_ratio'] = ratio
        stats[f'{rmse_type}_mean_below'] = float(np.mean(below_vals))
        stats[f'{rmse_type}_mean_above'] = float(np.mean(above_vals))
        
        if HAS_SCIPY and len(below_vals) >= 3 and len(above_vals) >= 3:
            try:
                U, p = mannwhitneyu(above_vals, below_vals, alternative='greater')
                stats[f'{rmse_type}_mannwhitney_U'] = float(U)
                stats[f'{rmse_type}_mannwhitney_p'] = float(p)
            except Exception:
                stats[f'{rmse_type}_mannwhitney_U'] = np.nan
                stats[f'{rmse_type}_mannwhitney_p'] = np.nan
        else:
            stats[f'{rmse_type}_mannwhitney_U'] = np.nan
            stats[f'{rmse_type}_mannwhitney_p'] = np.nan
    
    return stats


def compute_binned_statistics(table: Dict[str, np.ndarray]) -> Dict:
    """Compute binned RMSE statistics at each integer value of (N - N_max)."""
    bins = np.unique(table['N_minus_Nmax_rounded'])
    
    binned = {}
    for rmse_type in RMSE_TYPES:
        key = f'rmse_{rmse_type}'
        values = table[key]
        
        bin_stats = {
            'bins': [],
            'counts': [],
            'medians': [],
            'q25': [],
            'q75': [],
            'means': [],
            'stds': [],
        }
        
        for b in sorted(bins):
            mask = (table['N_minus_Nmax_rounded'] == b) & ~np.isnan(values)
            vals = values[mask]
            
            if len(vals) > 0:
                bin_stats['bins'].append(int(b))
                bin_stats['counts'].append(int(len(vals)))
                bin_stats['medians'].append(float(np.median(vals)))
                bin_stats['q25'].append(float(np.percentile(vals, 25)))
                bin_stats['q75'].append(float(np.percentile(vals, 75)))
                bin_stats['means'].append(float(np.mean(vals)))
                bin_stats['stds'].append(float(np.std(vals)))
        
        binned[rmse_type] = bin_stats
    
    return binned


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots(table: Dict[str, np.ndarray], stats: Dict,
                   binned: Dict, plot_dir: str, test_case: str):
    """Generate all plots for a single test case."""
    if not HAS_MATPLOTLIB:
        print("  matplotlib not available, skipping plots")
        return
    
    os.makedirs(plot_dir, exist_ok=True)
    
    formula_str = EXPECTED_FORMULAS.get(test_case, 'unknown')
    below = table['below_bound'] == 1
    above = table['below_bound'] == 0
    
    # =========================================================================
    # Plot A: Scatter of RMSE vs (N - N_max) — LOG y-axis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        key = f'rmse_{rmse_type}'
        x = table['N_minus_Nmax']
        y = table[key]
        valid = ~np.isnan(y) & (y > 0)
        
        if np.sum(valid) == 0:
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'RMSE {rmse_type.capitalize()}', fontsize=13)
            continue
        
        ax.scatter(x[valid], y[valid], alpha=0.3, s=15, c='steelblue', edgecolors='none')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='N = N_max')
        ax.set_yscale('log')
        ax.set_xlabel('N − N_max', fontsize=12)
        ax.set_ylabel(f'RMSE ({rmse_type})', fontsize=12)
        ax.set_title(f'RMSE {rmse_type.capitalize()}', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        med_b = stats.get(f'{rmse_type}_median_below', np.nan)
        med_a = stats.get(f'{rmse_type}_median_above', np.nan)
        if not np.isnan(med_b):
            ax.axhline(y=med_b, color='green', linestyle=':', alpha=0.7,
                       label=f'Median below: {med_b:.4f}')
        if not np.isnan(med_a):
            ax.axhline(y=med_a, color='orange', linestyle=':', alpha=0.7,
                       label=f'Median above: {med_a:.4f}')
        ax.legend(fontsize=8)
    
    fig.suptitle(f'{test_case} [TRUNCATED]: RMSE vs (N − N_max)  |  Formula: {formula_str}\n'
                 f'{stats["n_seeds"]} seeds, {stats["n_total"]} data points',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_A_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot A2: Same scatter but LINEAR y-axis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        key = f'rmse_{rmse_type}'
        x = table['N_minus_Nmax']
        y = table[key]
        valid = ~np.isnan(y) & (y > 0)
        
        if np.sum(valid) == 0:
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'RMSE {rmse_type.capitalize()}', fontsize=13)
            continue
        
        ax.scatter(x[valid], y[valid], alpha=0.3, s=15, c='steelblue', edgecolors='none')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='N = N_max')
        ax.set_xlabel('N − N_max', fontsize=12)
        ax.set_ylabel(f'RMSE ({rmse_type})', fontsize=12)
        ax.set_title(f'RMSE {rmse_type.capitalize()}', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        med_b = stats.get(f'{rmse_type}_median_below', np.nan)
        med_a = stats.get(f'{rmse_type}_median_above', np.nan)
        if not np.isnan(med_b):
            ax.axhline(y=med_b, color='green', linestyle=':', alpha=0.7,
                       label=f'Median below: {med_b:.4f}')
        if not np.isnan(med_a):
            ax.axhline(y=med_a, color='orange', linestyle=':', alpha=0.7,
                       label=f'Median above: {med_a:.4f}')
        ax.legend(fontsize=8)
    
    fig.suptitle(f'{test_case} [TRUNCATED]: RMSE vs (N − N_max) [linear]  |  Formula: {formula_str}\n'
                 f'{stats["n_seeds"]} seeds, {stats["n_total"]} data points',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_A2_scatter_linear.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot B: Box plots — below vs above (LOG y-axis)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        key = f'rmse_{rmse_type}'
        valid = ~np.isnan(table[key]) & (table[key] > 0)
        below_vals = table[key][below & valid]
        above_vals = table[key][above & valid]
        
        if len(below_vals) == 0 or len(above_vals) == 0:
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
            continue
        
        bp = ax.boxplot([below_vals, above_vals],
                       labels=[f'N ≤ N_max\n(n={len(below_vals)})',
                               f'N > N_max\n(n={len(above_vals)})'],
                       patch_artist=True,
                       showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
        
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightsalmon')
        
        ax.set_yscale('log')
        ax.set_ylabel(f'RMSE ({rmse_type})', fontsize=11)
        ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        ratio = stats.get(f'{rmse_type}_ratio', np.nan)
        p_val = stats.get(f'{rmse_type}_mannwhitney_p', np.nan)
        if not np.isnan(ratio):
            p_str = f'{p_val:.2e}' if not np.isnan(p_val) else 'N/A'
            ax.text(0.5, 0.95, f'Ratio: {ratio:.1f}×\np = {p_str}',
                   transform=ax.transAxes, ha='center', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(f'{test_case} [TRUNCATED]: RMSE Below vs Above Bound  |  {formula_str}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_B_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot B2: Box plots — LINEAR y-axis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        key = f'rmse_{rmse_type}'
        valid = ~np.isnan(table[key]) & (table[key] > 0)
        below_vals = table[key][below & valid]
        above_vals = table[key][above & valid]
        
        if len(below_vals) == 0 or len(above_vals) == 0:
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
            continue
        
        bp = ax.boxplot([below_vals, above_vals],
                       labels=[f'N ≤ N_max\n(n={len(below_vals)})',
                               f'N > N_max\n(n={len(above_vals)})'],
                       patch_artist=True,
                       showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
        
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightsalmon')
        
        ax.set_ylabel(f'RMSE ({rmse_type})', fontsize=11)
        ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        ratio = stats.get(f'{rmse_type}_ratio', np.nan)
        p_val = stats.get(f'{rmse_type}_mannwhitney_p', np.nan)
        if not np.isnan(ratio):
            p_str = f'{p_val:.2e}' if not np.isnan(p_val) else 'N/A'
            ax.text(0.5, 0.95, f'Ratio: {ratio:.1f}×\np = {p_str}',
                   transform=ax.transAxes, ha='center', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(f'{test_case} [TRUNCATED]: RMSE Below vs Above [linear]  |  {formula_str}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_B2_boxplots_linear.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot C: Binned median with IQR error bars (LOG y-axis)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        bstats = binned.get(rmse_type, {})
        if not bstats.get('bins'):
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
            continue
        
        bins_arr = np.array(bstats['bins'])
        medians = np.array(bstats['medians'])
        q25 = np.array(bstats['q25'])
        q75 = np.array(bstats['q75'])
        
        yerr_low = medians - q25
        yerr_high = q75 - medians
        yerr_low = np.maximum(yerr_low, 1e-10)
        yerr_high = np.maximum(yerr_high, 1e-10)
        
        ax.errorbar(bins_arr, medians, yerr=[yerr_low, yerr_high],
                   fmt='o-', color='steelblue', capsize=4, markersize=6,
                   linewidth=1.5, elinewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='N = N_max')
        ax.set_yscale('log')
        ax.set_xlabel('N − N_max (binned)', fontsize=12)
        ax.set_ylabel(f'Median RMSE ({rmse_type})', fontsize=11)
        ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{test_case} [TRUNCATED]: Binned Median RMSE  |  {formula_str}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_C_binned.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot C2: Binned median — LINEAR y-axis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        bstats = binned.get(rmse_type, {})
        if not bstats.get('bins'):
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
            continue
        
        bins_arr = np.array(bstats['bins'])
        medians = np.array(bstats['medians'])
        q25 = np.array(bstats['q25'])
        q75 = np.array(bstats['q75'])
        
        yerr_low = medians - q25
        yerr_high = q75 - medians
        
        ax.errorbar(bins_arr, medians, yerr=[yerr_low, yerr_high],
                   fmt='o-', color='steelblue', capsize=4, markersize=6,
                   linewidth=1.5, elinewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='N = N_max')
        ax.set_xlabel('N − N_max (binned)', fontsize=12)
        ax.set_ylabel(f'Median RMSE ({rmse_type})', fontsize=11)
        ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{test_case} [TRUNCATED]: Binned Median RMSE (linear)  |  {formula_str}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_C2_binned_linear.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot D: RMSE vs N overlay (light lines per seed + median) — LOG y-axis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    unique_seeds = np.unique(table['seed'])
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        key = f'rmse_{rmse_type}'
        
        has_data = np.sum(~np.isnan(table[key]) & (table[key] > 0)) > 0
        if not has_data:
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
            continue
        
        for s in unique_seeds:
            mask = table['seed'] == s
            N_vals = table['N'][mask]
            rmse_vals = table[key][mask]
            valid = ~np.isnan(rmse_vals)
            if np.sum(valid) > 1:
                order = np.argsort(N_vals[valid])
                ax.plot(N_vals[valid][order], rmse_vals[valid][order],
                       '-', alpha=0.15, color='steelblue', linewidth=0.8)
        
        unique_N = np.unique(table['N'])
        median_rmse = []
        for N_val in unique_N:
            mask = (table['N'] == N_val) & ~np.isnan(table[key])
            if np.sum(mask) > 0:
                median_rmse.append(float(np.median(table[key][mask])))
            else:
                median_rmse.append(np.nan)
        
        ax.plot(unique_N, median_rmse, 'o-', color='darkred', linewidth=2,
               markersize=5, label='Median', zorder=10)
        
        ax.set_yscale('log')
        ax.set_xlabel('N (number of sources)', fontsize=12)
        ax.set_ylabel(f'RMSE ({rmse_type})', fontsize=11)
        ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{test_case} [TRUNCATED]: RMSE vs N (all seeds)  |  {formula_str}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_D_rmse_vs_N.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot D2: RMSE vs N overlay — LINEAR y-axis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        key = f'rmse_{rmse_type}'
        
        has_data = np.sum(~np.isnan(table[key]) & (table[key] > 0)) > 0
        if not has_data:
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
            continue
        
        for s in unique_seeds:
            mask = table['seed'] == s
            N_vals = table['N'][mask]
            rmse_vals = table[key][mask]
            valid = ~np.isnan(rmse_vals)
            if np.sum(valid) > 1:
                order = np.argsort(N_vals[valid])
                ax.plot(N_vals[valid][order], rmse_vals[valid][order],
                       '-', alpha=0.15, color='steelblue', linewidth=0.8)
        
        unique_N = np.unique(table['N'])
        median_rmse = []
        for N_val in unique_N:
            mask = (table['N'] == N_val) & ~np.isnan(table[key])
            if np.sum(mask) > 0:
                median_rmse.append(float(np.median(table[key][mask])))
            else:
                median_rmse.append(np.nan)
        
        ax.plot(unique_N, median_rmse, 'o-', color='darkred', linewidth=2,
               markersize=5, label='Median', zorder=10)
        
        ax.set_xlabel('N (number of sources)', fontsize=12)
        ax.set_ylabel(f'RMSE ({rmse_type})', fontsize=11)
        ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{test_case} [TRUNCATED]: RMSE vs N (linear scale)  |  {formula_str}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_D2_rmse_vs_N_linear.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot E: n* histogram
    # =========================================================================
    _, seed_idx = np.unique(table['seed'], return_index=True)
    n_star_vals = table['n_star_max'][seed_idx]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(n_star_vals, bins=range(int(n_star_vals.min()), int(n_star_vals.max()) + 2),
            align='left', color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('n* (max usable Fourier mode)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{test_case} [TRUNCATED]: Distribution of n*\n'
                 f'Mean: {np.mean(n_star_vals):.1f}, Std: {np.std(n_star_vals):.1f}',
                 fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_E_nstar_hist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved 9 plots to {plot_dir}/")


# =============================================================================
# SUMMARY PRINTING
# =============================================================================

def print_summary(stats: Dict, binned: Dict, test_case: str):
    """Print summary statistics."""
    formula_str = EXPECTED_FORMULAS.get(test_case, 'unknown')
    
    print(f"\n{'='*75}")
    print(f"TRUNCATED RMSE TEST SUMMARY: {test_case}")
    print(f"Formula: {formula_str}")
    print(f"{'='*75}")
    
    print(f"\n  Total data points: {stats['n_total']}")
    print(f"  Number of seeds: {stats['n_seeds']}")
    print(f"  Points below N_max: {stats['n_below']}")
    print(f"  Points above N_max: {stats['n_above']}")
    
    print(f"\n  {'Metric':<12s} {'Med Below':>12s} {'Med Above':>12s} {'Ratio':>10s} {'p-value':>12s}")
    print(f"  {'-'*58}")
    
    for rmse_type in RMSE_TYPES:
        med_b = stats.get(f'{rmse_type}_median_below', np.nan)
        med_a = stats.get(f'{rmse_type}_median_above', np.nan)
        ratio = stats.get(f'{rmse_type}_ratio', np.nan)
        p_val = stats.get(f'{rmse_type}_mannwhitney_p', np.nan)
        
        med_b_str = f'{med_b:.4f}' if not np.isnan(med_b) else 'N/A'
        med_a_str = f'{med_a:.4f}' if not np.isnan(med_a) else 'N/A'
        ratio_str = f'{ratio:.2f}×' if not np.isnan(ratio) else 'N/A'
        p_str = f'{p_val:.2e}' if not np.isnan(p_val) else 'N/A'
        
        print(f"  {rmse_type:<12s} {med_b_str:>12s} {med_a_str:>12s} {ratio_str:>10s} {p_str:>12s}")
    
    # Binned breakdown for position RMSE
    print(f"\n  Binned breakdown (position RMSE):")
    print(f"  {'Offset':>8s} {'Count':>8s} {'Median':>10s} {'Q25':>10s} {'Q75':>10s}")
    print(f"  {'-'*50}")
    
    bstats = binned.get('position', {})
    if bstats.get('bins'):
        for i, b in enumerate(bstats['bins']):
            marker = " <<<" if abs(b) <= 1 else ""
            print(f"  {b:8d} {bstats['counts'][i]:8d} {bstats['medians'][i]:10.4f} "
                  f"{bstats['q25'][i]:10.4f} {bstats['q75'][i]:10.4f}{marker}")
    
    print(f"{'='*75}\n")


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_one_case(results_dir: str, test_case: str, output_base: str):
    """Process results for one test case."""
    print(f"\nProcessing: {test_case}")
    print(f"  Results dir: {results_dir}")
    
    results = load_results(results_dir)
    if not results:
        print(f"  No results found in {results_dir}")
        return None, None, None
    
    print(f"  Loaded {len(results)} seed results")
    
    # Build pooled table
    table = build_pooled_table(results, test_case)
    print(f"  Pooled {len(table['seed'])} data points")
    
    # Compute statistics
    stats = compute_pooled_statistics(table)
    binned = compute_binned_statistics(table)
    
    # Print summary
    print_summary(stats, binned, test_case)
    
    # Generate plots
    plot_dir = os.path.join(output_base, f'plots_truncated_{test_case}')
    generate_plots(table, stats, binned, plot_dir, test_case)
    
    # Save summary JSON
    summary = {
        'test_case': test_case,
        'test_type': 'truncated_rmse',
        'formula': EXPECTED_FORMULAS.get(test_case, 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'binned': binned,
    }
    
    summary_path = os.path.join(output_base, f'rmse_truncated_summary_{test_case}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")
    
    return table, stats, binned


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate truncated RMSE test (Test B) results"
    )
    parser.add_argument('--results-dir', type=str,
                        help='Results directory for one test case')
    parser.add_argument('--all-cases', action='store_true',
                        help='Process all available test cases')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for plots and summaries')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    
    args = parser.parse_args()
    
    if args.no_plots:
        global HAS_MATPLOTLIB
        HAS_MATPLOTLIB = False
    
    if args.results_dir:
        # Infer test case from directory name
        dirname = os.path.basename(args.results_dir.rstrip('/'))
        test_case = 'unknown'
        for tc in TEST_CASES:
            if tc in dirname:
                test_case = tc
                break
        process_one_case(args.results_dir, test_case, args.output_dir)
    
    elif args.all_cases:
        for tc in TEST_CASES:
            for suffix in ['_random_rho', '']:
                results_dir = f"rmse_truncated_results_{tc}{suffix}"
                if os.path.exists(results_dir):
                    process_one_case(results_dir, tc, args.output_dir)
    else:
        print("Specify --results-dir or --all-cases")
        parser.print_help()


if __name__ == '__main__':
    main()
