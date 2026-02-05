#!/usr/bin/env python3
"""
Aggregate Statistical Validation Results - Pooled Analysis
============================================================

Instead of detecting N_transition per seed (which requires arbitrary thresholds),
this pools all (seed, N) data points and asks:

    "Is RMSE below the predicted N_max systematically lower than above?"

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
    python aggregate_statistical_results.py --results-dir stat_results_same_radius_random_rho
    python aggregate_statistical_results.py --all-cases
    python aggregate_statistical_results.py --all-cases --no-plots
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple

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

FORMULA_MULTIPLIERS = {
    'same_radius': 1.0,
    'same_angle': 0.5,
    'general': 2.0/3.0,
    'general_random_intensity': 2.0/3.0,
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
            result_file = os.path.join(folder_path, 'results.json')
            if os.path.exists(result_file):
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
    
    For each result and each N tested, creates a row with:
    - seed, N, N_max_predicted, N_minus_Nmax, rmse_position, rmse_intensity, rmse_total
    
    Returns dict of arrays.
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
        'below_bound': [],  # 1 if N <= N_max, 0 otherwise
    }
    
    for result in results:
        seed = result.get('seed', 0)
        rho_min = result.get('rho_min', 0)
        n_star = result.get('n_star_max', result.get('n_star_actual', 0))
        N_max = result.get('N_max_predicted', 0)
        
        rmse_pos = result.get('rmse_position', result.get('rmse', {}))
        rmse_int = result.get('rmse_intensity', {})
        rmse_tot = result.get('rmse_total', {})
        
        N_values = result.get('N_values_tested', [])
        if not N_values:
            N_values = sorted([int(k) for k in rmse_pos.keys()])
        
        for N in N_values:
            N_str = str(N)
            N_int = int(N)
            
            pos = rmse_pos.get(N_str, rmse_pos.get(N_int))
            inten = rmse_int.get(N_str, rmse_int.get(N_int))
            tot = rmse_tot.get(N_str, rmse_tot.get(N_int))
            
            if pos is None:
                continue
            
            diff = N_int - N_max
            
            rows['seed'].append(seed)
            rows['N'].append(N_int)
            rows['N_max_predicted'].append(N_max)
            rows['N_minus_Nmax'].append(diff)
            rows['N_minus_Nmax_rounded'].append(int(np.floor(diff)))
            rows['rho_min'].append(rho_min)
            rows['n_star_max'].append(n_star)
            rows['rmse_position'].append(float(pos) if pos is not None else np.nan)
            rows['rmse_intensity'].append(float(inten) if inten is not None else np.nan)
            rows['rmse_total'].append(float(tot) if tot is not None else np.nan)
            rows['below_bound'].append(1 if N_int <= N_max else 0)
    
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
        
        bin_stats = []
        for b in bins:
            mask = (table['N_minus_Nmax_rounded'] == b) & ~np.isnan(values)
            if np.sum(mask) == 0:
                continue
            vals = values[mask]
            bin_stats.append({
                'bin': int(b),
                'n': int(np.sum(mask)),
                'median': float(np.median(vals)),
                'q25': float(np.percentile(vals, 25)),
                'q75': float(np.percentile(vals, 75)),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            })
        binned[rmse_type] = bin_stats
    
    return binned


def compute_split_sweep(table: Dict[str, np.ndarray]) -> Dict:
    """
    Sweep the split point to test WHERE the transition occurs.
    
    For each candidate split c, partition data into:
      - group_low:  N - N_max <= c   (expected easy)
      - group_high: N - N_max > c    (expected hard)
    
    Compute Mann-Whitney U statistic at each c.
    If N_max is correct, the optimal split (max U / min p) should be near c=0.
    If optimal c is far from 0, the bound is biased.
    
    Returns dict with:
      - 'split_points': array of candidate c values
      - For each rmse_type:
        - '{type}_U': U statistics
        - '{type}_p': p-values
        - '{type}_optimal_c': split point with best separation
        - '{type}_ratio_at_c': median ratio at each split
    """
    if not HAS_SCIPY:
        return {}
    
    diff = table['N_minus_Nmax']
    
    # Sweep split points: integers from min to max observed, but limit range
    c_min = max(int(np.floor(np.min(diff))), -10)
    c_max = min(int(np.ceil(np.max(diff))), 10)
    split_points = np.arange(c_min, c_max + 1, dtype=float)
    
    result = {'split_points': split_points.tolist()}
    
    for rmse_type in RMSE_TYPES:
        key = f'rmse_{rmse_type}'
        values = table[key]
        valid = ~np.isnan(values) & (values > 0)
        
        U_vals = []
        p_vals = []
        ratios = []
        
        for c in split_points:
            low_mask = (diff <= c) & valid
            high_mask = (diff > c) & valid
            
            n_low = np.sum(low_mask)
            n_high = np.sum(high_mask)
            
            if n_low < 3 or n_high < 3:
                U_vals.append(np.nan)
                p_vals.append(np.nan)
                ratios.append(np.nan)
                continue
            
            low_vals = values[low_mask]
            high_vals = values[high_mask]
            
            try:
                U, p = mannwhitneyu(high_vals, low_vals, alternative='greater')
                U_vals.append(float(U))
                p_vals.append(float(p))
            except Exception:
                U_vals.append(np.nan)
                p_vals.append(np.nan)
            
            med_low = np.median(low_vals)
            med_high = np.median(high_vals)
            ratios.append(float(med_high / med_low) if med_low > 0 else float('inf'))
        
        U_arr = np.array(U_vals)
        p_arr = np.array(p_vals)
        
        # Find optimal split: smallest p-value (most significant separation)
        valid_p = ~np.isnan(p_arr)
        if np.any(valid_p):
            best_idx = np.nanargmin(p_arr)
            optimal_c = float(split_points[best_idx])
        else:
            optimal_c = np.nan
        
        result[f'{rmse_type}_U'] = U_vals
        result[f'{rmse_type}_p'] = p_vals
        result[f'{rmse_type}_ratio_at_c'] = ratios
        result[f'{rmse_type}_optimal_c'] = optimal_c
    
    return result


def compute_narrow_band_test(table: Dict[str, np.ndarray],
                              band_below: Tuple[int, int] = (-2, 0),
                              band_above: Tuple[int, int] = (1, 3)) -> Dict:
    """
    Test the transition specifically in a narrow band around N_max.
    
    Instead of comparing ALL below vs ALL above (which is trivially true),
    compare just the borderline cases:
      - "just below": band_below[0] <= (N - N_max) <= band_below[1]
      - "just above": band_above[0] <= (N - N_max) <= band_above[1]
    
    This is sensitive to WHERE the transition happens.
    If N_max is correct, even this narrow comparison should be significant.
    If N_max is off by 2+, the narrow band test will fail.
    """
    diff = table['N_minus_Nmax']
    
    below_mask_base = (diff >= band_below[0]) & (diff <= band_below[1])
    above_mask_base = (diff >= band_above[0]) & (diff <= band_above[1])
    
    result = {
        'band_below': list(band_below),
        'band_above': list(band_above),
    }
    
    for rmse_type in RMSE_TYPES:
        key = f'rmse_{rmse_type}'
        values = table[key]
        valid = ~np.isnan(values) & (values > 0)
        
        below_vals = values[below_mask_base & valid]
        above_vals = values[above_mask_base & valid]
        
        result[f'{rmse_type}_n_below'] = int(len(below_vals))
        result[f'{rmse_type}_n_above'] = int(len(above_vals))
        
        if len(below_vals) < 3 or len(above_vals) < 3:
            result[f'{rmse_type}_median_below'] = float(np.median(below_vals)) if len(below_vals) > 0 else np.nan
            result[f'{rmse_type}_median_above'] = float(np.median(above_vals)) if len(above_vals) > 0 else np.nan
            result[f'{rmse_type}_ratio'] = np.nan
            result[f'{rmse_type}_p'] = np.nan
            continue
        
        med_b = float(np.median(below_vals))
        med_a = float(np.median(above_vals))
        
        result[f'{rmse_type}_median_below'] = med_b
        result[f'{rmse_type}_median_above'] = med_a
        result[f'{rmse_type}_ratio'] = float(med_a / med_b) if med_b > 0 else float('inf')
        
        if HAS_SCIPY:
            try:
                _, p = mannwhitneyu(above_vals, below_vals, alternative='greater')
                result[f'{rmse_type}_p'] = float(p)
            except Exception:
                result[f'{rmse_type}_p'] = np.nan
        else:
            result[f'{rmse_type}_p'] = np.nan
    
    return result


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
    # Plot A: Scatter of RMSE vs (N - N_max) — THE MONEY PLOT
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
    
    fig.suptitle(f'{test_case}: RMSE vs (N − N_max)  |  Formula: {formula_str}\n'
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
        # LINEAR scale
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
    
    fig.suptitle(f'{test_case}: RMSE vs (N − N_max) [linear]  |  Formula: {formula_str}\n'
                 f'{stats["n_seeds"]} seeds, {stats["n_total"]} data points',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_A2_scatter_linear.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot B: Box plots — below vs above
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
    
    fig.suptitle(f'{test_case}: RMSE Below vs Above Bound  |  {formula_str}',
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
        
        # LINEAR scale
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
    
    fig.suptitle(f'{test_case}: RMSE Below vs Above Bound (linear)  |  {formula_str}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_B2_boxplots_linear.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot C: Binned median with IQR error bars
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        bin_data = binned.get(rmse_type, [])
        if not bin_data:
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
            continue
        
        bins_x = [b['bin'] for b in bin_data]
        medians = [b['median'] for b in bin_data]
        q25 = [b['q25'] for b in bin_data]
        q75 = [b['q75'] for b in bin_data]
        counts = [b['n'] for b in bin_data]
        
        yerr_low = [max(m - q, 1e-10) for m, q in zip(medians, q25)]
        yerr_high = [q - m for m, q in zip(medians, q75)]
        
        colors = ['green' if b <= 0 else 'red' for b in bins_x]
        
        ax.errorbar(bins_x, medians, yerr=[yerr_low, yerr_high],
                    fmt='none', capsize=4, capthick=1.5,
                    ecolor='gray', elinewidth=1)
        
        for bx, med, col in zip(bins_x, medians, colors):
            ax.plot(bx, med, 'o', color=col, markersize=7, zorder=5)
        
        for bx, med, n in zip(bins_x, medians, counts):
            ax.annotate(f'n={n}', (bx, med), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=7, alpha=0.7)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_yscale('log')
        ax.set_xlabel('N − N_max', fontsize=12)
        ax.set_ylabel(f'Median RMSE ({rmse_type})', fontsize=11)
        ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{test_case}: Binned Median RMSE (25th–75th percentile)  |  {formula_str}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_C_binned.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot C2: Same as C but LINEAR y-axis
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, rmse_type in zip(axes, RMSE_TYPES):
        bin_data = binned.get(rmse_type, [])
        if not bin_data:
            ax.text(0.5, 0.5, f'{rmse_type}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
            continue
        
        bins_x = [b['bin'] for b in bin_data]
        medians = [b['median'] for b in bin_data]
        q25 = [b['q25'] for b in bin_data]
        q75 = [b['q75'] for b in bin_data]
        counts = [b['n'] for b in bin_data]
        
        yerr_low = [max(m - q, 0) for m, q in zip(medians, q25)]
        yerr_high = [q - m for m, q in zip(medians, q75)]
        
        colors = ['green' if b <= 0 else 'red' for b in bins_x]
        
        ax.errorbar(bins_x, medians, yerr=[yerr_low, yerr_high],
                    fmt='none', capsize=4, capthick=1.5,
                    ecolor='gray', elinewidth=1)
        
        for bx, med, col in zip(bins_x, medians, colors):
            ax.plot(bx, med, 'o', color=col, markersize=7, zorder=5)
        
        for bx, med, n in zip(bins_x, medians, counts):
            ax.annotate(f'n={n}', (bx, med), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=7, alpha=0.7)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        # LINEAR scale — no set_yscale
        ax.set_xlabel('N − N_max', fontsize=12)
        ax.set_ylabel(f'Median RMSE ({rmse_type})', fontsize=11)
        ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{test_case}: Binned Median RMSE (linear scale)  |  {formula_str}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_C2_binned_linear.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot D: RMSE vs N overlay (light lines per seed + median)
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
    
    fig.suptitle(f'{test_case}: RMSE vs N (all seeds)  |  {formula_str}',
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
        
        # LINEAR scale
        ax.set_xlabel('N (number of sources)', fontsize=12)
        ax.set_ylabel(f'RMSE ({rmse_type})', fontsize=11)
        ax.set_title(f'{rmse_type.capitalize()}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{test_case}: RMSE vs N (linear scale)  |  {formula_str}',
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
    ax.set_title(f'{test_case}: Distribution of n*\n'
                 f'Mean: {np.mean(n_star_vals):.1f}, Std: {np.std(n_star_vals):.1f}',
                 fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'plot_E_nstar_hist.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved 9 plots to {plot_dir}/")


def generate_comparison_plots(all_stats: Dict[str, Dict], plot_dir: str):
    """Generate comparison plots across test cases."""
    if not HAS_MATPLOTLIB:
        return
    
    os.makedirs(plot_dir, exist_ok=True)
    
    test_cases = sorted(all_stats.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(test_cases))
    width = 0.25
    
    for i, rmse_type in enumerate(RMSE_TYPES):
        ratios = [all_stats[tc].get(f'{rmse_type}_ratio', 0) for tc in test_cases]
        ax.bar(x + i * width, ratios, width, label=f'RMSE {rmse_type}', alpha=0.8)
        
        for j, (tc, r) in enumerate(zip(test_cases, ratios)):
            p = all_stats[tc].get(f'{rmse_type}_mannwhitney_p', np.nan)
            if not np.isnan(p):
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                ax.text(x[j] + i * width, r + 0.1, sig, ha='center', fontsize=8)
    
    ax.set_xlabel('Test Case', fontsize=12)
    ax.set_ylabel('Median Ratio (above / below)', fontsize=12)
    ax.set_title('Bound Effectiveness: Median RMSE Ratio Above/Below N_max', fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{tc}\n{EXPECTED_FORMULAS[tc]}' for tc in test_cases], fontsize=10)
    ax.legend(fontsize=10)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'comparison_ratios.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison plot to {plot_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate statistical validation results using pooled RMSE analysis")
    parser.add_argument('--results-dir', type=str, help='Single results directory')
    parser.add_argument('--results-dirs', nargs='+', type=str, help='Multiple directories')
    parser.add_argument('--all-cases', action='store_true',
                       help='Auto-detect all stat_results_* directories')
    parser.add_argument('--domain', type=str, default=None,
                       choices=['disk', 'ellipse', 'brain'],
                       help='Filter to specific domain (for multi-domain results)')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--output', type=str, default='aggregate_summary.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Find directories
    directories = {}
    
    if args.all_cases:
        for name in sorted(os.listdir('.')):
            if name.startswith('stat_results_') and os.path.isdir(name):
                # If --domain specified, only match dirs containing that domain
                if args.domain and args.domain not in name:
                    continue
                # If no --domain, skip dirs that contain a domain prefix
                # (to avoid mixing disk and ellipse results)
                if not args.domain:
                    has_domain_prefix = any(f'_{d}_' in name or name.endswith(f'_{d}')
                                           for d in ['ellipse', 'brain'])
                    # 'disk' is the legacy format without domain in name
                    # so we skip dirs with explicit non-disk domain prefixes
                    if has_domain_prefix:
                        continue
                
                for tc in TEST_CASES:
                    if tc in name:
                        directories[tc] = name
                        break
    elif args.results_dirs:
        for d in args.results_dirs:
            for tc in TEST_CASES:
                if tc in d:
                    directories[tc] = d
                    break
    elif args.results_dir:
        for tc in TEST_CASES:
            if tc in args.results_dir:
                directories[tc] = args.results_dir
                break
        if not directories:
            directories['unknown'] = args.results_dir
    
    if not directories:
        print("No result directories found!")
        print("Use --results-dir <path>, --all-cases, or --results-dirs <dir1> <dir2>")
        return
    
    print(f"{'='*70}")
    print("POOLED RMSE ANALYSIS")
    if args.domain:
        print(f"Domain: {args.domain}")
    print(f"{'='*70}")
    
    all_stats = {}
    all_outputs = {}
    
    for test_case, results_dir in sorted(directories.items()):
        print(f"\n{'='*60}")
        print(f"TEST CASE: {test_case}  |  Dir: {results_dir}")
        print(f"Formula: {EXPECTED_FORMULAS.get(test_case, 'unknown')}")
        print(f"{'='*60}")
        
        # Load
        results = load_results(results_dir)
        if not results:
            print("  No results found!")
            continue
        
        print(f"  Loaded {len(results)} seeds")
        
        # Check for missing rmse_total
        n_missing_total = sum(1 for r in results if 'rmse_total' not in r)
        if n_missing_total > 0:
            print(f"  NOTE: {n_missing_total}/{len(results)} seeds missing rmse_total (need re-run for total RMSE)")
        
        # Build pooled table
        table = build_pooled_table(results, test_case)
        
        # Compute statistics
        stats = compute_pooled_statistics(table)
        binned = compute_binned_statistics(table)
        
        stats['test_case'] = test_case
        stats['expected_formula'] = EXPECTED_FORMULAS.get(test_case, 'unknown')
        
        all_stats[test_case] = stats
        
        # Print summary
        print(f"\n  Data points: {stats['n_total']} ({stats['n_below']} below, {stats['n_above']} above)")
        
        for rmse_type in RMSE_TYPES:
            med_b = stats.get(f'{rmse_type}_median_below', np.nan)
            med_a = stats.get(f'{rmse_type}_median_above', np.nan)
            ratio = stats.get(f'{rmse_type}_ratio', np.nan)
            p_val = stats.get(f'{rmse_type}_mannwhitney_p', np.nan)
            
            if np.isnan(med_b) and np.isnan(med_a):
                print(f"\n  {rmse_type.upper()}: No data")
                continue
            
            print(f"\n  {rmse_type.upper()}:")
            print(f"    Median below N_max: {med_b:.6f}")
            print(f"    Median above N_max: {med_a:.6f}")
            print(f"    Ratio (above/below): {ratio:.2f}×")
            if not np.isnan(p_val):
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                print(f"    Mann-Whitney U p-value: {p_val:.2e} ({sig})")
        
        # Plots
        if not args.no_plots:
            domain_prefix = f'{args.domain}_' if args.domain else ''
            plot_dir = f'statistical_plots_{domain_prefix}{test_case}'
            generate_plots(table, stats, binned, plot_dir, test_case)
        
        all_outputs[test_case] = {
            'stats': stats,
            'binned': binned,
        }
    
    # Comparison plots
    if len(all_stats) > 1 and not args.no_plots:
        print(f"\n{'='*60}")
        print("COMPARISON ACROSS TEST CASES")
        print(f"{'='*60}")
        domain_prefix = f'{args.domain}_' if args.domain else ''
        generate_comparison_plots(all_stats, f'statistical_plots_{domain_prefix}comparison')
    
    # Final summary table
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Test Case':<15} {'RMSE Type':<12} {'Med Below':<12} {'Med Above':<12} {'Ratio':<8} {'p-value':<12}")
    print("-" * 70)
    
    for test_case in sorted(all_stats.keys()):
        stats = all_stats[test_case]
        for rmse_type in RMSE_TYPES:
            med_b = stats.get(f'{rmse_type}_median_below', np.nan)
            med_a = stats.get(f'{rmse_type}_median_above', np.nan)
            ratio = stats.get(f'{rmse_type}_ratio', np.nan)
            p_val = stats.get(f'{rmse_type}_mannwhitney_p', np.nan)
            
            if np.isnan(med_b):
                continue
            
            p_str = f'{p_val:.2e}' if not np.isnan(p_val) else 'N/A'
            print(f"{test_case:<15} {rmse_type:<12} {med_b:<12.6f} {med_a:<12.6f} {ratio:<8.1f} {p_str:<12}")
    
    # Save
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'method': 'pooled_rmse_analysis',
        'test_cases': list(all_outputs.keys()),
        'results': all_outputs,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
