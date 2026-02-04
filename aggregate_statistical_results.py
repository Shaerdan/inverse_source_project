#!/usr/bin/env python3
"""
Aggregate Source Configuration Experiment Results (v2)
=======================================================

Comprehensive analysis for different source configurations:
  - same_radius: Expected N_max = n*
  - same_angle:  Expected N_max = (1/2)n*
  - general:     Expected N_max = (2/3)n*

For each test case, checks if the CORRECT formula predicts N_transition.

Usage:
    # Aggregate single test case
    python aggregate_statistical_results.py --results-dir stat_results_same_radius

    # Aggregate all test cases and compare
    python aggregate_statistical_results.py --all-cases

    # Custom directories
    python aggregate_statistical_results.py --results-dirs stat_results_same_radius stat_results_general
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_CASES = ['same_radius', 'same_angle', 'general']

EXPECTED_FORMULAS = {
    'same_radius': 'N_max = n*',
    'same_angle': 'N_max = (1/2)n*',
    'general': 'N_max = (2/3)n*',
}

FORMULA_MULTIPLIERS = {
    'same_radius': 1.0,      # N_max = 1.0 * n*
    'same_angle': 0.5,       # N_max = 0.5 * n*
    'general': 2.0/3.0,      # N_max = (2/3) * n*
}


# =============================================================================
# DATA LOADING
# =============================================================================

def find_result_folders(base_dir: str) -> List[str]:
    """Find all result folders containing results.json"""
    if not os.path.exists(base_dir):
        return []
    
    folders = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            results_file = os.path.join(path, 'results.json')
            if os.path.exists(results_file):
                folders.append(path)
    return sorted(folders)


def load_all_results(base_dir: str) -> List[dict]:
    """Load results.json from all seed folders."""
    folders = find_result_folders(base_dir)
    
    if not folders:
        print(f"  Warning: No results found in {base_dir}")
        return []
    
    print(f"  Found {len(folders)} result folders in {base_dir}")
    
    results = []
    for folder in folders:
        results_path = os.path.join(folder, 'results.json')
        config_path = os.path.join(folder, 'config.json')
        
        try:
            with open(results_path, 'r') as f:
                result = json.load(f)
            
            # Also load config for reference values
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    result['_config'] = config
            
            results.append(result)
        except Exception as e:
            print(f"    Error loading {results_path}: {e}")
    
    return results


# =============================================================================
# TRANSITION DETECTION
# =============================================================================

def detect_transition_ratio(rmse_dict: Dict, ratio_threshold: float = 2.0) -> int:
    """Detect transition using ratio method."""
    rmse_dict_int = {int(k): v for k, v in rmse_dict.items()}
    N_values = sorted(rmse_dict_int.keys())
    
    for i in range(len(N_values) - 1):
        N_curr = N_values[i]
        N_next = N_values[i + 1]
        
        rmse_curr = rmse_dict_int[N_curr]
        rmse_next = rmse_dict_int[N_next]
        
        if rmse_curr > 0:
            ratio = rmse_next / rmse_curr
            if ratio > ratio_threshold:
                return N_next
    
    return N_values[-1]


def compute_all_transitions(result: dict) -> dict:
    """Compute transition points from all RMSE types."""
    transitions = {}
    
    rmse_pos = result.get('rmse_position', result.get('rmse', {}))
    if rmse_pos:
        transitions['N_transition_position'] = detect_transition_ratio(rmse_pos)
    else:
        transitions['N_transition_position'] = None
    
    rmse_int = result.get('rmse_intensity', {})
    if rmse_int:
        transitions['N_transition_intensity'] = detect_transition_ratio(rmse_int)
    else:
        transitions['N_transition_intensity'] = None
    
    pos_trans = transitions['N_transition_position']
    int_trans = transitions['N_transition_intensity']
    
    if pos_trans is not None and int_trans is not None:
        transitions['N_transition_either'] = min(pos_trans, int_trans)
    elif pos_trans is not None:
        transitions['N_transition_either'] = pos_trans
    elif int_trans is not None:
        transitions['N_transition_either'] = int_trans
    else:
        transitions['N_transition_either'] = None
    
    return transitions


# =============================================================================
# ANALYSIS FOR SINGLE TEST CASE
# =============================================================================

def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute correlation, handling edge cases."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def analyze_test_case(results: List[dict], test_case: str = None) -> dict:
    """
    Analyze results for a single test case.
    
    Computes N_max using all three formulas and checks which one
    best predicts N_transition.
    """
    if not results:
        return {'error': 'No results to analyze'}
    
    # Auto-detect test case from results if not specified
    if test_case is None:
        test_case = results[0].get('test_case', 'unknown')
    
    # Collect data
    data = {
        'n_star_max': [],
        'N_max_same_radius': [],    # N_max = n*
        'N_max_same_angle': [],     # N_max = (1/2)n*
        'N_max_general': [],        # N_max = (2/3)n*
        'N_max_expected': [],       # N_max using correct formula for this test case
        'N_transition_position': [],
        'N_transition_intensity': [],
        'N_transition_either': [],
    }
    
    for result in results:
        n_star = result.get('n_star_max', result.get('n_star_actual', 0))
        
        data['n_star_max'].append(n_star)
        data['N_max_same_radius'].append(n_star)           # 1.0 * n*
        data['N_max_same_angle'].append(0.5 * n_star)      # 0.5 * n*
        data['N_max_general'].append((2.0/3.0) * n_star)   # (2/3) * n*
        
        # Expected N_max for THIS test case
        multiplier = FORMULA_MULTIPLIERS.get(test_case, 2.0/3.0)
        data['N_max_expected'].append(multiplier * n_star)
        
        # Transitions
        transitions = compute_all_transitions(result)
        data['N_transition_position'].append(transitions['N_transition_position'])
        data['N_transition_intensity'].append(transitions['N_transition_intensity'])
        data['N_transition_either'].append(transitions['N_transition_either'])
    
    # Convert to arrays
    for key in data:
        data[key] = np.array([x if x is not None else np.nan for x in data[key]])
    
    # Compute correlations and errors for each formula vs each transition type
    formulas = ['N_max_same_radius', 'N_max_same_angle', 'N_max_general']
    transitions = ['N_transition_position', 'N_transition_intensity', 'N_transition_either']
    
    correlations = {}
    errors = {}
    
    for formula in formulas:
        correlations[formula] = {}
        errors[formula] = {}
        
        formula_data = data[formula]
        valid_formula = ~np.isnan(formula_data)
        
        for trans in transitions:
            trans_data = data[trans]
            valid_trans = ~np.isnan(trans_data)
            valid = valid_formula & valid_trans
            
            if np.sum(valid) >= 2:
                corr = safe_correlation(formula_data[valid], trans_data[valid])
                error = trans_data[valid] - formula_data[valid]
                
                correlations[formula][trans] = corr
                errors[formula][trans] = {
                    'mean': float(np.mean(error)),
                    'std': float(np.std(error)),
                    'n': int(np.sum(valid)),
                }
            else:
                correlations[formula][trans] = 0.0
                errors[formula][trans] = {'mean': 0.0, 'std': 0.0, 'n': 0}
    
    # Find best formula for this test case
    expected_formula = {
        'same_radius': 'N_max_same_radius',
        'same_angle': 'N_max_same_angle',
        'general': 'N_max_general',
    }.get(test_case, 'N_max_general')
    
    # Check if expected formula is indeed the best
    best_corr = -1
    best_formula = ''
    for formula in formulas:
        corr = correlations[formula]['N_transition_position']
        if corr > best_corr:
            best_corr = corr
            best_formula = formula
    
    # Summary
    summary = {
        'test_case': test_case,
        'expected_formula': EXPECTED_FORMULAS.get(test_case, 'unknown'),
        'n_seeds': len(results),
        
        # n* statistics
        'n_star_max_mean': float(np.nanmean(data['n_star_max'])),
        'n_star_max_std': float(np.nanstd(data['n_star_max'])),
        
        # Expected N_max (using correct formula)
        'N_max_expected_mean': float(np.nanmean(data['N_max_expected'])),
        'N_max_expected_std': float(np.nanstd(data['N_max_expected'])),
        
        # Transitions
        'N_transition_position_mean': float(np.nanmean(data['N_transition_position'])),
        'N_transition_position_std': float(np.nanstd(data['N_transition_position'])),
        
        # Correlation with expected formula
        'correlation_expected': correlations[expected_formula]['N_transition_position'],
        'error_expected_mean': errors[expected_formula]['N_transition_position']['mean'],
        'error_expected_std': errors[expected_formula]['N_transition_position']['std'],
        
        # Best formula found
        'best_formula': best_formula,
        'best_correlation': best_corr,
        'expected_is_best': best_formula == expected_formula,
        
        # Full correlation matrix
        'correlation_matrix': correlations,
        'error_matrix': errors,
        'raw_data': {k: v.tolist() for k, v in data.items()},
    }
    
    return summary


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots_single_case(results: List[dict], summary: dict, output_dir: str, test_case: str):
    """Generate plots for a single test case."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = {k: np.array(v) for k, v in summary['raw_data'].items()}
    
    # =========================================================================
    # Plot 1: N_transition vs N_max (3 formulas)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    formulas = [
        ('N_max_same_radius', 'N_max = n*', 'steelblue'),
        ('N_max_same_angle', 'N_max = (1/2)n*', 'seagreen'),
        ('N_max_general', 'N_max = (2/3)n*', 'coral'),
    ]
    
    expected_formula = {
        'same_radius': 'N_max_same_radius',
        'same_angle': 'N_max_same_angle',
        'general': 'N_max_general',
    }.get(test_case, 'N_max_general')
    
    for idx, (formula_key, formula_label, color) in enumerate(formulas):
        ax = axes[idx]
        
        pred_data = data[formula_key]
        trans_data = data['N_transition_position']
        valid = ~np.isnan(pred_data) & ~np.isnan(trans_data)
        
        if np.sum(valid) > 0:
            ax.scatter(pred_data[valid], trans_data[valid], alpha=0.6, s=50, 
                      color=color, edgecolors='black', linewidth=0.5)
            
            min_val = min(np.nanmin(pred_data[valid]), np.nanmin(trans_data[valid])) - 1
            max_val = max(np.nanmax(pred_data[valid]), np.nanmax(trans_data[valid])) + 1
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7)
            
            corr = summary['correlation_matrix'][formula_key]['N_transition_position']
            
            # Mark if this is the expected formula
            title_suffix = " (EXPECTED)" if formula_key == expected_formula else ""
            ax.set_title(f'{formula_label}{title_suffix}\nCorr = {corr:.3f}', 
                        fontsize=12, fontweight='bold')
        
        ax.set_xlabel('N_max (predicted)', fontsize=11)
        ax.set_ylabel('N_transition (actual)', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Test Case: {test_case}\nN_transition vs N_max (Different Formulas)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{test_case}_formula_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: {test_case}_formula_comparison.png")
    
    # =========================================================================
    # Plot 2: n* distribution
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid = ~np.isnan(data['n_star_max'])
    ax.hist(data['n_star_max'][valid], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=np.nanmean(data['n_star_max']), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {np.nanmean(data["n_star_max"]):.1f}')
    ax.set_xlabel('n*_max (max usable mode)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Test Case: {test_case}\nn* Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{test_case}_n_star_histogram.png'), dpi=150)
    plt.close()
    print(f"  Saved: {test_case}_n_star_histogram.png")
    
    # =========================================================================
    # Plot 3: Error histogram (expected formula)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pred_data = data['N_max_expected']
    trans_data = data['N_transition_position']
    valid = ~np.isnan(pred_data) & ~np.isnan(trans_data)
    
    if np.sum(valid) > 0:
        error = trans_data[valid] - pred_data[valid]
        ax.hist(error, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=np.mean(error), color='green', linestyle='-', linewidth=2,
                  label=f'Mean = {np.mean(error):.2f}')
        ax.set_xlabel(f'N_transition - N_max_expected', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Test Case: {test_case}\nError Distribution (using {EXPECTED_FORMULAS[test_case]})\n'
                    f'Mean = {np.mean(error):.2f}, Std = {np.std(error):.2f}', fontsize=14)
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{test_case}_error_histogram.png'), dpi=150)
    plt.close()
    print(f"  Saved: {test_case}_error_histogram.png")
    
    # =========================================================================
    # Plot 4: RMSE curves
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    N_max_expected_mean = np.nanmean(data['N_max_expected'])
    
    for result in results:
        rmse_pos = result.get('rmse_position', result.get('rmse', {}))
        if rmse_pos:
            N_vals = sorted([int(k) for k in rmse_pos.keys()])
            rmse_vals = [rmse_pos.get(str(N), rmse_pos.get(N, 0)) for N in N_vals]
            axes[0].plot(N_vals, rmse_vals, 'b-', alpha=0.15, linewidth=1)
        
        rmse_int = result.get('rmse_intensity', {})
        if rmse_int:
            N_vals = sorted([int(k) for k in rmse_int.keys()])
            rmse_vals = [rmse_int.get(str(N), rmse_int.get(N, 0)) for N in N_vals]
            axes[1].plot(N_vals, rmse_vals, 'b-', alpha=0.15, linewidth=1)
    
    for ax, title in zip(axes, ['Position RMSE', 'Intensity RMSE']):
        ax.axvline(x=N_max_expected_mean, color='green', linestyle='--', linewidth=2,
                  label=f'N_max_expected = {N_max_expected_mean:.1f}')
        ax.set_xlabel('Number of Sources (N)', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title(f'{title}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle(f'Test Case: {test_case}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{test_case}_rmse_curves.png'), dpi=150)
    plt.close()
    print(f"  Saved: {test_case}_rmse_curves.png")


def generate_comparison_plot(all_summaries: Dict[str, dict], output_dir: str):
    """Generate comparison plot across all test cases."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # Comparison bar chart: Correlation of expected formula
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    test_cases = list(all_summaries.keys())
    correlations = [all_summaries[tc].get('correlation_expected', 0) for tc in test_cases]
    expected_is_best = [all_summaries[tc].get('expected_is_best', False) for tc in test_cases]
    
    colors = ['green' if is_best else 'orange' for is_best in expected_is_best]
    
    bars = ax.bar(test_cases, correlations, color=colors, edgecolor='black', alpha=0.7)
    
    for i, (tc, corr) in enumerate(zip(test_cases, correlations)):
        formula = EXPECTED_FORMULAS.get(tc, '?')
        ax.text(i, corr + 0.02, f'{corr:.3f}', ha='center', fontsize=12, fontweight='bold')
        ax.text(i, -0.15, formula, ha='center', fontsize=10, style='italic')
    
    ax.set_ylabel('Correlation (N_max_expected vs N_transition)', fontsize=12)
    ax.set_title('Does the Expected Formula Predict Transitions?\n'
                '(Green = expected formula is best, Orange = another formula is better)', 
                fontsize=14)
    ax.set_ylim(-0.2, 1.1)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good threshold (0.7)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_case_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: test_case_comparison.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Aggregate source config experiment results")
    parser.add_argument('--results-dir', type=str, 
                       help='Directory containing results for single test case')
    parser.add_argument('--results-dirs', type=str, nargs='+',
                       help='Multiple result directories')
    parser.add_argument('--all-cases', action='store_true',
                       help='Load all test cases from stat_results_<case>/ directories')
    parser.add_argument('--output', type=str, default='statistical_validation_summary.json',
                       help='Output JSON file')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Determine which directories to load
    results_dirs = {}
    
    if args.all_cases:
        for tc in TEST_CASES:
            dir_name = f'stat_results_{tc}'
            if os.path.exists(dir_name):
                results_dirs[tc] = dir_name
    elif args.results_dirs:
        for dir_path in args.results_dirs:
            # Try to infer test case from directory name
            for tc in TEST_CASES:
                if tc in dir_path:
                    results_dirs[tc] = dir_path
                    break
            else:
                results_dirs[os.path.basename(dir_path)] = dir_path
    elif args.results_dir:
        # Single directory - try to infer test case
        for tc in TEST_CASES:
            if tc in args.results_dir:
                results_dirs[tc] = args.results_dir
                break
        else:
            results_dirs['unknown'] = args.results_dir
    else:
        # Default: try all cases
        for tc in TEST_CASES:
            dir_name = f'stat_results_{tc}'
            if os.path.exists(dir_name):
                results_dirs[tc] = dir_name
        
        if not results_dirs:
            # Fallback to old directory name
            if os.path.exists('stat_results'):
                results_dirs['unknown'] = 'stat_results'
    
    if not results_dirs:
        print("Error: No result directories found!")
        print("Specify --results-dir, --results-dirs, or --all-cases")
        return
    
    print(f"Loading results from {len(results_dirs)} directories...")
    
    # Load and analyze each test case
    all_results = {}
    all_summaries = {}
    
    for test_case, dir_path in results_dirs.items():
        print(f"\n{'='*60}")
        print(f"TEST CASE: {test_case}")
        print(f"Directory: {dir_path}")
        print(f"{'='*60}")
        
        results = load_all_results(dir_path)
        
        if not results:
            print(f"  No results found, skipping...")
            continue
        
        all_results[test_case] = results
        
        # Analyze
        summary = analyze_test_case(results, test_case)
        all_summaries[test_case] = summary
        
        # Print summary
        print(f"\n  Expected formula: {summary.get('expected_formula', 'unknown')}")
        print(f"  n*_max mean: {summary.get('n_star_max_mean', 0):.2f} ± {summary.get('n_star_max_std', 0):.2f}")
        print(f"  N_max_expected mean: {summary.get('N_max_expected_mean', 0):.2f}")
        print(f"  N_transition mean: {summary.get('N_transition_position_mean', 0):.2f}")
        print(f"  Correlation (expected): {summary.get('correlation_expected', 0):.3f}")
        print(f"  Error mean: {summary.get('error_expected_mean', 0):.2f} ± {summary.get('error_expected_std', 0):.2f}")
        print(f"  Expected is best formula: {summary.get('expected_is_best', False)}")
        
        # Generate plots
        if not args.no_plots:
            plot_dir = f'statistical_plots_{test_case}'
            print(f"\n  Generating plots in {plot_dir}/...")
            generate_plots_single_case(results, summary, plot_dir, test_case)
    
    # Generate comparison plot if multiple test cases
    if len(all_summaries) > 1 and not args.no_plots:
        print(f"\n{'='*60}")
        print("GENERATING COMPARISON PLOTS")
        print(f"{'='*60}")
        generate_comparison_plot(all_summaries, 'statistical_plots_comparison')
    
    # Save combined output
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'test_cases': list(all_summaries.keys()),
        'summaries': {tc: {k: v for k, v in s.items() if k != 'raw_data'} 
                     for tc, s in all_summaries.items()},
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {args.output}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: Does Each Formula Work?")
    print(f"{'='*70}")
    print(f"{'Test Case':<15} {'Formula':<20} {'Correlation':<12} {'Best?':<8}")
    print("-" * 70)
    
    for tc in TEST_CASES:
        if tc in all_summaries:
            s = all_summaries[tc]
            formula = EXPECTED_FORMULAS[tc]
            corr = s.get('correlation_expected', 0)
            is_best = '✓ YES' if s.get('expected_is_best', False) else '✗ NO'
            print(f"{tc:<15} {formula:<20} {corr:<12.3f} {is_best:<8}")
    
    print("-" * 70)
    
    # Interpretation
    all_validated = all(s.get('expected_is_best', False) and s.get('correlation_expected', 0) > 0.5 
                       for s in all_summaries.values())
    
    if all_validated:
        print("\n✓ THEORY VALIDATED: Each test case's expected formula best predicts transitions!")
    else:
        print("\n~ MIXED RESULTS: Not all expected formulas are optimal predictors.")
        print("  Check individual test case plots for details.")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
