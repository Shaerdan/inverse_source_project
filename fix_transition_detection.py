#!/usr/bin/env python3
"""
Post-Processing Script: Fix Transition Detection in Existing Results
=====================================================================

Updates existing results.json files with improved transition detection:
1. N_transition_all: First jump > threshold (original behavior)
2. N_transition_skip2: First jump > threshold, excluding 2→4 transition
3. N_transition_max_jump: The N where the LARGEST ratio jump occurs
4. All jump candidates with their ratios

Usage:
    # Process all test cases
    python fix_transition_detection.py --all-cases
    
    # Process specific directory
    python fix_transition_detection.py --results-dir stat_results_same_radius_random_rho
    
    # Dry run (show what would change without saving)
    python fix_transition_detection.py --all-cases --dry-run
"""

import os
import json
import argparse
from typing import Dict, List, Tuple
from datetime import datetime


def detect_all_transitions(rmse_dict: Dict, ratio_threshold: float = 2.0) -> Dict:
    """
    Comprehensive transition detection.
    
    Returns dict with:
    - all_jumps: List of (N, ratio) for all jumps > threshold
    - first_jump: First N where ratio > threshold (original method)
    - first_jump_skip2: First N where ratio > threshold, excluding 2→4
    - max_jump: N where the largest ratio occurs
    - max_jump_ratio: The largest ratio value
    """
    # Convert string keys to int if needed
    rmse_dict_int = {int(k): float(v) for k, v in rmse_dict.items()}
    N_values = sorted(rmse_dict_int.keys())
    
    if len(N_values) < 2:
        return {
            'all_jumps': [],
            'first_jump': N_values[0] if N_values else None,
            'first_jump_skip2': N_values[0] if N_values else None,
            'max_jump': N_values[0] if N_values else None,
            'max_jump_ratio': 0.0,
        }
    
    # Collect all jumps
    all_jumps = []
    for i in range(len(N_values) - 1):
        N_curr = N_values[i]
        N_next = N_values[i + 1]
        
        rmse_curr = rmse_dict_int[N_curr]
        rmse_next = rmse_dict_int[N_next]
        
        if rmse_curr > 0:
            ratio = rmse_next / rmse_curr
            all_jumps.append({
                'N_from': N_curr,
                'N_to': N_next,
                'ratio': ratio,
                'rmse_from': rmse_curr,
                'rmse_to': rmse_next,
            })
    
    # Find first jump > threshold (original)
    first_jump = N_values[-1]
    for jump in all_jumps:
        if jump['ratio'] > ratio_threshold:
            first_jump = jump['N_to']
            break
    
    # Find first jump > threshold, excluding 2→4 transition
    first_jump_skip2 = N_values[-1]
    for jump in all_jumps:
        if jump['ratio'] > ratio_threshold and jump['N_from'] != 2:
            first_jump_skip2 = jump['N_to']
            break
    
    # Find max jump
    max_jump = N_values[-1]
    max_jump_ratio = 0.0
    for jump in all_jumps:
        if jump['ratio'] > max_jump_ratio:
            max_jump_ratio = jump['ratio']
            max_jump = jump['N_to']
    
    # Find max jump excluding 2→4
    max_jump_skip2 = N_values[-1]
    max_jump_ratio_skip2 = 0.0
    for jump in all_jumps:
        if jump['N_from'] != 2 and jump['ratio'] > max_jump_ratio_skip2:
            max_jump_ratio_skip2 = jump['ratio']
            max_jump_skip2 = jump['N_to']
    
    # Significant jumps only (> threshold)
    significant_jumps = [j for j in all_jumps if j['ratio'] > ratio_threshold]
    significant_jumps_skip2 = [j for j in significant_jumps if j['N_from'] != 2]
    
    return {
        'all_jumps': all_jumps,
        'significant_jumps': significant_jumps,
        'significant_jumps_skip2': significant_jumps_skip2,
        'first_jump': first_jump,
        'first_jump_skip2': first_jump_skip2,
        'max_jump': max_jump,
        'max_jump_ratio': max_jump_ratio,
        'max_jump_skip2': max_jump_skip2,
        'max_jump_ratio_skip2': max_jump_ratio_skip2,
    }


def update_result_file(result_path: str, dry_run: bool = False) -> Dict:
    """
    Update a single results.json file with improved transition detection.
    
    Returns summary of changes made.
    """
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    changes = {}
    
    # Process position RMSE
    rmse_pos = result.get('rmse_position', result.get('rmse', {}))
    if rmse_pos:
        trans_pos = detect_all_transitions(rmse_pos)
        
        old_transition = result.get('N_transition', result.get('N_transition_pos_ratio'))
        
        # Add new fields
        result['transition_analysis_position'] = {
            'all_jumps': trans_pos['all_jumps'],
            'significant_jumps': trans_pos['significant_jumps'],
            'significant_jumps_skip2': trans_pos['significant_jumps_skip2'],
        }
        result['N_transition_pos_first'] = trans_pos['first_jump']
        result['N_transition_pos_skip2'] = trans_pos['first_jump_skip2']
        result['N_transition_pos_max_jump'] = trans_pos['max_jump']
        result['N_transition_pos_max_ratio'] = trans_pos['max_jump_ratio']
        result['N_transition_pos_max_jump_skip2'] = trans_pos['max_jump_skip2']
        result['N_transition_pos_max_ratio_skip2'] = trans_pos['max_jump_ratio_skip2']
        
        changes['position'] = {
            'old_transition': old_transition,
            'new_first': trans_pos['first_jump'],
            'new_skip2': trans_pos['first_jump_skip2'],
            'new_max_jump': trans_pos['max_jump'],
            'max_ratio': trans_pos['max_jump_ratio'],
        }
    
    # Process intensity RMSE
    rmse_int = result.get('rmse_intensity', {})
    if rmse_int:
        trans_int = detect_all_transitions(rmse_int)
        
        result['transition_analysis_intensity'] = {
            'all_jumps': trans_int['all_jumps'],
            'significant_jumps': trans_int['significant_jumps'],
            'significant_jumps_skip2': trans_int['significant_jumps_skip2'],
        }
        result['N_transition_int_first'] = trans_int['first_jump']
        result['N_transition_int_skip2'] = trans_int['first_jump_skip2']
        result['N_transition_int_max_jump'] = trans_int['max_jump']
        result['N_transition_int_max_ratio'] = trans_int['max_jump_ratio']
        result['N_transition_int_max_jump_skip2'] = trans_int['max_jump_skip2']
        result['N_transition_int_max_ratio_skip2'] = trans_int['max_jump_ratio_skip2']
        
        changes['intensity'] = {
            'new_first': trans_int['first_jump'],
            'new_skip2': trans_int['first_jump_skip2'],
            'new_max_jump': trans_int['max_jump'],
            'max_ratio': trans_int['max_jump_ratio'],
        }
    
    # Compute errors against N_max_predicted
    N_max_pred = result.get('N_max_predicted', 0)
    if N_max_pred and 'N_transition_pos_skip2' in result:
        result['error_vs_first'] = result['N_transition_pos_first'] - N_max_pred
        result['error_vs_skip2'] = result['N_transition_pos_skip2'] - N_max_pred
        result['error_vs_max_jump'] = result['N_transition_pos_max_jump'] - N_max_pred
        result['error_vs_max_jump_skip2'] = result['N_transition_pos_max_jump_skip2'] - N_max_pred
        
        changes['errors'] = {
            'N_max_predicted': N_max_pred,
            'error_first': result['error_vs_first'],
            'error_skip2': result['error_vs_skip2'],
            'error_max_jump': result['error_vs_max_jump'],
            'error_max_jump_skip2': result['error_vs_max_jump_skip2'],
        }
    
    # Add processing timestamp
    result['transition_fix_timestamp'] = datetime.now().isoformat()
    
    # Save if not dry run
    if not dry_run:
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    return changes


def find_result_files(base_dir: str) -> List[str]:
    """Find all results.json files in the directory."""
    result_files = []
    
    if not os.path.exists(base_dir):
        return result_files
    
    for name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, name)
        if os.path.isdir(folder_path):
            result_path = os.path.join(folder_path, 'results.json')
            if os.path.exists(result_path):
                result_files.append(result_path)
    
    return sorted(result_files)


def process_directory(base_dir: str, dry_run: bool = False, verbose: bool = True) -> Dict:
    """Process all results in a directory."""
    result_files = find_result_files(base_dir)
    
    if not result_files:
        print(f"  No result files found in {base_dir}")
        return {'processed': 0, 'errors': 0}
    
    print(f"  Found {len(result_files)} result files")
    
    processed = 0
    errors = 0
    
    # Track cases where skip2 differs from first
    skip2_differs = []
    
    for result_path in result_files:
        try:
            changes = update_result_file(result_path, dry_run=dry_run)
            processed += 1
            
            # Check if skip2 made a difference
            if 'position' in changes:
                pos = changes['position']
                if pos['new_first'] != pos['new_skip2']:
                    seed_folder = os.path.basename(os.path.dirname(result_path))
                    skip2_differs.append({
                        'seed': seed_folder,
                        'first': pos['new_first'],
                        'skip2': pos['new_skip2'],
                        'max_jump': pos['new_max_jump'],
                    })
            
            if verbose and processed % 20 == 0:
                print(f"    Processed {processed}/{len(result_files)}")
                
        except Exception as e:
            errors += 1
            print(f"    Error processing {result_path}: {e}")
    
    # Summary
    summary = {
        'processed': processed,
        'errors': errors,
        'skip2_differs_count': len(skip2_differs),
        'skip2_differs_examples': skip2_differs[:10],  # First 10 examples
    }
    
    if skip2_differs:
        print(f"\n  Cases where skip2 differs from first: {len(skip2_differs)}/{processed}")
        if verbose and len(skip2_differs) <= 10:
            for case in skip2_differs:
                print(f"    {case['seed']}: first={case['first']}, skip2={case['skip2']}, max_jump={case['max_jump']}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fix transition detection in existing results")
    parser.add_argument('--results-dir', type=str, help='Single results directory to process')
    parser.add_argument('--all-cases', action='store_true', help='Process all test case directories')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without saving')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    directories = []
    
    if args.all_cases:
        # Find all stat_results_* directories
        for name in os.listdir('.'):
            if name.startswith('stat_results_') and os.path.isdir(name):
                directories.append(name)
    elif args.results_dir:
        directories.append(args.results_dir)
    else:
        # Default: find all
        for name in os.listdir('.'):
            if name.startswith('stat_results_') and os.path.isdir(name):
                directories.append(name)
    
    if not directories:
        print("No result directories found!")
        print("Use --results-dir <path> or --all-cases")
        return
    
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(directories)} directories...")
    print()
    
    total_processed = 0
    total_errors = 0
    total_skip2_differs = 0
    
    for dir_path in sorted(directories):
        print(f"Processing: {dir_path}")
        summary = process_directory(dir_path, dry_run=args.dry_run, verbose=not args.quiet)
        total_processed += summary['processed']
        total_errors += summary['errors']
        total_skip2_differs += summary['skip2_differs_count']
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total files processed: {total_processed}")
    print(f"  Total errors: {total_errors}")
    print(f"  Cases where skip2 differs: {total_skip2_differs}")
    if args.dry_run:
        print("\n  [DRY RUN - no files were modified]")
    else:
        print("\n  All result files have been updated with new transition fields:")
        print("    - N_transition_pos_first: Original first-jump method")
        print("    - N_transition_pos_skip2: First jump excluding 2→4")
        print("    - N_transition_pos_max_jump: N with largest ratio jump")
        print("    - error_vs_first, error_vs_skip2, error_vs_max_jump: Errors vs N_max_predicted")


if __name__ == '__main__':
    main()
