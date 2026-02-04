#!/usr/bin/env python3
"""
Assemble statistics from SLURM .out log files into a single JSON.

Usage:
    python assemble_logs.py --logs-dir logs/ --output assembled_logs.json
"""

import os
import re
import json
import argparse
from glob import glob


def parse_log_file(filepath):
    """
    Parse a single .out log file and extract statistics.
    
    Returns dict with seed, rmse values, and summary stats.
    """
    result = {
        'filepath': filepath,
        'seed': None,
        'rmse_position': {},
        'rmse_intensity': {},
        'n_star_predicted': None,
        'n_star_actual': None,
        'N_max_predicted': None,
        'N_max_actual': None,
        'N_transition': None,
        'time_seconds': None,
    }
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract seed from "Statistical Validation - Seed X" or "SUMMARY: Seed X"
    seed_match = re.search(r'Seed\s+(\d+)', content)
    if seed_match:
        result['seed'] = int(seed_match.group(1))
    
    # Extract RMSE lines: "Seed X, N=Y: RMSE_pos = Z, RMSE_int = W"
    rmse_pattern = r'Seed\s+\d+,\s+N=(\d+):\s+RMSE_pos\s*=\s*([\d.]+),\s+RMSE_int\s*=\s*([\d.]+)'
    for match in re.finditer(rmse_pattern, content):
        N = int(match.group(1))
        rmse_pos = float(match.group(2))
        rmse_int = float(match.group(3))
        result['rmse_position'][N] = rmse_pos
        result['rmse_intensity'][N] = rmse_int
    
    # Extract summary stats
    # n*_predicted = 25.82
    n_star_pred_match = re.search(r'n\*_predicted\s*=\s*([\d.]+)', content)
    if n_star_pred_match:
        result['n_star_predicted'] = float(n_star_pred_match.group(1))
    
    # n*_actual = 13
    n_star_actual_match = re.search(r'n\*_actual\s*=\s*(\d+)', content)
    if n_star_actual_match:
        result['n_star_actual'] = int(n_star_actual_match.group(1))
    
    # N_max_predicted = 17.22
    N_max_pred_match = re.search(r'N_max_predicted\s*=\s*([\d.]+)', content)
    if N_max_pred_match:
        result['N_max_predicted'] = float(N_max_pred_match.group(1))
    
    # N_max_actual = 8.67
    N_max_actual_match = re.search(r'N_max_actual\s*=\s*([\d.]+)', content)
    if N_max_actual_match:
        result['N_max_actual'] = float(N_max_actual_match.group(1))
    
    # N_transition = 12
    N_transition_match = re.search(r'N_transition\s*=\s*(\d+)', content)
    if N_transition_match:
        result['N_transition'] = int(N_transition_match.group(1))
    
    # Time: 3306.9s
    time_match = re.search(r'Time:\s*([\d.]+)s', content)
    if time_match:
        result['time_seconds'] = float(time_match.group(1))
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Assemble .out log files into JSON")
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Directory containing .out files')
    parser.add_argument('--pattern', type=str, default='stat_seed*.out',
                       help='Glob pattern for log files')
    parser.add_argument('--output', type=str, default='assembled_logs.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Find all log files
    log_pattern = os.path.join(args.logs_dir, args.pattern)
    log_files = sorted(glob(log_pattern))
    
    print(f"Found {len(log_files)} log files matching {log_pattern}")
    
    if not log_files:
        print("No log files found!")
        return
    
    # Parse all log files
    results = []
    for filepath in log_files:
        try:
            result = parse_log_file(filepath)
            if result['seed'] is not None:
                results.append(result)
                print(f"  Parsed seed {result['seed']}: N_transition={result['N_transition']}")
            else:
                print(f"  Warning: Could not extract seed from {filepath}")
        except Exception as e:
            print(f"  Error parsing {filepath}: {e}")
    
    # Sort by seed
    results.sort(key=lambda x: x['seed'] if x['seed'] is not None else -1)
    
    # Save to JSON
    output_data = {
        'n_logs': len(results),
        'logs_dir': args.logs_dir,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved {len(results)} parsed logs to {args.output}")
    
    # Print quick summary
    if results:
        seeds = [r['seed'] for r in results if r['seed'] is not None]
        transitions = [r['N_transition'] for r in results if r['N_transition'] is not None]
        
        print(f"\nQuick summary:")
        print(f"  Seeds: {min(seeds)} to {max(seeds)}")
        print(f"  N_transition range: {min(transitions)} to {max(transitions)}")
        print(f"  N_transition mean: {sum(transitions)/len(transitions):.2f}")


if __name__ == '__main__':
    main()
