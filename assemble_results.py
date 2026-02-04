#!/usr/bin/env python3
"""
Assemble all results.json files from stat_results/ into a single JSON.

Usage:
    python assemble_results.py --results-dir stat_results/ --output assembled_results.json
"""

import os
import json
import argparse
from glob import glob


def find_result_files(results_dir):
    """
    Find all results.json files in seed*_*/ subdirectories.
    """
    pattern = os.path.join(results_dir, 'seed*_*', 'results.json')
    return sorted(glob(pattern))


def load_result_file(filepath):
    """
    Load a single results.json file.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Add source filepath for reference
    data['_source_file'] = filepath
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Assemble results.json files into single JSON")
    parser.add_argument('--results-dir', type=str, default='stat_results',
                       help='Directory containing seed*_*/ subdirectories')
    parser.add_argument('--output', type=str, default='assembled_results.json',
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Find all results.json files
    result_files = find_result_files(args.results_dir)
    
    print(f"Found {len(result_files)} results.json files in {args.results_dir}")
    
    if not result_files:
        print("No results.json files found!")
        return
    
    # Load all results
    results = []
    for filepath in result_files:
        try:
            result = load_result_file(filepath)
            results.append(result)
            seed = result.get('seed', 'unknown')
            n_transition = result.get('N_transition', 'N/A')
            print(f"  Loaded seed {seed}: N_transition={n_transition}")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
    
    # Sort by seed
    results.sort(key=lambda x: x.get('seed', -1))
    
    # Save to JSON
    output_data = {
        'n_results': len(results),
        'results_dir': args.results_dir,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved {len(results)} results to {args.output}")
    
    # Print quick summary
    if results:
        seeds = [r['seed'] for r in results if r.get('seed') is not None]
        n_star_max_vals = [r['n_star_max'] for r in results if r.get('n_star_max') is not None]
        K_vals = [r['K'] for r in results if r.get('K') is not None]
        transitions = [r['N_transition'] for r in results if r.get('N_transition') is not None]
        
        print(f"\nQuick summary:")
        print(f"  Seeds: {min(seeds)} to {max(seeds)} ({len(seeds)} total)")
        
        if n_star_max_vals:
            print(f"  n*_max range: {min(n_star_max_vals)} to {max(n_star_max_vals)}, mean={sum(n_star_max_vals)/len(n_star_max_vals):.2f}")
        
        if K_vals:
            print(f"  K range: {min(K_vals)} to {max(K_vals)}, mean={sum(K_vals)/len(K_vals):.2f}")
        
        if transitions:
            print(f"  N_transition range: {min(transitions)} to {max(transitions)}, mean={sum(transitions)/len(transitions):.2f}")


if __name__ == '__main__':
    main()
