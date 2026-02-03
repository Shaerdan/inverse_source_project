#!/usr/bin/env python3
"""
Wrapper for test_bound_theory.py with proper output organization.

Creates output folder: results/{domain}_rho{rho:.2f}_seed{seed}_{hash}/
Saves: config.json, results.json, all plots

Usage:
    python run_bound_experiment.py --domain disk --rho 0.7 --seed 42
    python run_bound_experiment.py --domain all --rho 0.7 --seed 42
"""

import os
import sys
import json
import hashlib
import argparse
import numpy as np
from datetime import datetime

def generate_hash(length=8):
    """Generate unique run hash."""
    data = f"{datetime.now().isoformat()}{os.urandom(8)}".encode()
    return hashlib.sha256(data).hexdigest()[:length]

def run_experiment(domain, rho, sigma_noise, sensors, restarts, seed, threshold, base_output_dir):
    """Run experiment for a single domain and rho."""
    
    # Generate hash
    run_hash = generate_hash()
    
    # Create output directory - includes seed in folder name
    folder_name = f"{domain}_rho{rho:.2f}_seed{seed}_{run_hash}"
    output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"RUN_HASH={run_hash}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"{'='*70}")
    
    # Save config
    config = {
        'run_hash': run_hash,
        'timestamp': datetime.now().isoformat(),
        'domain': domain,
        'rho': rho,
        'sigma_noise': sigma_noise,
        'n_sensors': sensors,
        'n_restarts': restarts,
        'seed': seed,
        'rmse_threshold': threshold,
        'sigma_four': sigma_noise / (sensors ** 0.5),
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")
    
    # Import and run the actual test
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from test_bound_theory import run_bound_validation
    
    # Run validation - uses user's tested version
    # Note: user's version does NOT take run_hash parameter
    results = run_bound_validation(
        domain=domain,
        rho_target=rho,
        sigma_noise=sigma_noise,
        n_sensors=sensors,
        n_restarts=restarts,
        seed=seed,
        rmse_threshold=threshold,
        save_plots=True,
        output_dir=output_dir
    )
    
    # Convert TestResult objects to dicts for JSON serialization
    results_data = []
    for r in results:
        results_data.append({
            'N_tested': r.N_tested,
            'rho_target': r.rho_target,
            'rho_actual': [float(x) for x in r.rho_actual],
            'rho_min': float(np.min(r.rho_actual)),
            'sigma_four': r.sigma_four,
            'N_max_theory': r.N_max_theory,
            'expected_success': r.expected_success,
            'actual_success': r.actual_success,
            'position_rmse': r.position_rmse,
            'time_seconds': r.time_seconds,
        })
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved results: {results_path}")
    
    return run_hash, output_dir, results

def main():
    parser = argparse.ArgumentParser(description="Run bound validation experiment")
    parser.add_argument('--domain', type=str, default='disk',
                       choices=['disk', 'ellipse', 'brain', 'all'])
    parser.add_argument('--rho', type=float, default=0.7)
    parser.add_argument('--sigma-noise', type=float, default=0.001)
    parser.add_argument('--sensors', type=int, default=100)
    parser.add_argument('--restarts', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--output-dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine domains to run
    if args.domain == 'all':
        domains = ['disk', 'ellipse', 'brain']
    else:
        domains = [args.domain]
    
    # Run for each domain
    all_results = {}
    for domain in domains:
        run_hash, output_dir, results = run_experiment(
            domain=domain,
            rho=args.rho,
            sigma_noise=args.sigma_noise,
            sensors=args.sensors,
            restarts=args.restarts,
            seed=args.seed,
            threshold=args.threshold,
            base_output_dir=args.output_dir
        )
        all_results[domain] = {
            'hash': run_hash,
            'output_dir': output_dir,
        }
    
    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    for domain, info in all_results.items():
        print(f"  {domain}: {info['output_dir']}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
