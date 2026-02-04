#!/usr/bin/env python3
"""
Wrapper for test_statistical_validation.py with output organization.

Creates output folder: stat_results/seed{seed:03d}_{hash}/
Saves: config.json, results.json

Usage:
    python run_statistical_experiment.py --seed 0
    python run_statistical_experiment.py --seed 42 --output-dir stat_results/
"""

import os
import sys
import json
import hashlib
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_hash(length=8):
    """Generate unique run hash."""
    data = f"{datetime.now().isoformat()}{os.urandom(8)}".encode()
    return hashlib.sha256(data).hexdigest()[:length]


def run_experiment(seed, rho, sigma_noise, n_sensors, n_restarts, N_values, base_output_dir):
    """Run statistical validation for one seed with output organization."""
    
    run_hash = generate_hash()
    
    # Create output folder
    folder_name = f"seed{seed:03d}_{run_hash}"
    output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL VALIDATION: Seed {seed}")
    print(f"RUN_HASH={run_hash}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"{'='*70}")
    
    # Compute reference values for config
    sigma_four = sigma_noise / np.sqrt(n_sensors)
    n_star_predicted = np.log(sigma_four) / np.log(rho)
    N_max_predicted = (2.0 / 3.0) * n_star_predicted
    
    # Save config (explicit type conversion for JSON compatibility)
    config = {
        'run_hash': run_hash,
        'timestamp': datetime.now().isoformat(),
        'seed': int(seed),
        'domain': 'disk',
        'rho': float(rho),
        'sigma_noise': float(sigma_noise),
        'n_sensors': int(n_sensors),
        'n_restarts': int(n_restarts),
        'N_values_tested': [int(n) for n in N_values],
        'sigma_four': float(sigma_four),
        'n_star_predicted': float(n_star_predicted),
        'N_max_predicted': float(N_max_predicted),
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")
    
    # Import and run the actual test
    from test_statistical_validation import run_single_seed_validation
    
    result = run_single_seed_validation(
        seed=seed,
        rho=rho,
        sigma_noise=sigma_noise,
        n_sensors=n_sensors,
        n_restarts=n_restarts,
        N_values=N_values
    )
    
    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved results: {results_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Seed {seed}")
    print(f"{'='*70}")
    print(f"  n*_predicted = {n_star_predicted:.2f}")
    print(f"  n*_actual    = {result['n_star_actual']}")
    print(f"  N_max_predicted = {N_max_predicted:.2f}")
    print(f"  N_max_actual    = {result['N_max_actual']:.2f}")
    print(f"  N_transition    = {result['N_transition']}")
    print(f"  Time: {result['time_seconds']:.1f}s")
    print(f"{'='*70}\n")
    
    return run_hash, output_dir, result


def main():
    parser = argparse.ArgumentParser(description="Run statistical validation experiment")
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--rho', type=float, default=0.7, help='Conformal radius')
    parser.add_argument('--sigma-noise', type=float, default=0.001, help='Noise std dev')
    parser.add_argument('--sensors', type=int, default=100, help='Number of sensors')
    parser.add_argument('--restarts', type=int, default=15, help='Optimizer restarts')
    parser.add_argument('--output-dir', type=str, default='stat_results', help='Output directory')
    
    args = parser.parse_args()
    
    # N values to test (from spec)
    N_values = [10, 12, 14, 16, 18, 20, 22, 24]
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    run_hash, output_dir, result = run_experiment(
        seed=args.seed,
        rho=args.rho,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_restarts=args.restarts,
        N_values=N_values,
        base_output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
