#!/usr/bin/env python3
"""
Wrapper for test_statistical_validation.py (v2)
================================================

Supports all three test cases:
  - same_radius: N_max = n*
  - same_angle:  N_max = (1/2)n*
  - general:     N_max = (2/3)n*

Uses dynamic N values centered around N_predicted.

Output folder: stat_results_{test_case}/seed{seed:03d}_{hash}/
Saves: config.json, results.json

Usage:
    python run_statistical_experiment.py --seed 0 --test-case same_radius
    python run_statistical_experiment.py --seed 42 --test-case general
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


def get_formula_description(test_case):
    """Return formula description for each test case."""
    formulas = {
        'same_radius': 'N_max = n*',
        'same_angle': 'N_max = (1/2) n*',
        'general': 'N_max = (2/3) n*',
    }
    return formulas.get(test_case, 'unknown')


def run_experiment(seed, test_case, rho, r_min, r_max, theta_0, 
                   sigma_noise, n_sensors, n_restarts, base_output_dir):
    """Run statistical validation for one seed with output organization."""
    
    run_hash = generate_hash()
    
    # Create output folder
    folder_name = f"seed{seed:03d}_{run_hash}"
    output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL VALIDATION: Seed {seed}")
    print(f"TEST_CASE={test_case}")
    print(f"RUN_HASH={run_hash}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"{'='*70}")
    
    # Determine rho_min for config
    if test_case == 'same_radius':
        rho_min = rho
    else:
        rho_min = r_min
    
    # Compute reference values
    sigma_four = sigma_noise / np.sqrt(n_sensors)
    n_star_sigma_four = np.log(sigma_four) / np.log(rho_min)
    
    # N_max formula depends on test case
    if test_case == 'same_radius':
        N_max_sigma_four = n_star_sigma_four  # N_max = n*
    elif test_case == 'same_angle':
        N_max_sigma_four = 0.5 * n_star_sigma_four  # N_max = (1/2)n*
    else:  # general
        N_max_sigma_four = (2.0 / 3.0) * n_star_sigma_four  # N_max = (2/3)n*
    
    # Save config (explicit type conversion for JSON compatibility)
    config = {
        'run_hash': run_hash,
        'timestamp': datetime.now().isoformat(),
        'seed': int(seed),
        'test_case': test_case,
        'formula': get_formula_description(test_case),
        'domain': 'disk',
        'rho': float(rho),
        'r_min': float(r_min),
        'r_max': float(r_max),
        'theta_0': float(theta_0),
        'rho_min': float(rho_min),
        'sigma_noise': float(sigma_noise),
        'n_sensors': int(n_sensors),
        'n_restarts': int(n_restarts),
        'N_values_tested': 'dynamic',  # Will be updated after run
        'sigma_four': float(sigma_four),
        'n_star_sigma_four': float(n_star_sigma_four),
        'N_max_sigma_four': float(N_max_sigma_four),
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")
    
    # Import and run the actual test
    from test_statistical_validation import run_single_seed_validation
    
    result = run_single_seed_validation(
        seed=seed,
        test_case=test_case,
        rho=rho,
        r_min=r_min,
        r_max=r_max,
        theta_0=theta_0,
        sigma_noise=sigma_noise,
        n_sensors=n_sensors,
        n_restarts=n_restarts,
        use_dynamic_N=True  # Use dynamic N values centered around prediction
    )
    
    # Update config with actual N values tested
    config['N_values_tested'] = result.get('N_values_tested', [])
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved results: {results_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Seed {seed}, Test Case: {test_case}")
    print(f"{'='*70}")
    print(f"  Formula: {get_formula_description(test_case)}")
    print(f"  rho_min = {rho_min:.3f}")
    print(f"  n*_actual (max usable mode) = {result['n_star_max']}")
    print(f"  N_max_predicted = {result['N_max_predicted']:.2f}")
    print(f"  N_transition = {result['N_transition']}")
    print(f"  Error = {result['N_transition'] - result['N_max_predicted']:.2f}")
    print(f"  N values tested: {result.get('N_values_tested', 'N/A')}")
    print(f"  Time: {result['time_seconds']:.1f}s")
    print(f"{'='*70}\n")
    
    return run_hash, output_dir, result


def main():
    parser = argparse.ArgumentParser(description="Run statistical validation experiment")
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--test-case', type=str, default='same_radius',
                       choices=['same_radius', 'same_angle', 'general'],
                       help='Test case type')
    parser.add_argument('--rho', type=float, default=0.7, 
                       help='Common radius for same_radius case')
    parser.add_argument('--r-min', type=float, default=0.5, 
                       help='Min radius for same_angle and general cases')
    parser.add_argument('--r-max', type=float, default=0.9, 
                       help='Max radius for same_angle and general cases')
    parser.add_argument('--theta-0', type=float, default=0.0, 
                       help='Common angle for same_angle case')
    parser.add_argument('--sigma-noise', type=float, default=0.001, 
                       help='Noise std dev')
    parser.add_argument('--sensors', type=int, default=100, 
                       help='Number of sensors')
    parser.add_argument('--restarts', type=int, default=15, 
                       help='Optimizer restarts')
    parser.add_argument('--output-dir', type=str, default='stat_results', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    run_hash, output_dir, result = run_experiment(
        seed=args.seed,
        test_case=args.test_case,
        rho=args.rho,
        r_min=args.r_min,
        r_max=args.r_max,
        theta_0=args.theta_0,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_restarts=args.restarts,
        base_output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
