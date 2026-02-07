#!/usr/bin/env python3
"""
Wrapper for test_rmse_truncated_conformal.py
=============================================

Runs the truncated RMSE test for conformal domains (ellipse, brain)
for one seed with output organisation.

Usage:
    python run_rmse_truncated_conformal.py --seed 0 --domain ellipse
    python run_rmse_truncated_conformal.py --seed 42 --domain brain --random-rho-min
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
    data = f"{datetime.now().isoformat()}{os.urandom(8)}".encode()
    return hashlib.sha256(data).hexdigest()[:length]


FORMULAS = {
    'same_radius': 'N_max = n*',
    'same_angle': 'N_max = (1/2) n*',
    'general': 'N_max = (2/3) n*',
    'general_random_intensity': 'N_max = (2/3) n*',
}


def run_experiment(seed, domain, test_case, r_min, r_max,
                   sigma_noise, n_sensors, n_restarts, base_output_dir,
                   random_rho_min=False, rho_min_low=0.5, rho_min_high=0.7,
                   intensity_low=0.5, intensity_high=2.0,
                   ellipse_a=1.5, ellipse_b=0.8):
    
    run_hash = generate_hash()
    folder_name = f"seed{seed:03d}_{run_hash}"
    output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"TRUNCATED RMSE TEST (Conformal): Seed {seed}, Domain: {domain}")
    print(f"TEST_CASE={test_case}, RUN_HASH={run_hash}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"{'='*70}")
    
    # Reference values for config
    if random_rho_min:
        rho_min_ref = (rho_min_low + rho_min_high) / 2
    else:
        rho_min_ref = r_min
    
    sigma_four = sigma_noise / np.sqrt(n_sensors)
    n_star_sigma_four = np.log(sigma_four) / np.log(rho_min_ref)
    
    config = {
        'test_type': 'rmse_truncated_conformal',
        'run_hash': run_hash,
        'timestamp': datetime.now().isoformat(),
        'seed': int(seed),
        'domain': domain,
        'test_case': test_case,
        'formula': FORMULAS.get(test_case, 'unknown'),
        'r_min': float(r_min),
        'r_max': float(r_max),
        'random_rho_min': random_rho_min,
        'rho_min_low': float(rho_min_low) if random_rho_min else None,
        'rho_min_high': float(rho_min_high) if random_rho_min else None,
        'sigma_noise': float(sigma_noise),
        'n_sensors': int(n_sensors),
        'n_restarts': int(n_restarts),
        'sigma_four': float(sigma_four),
        'n_star_sigma_four': float(n_star_sigma_four),
        'ellipse_a': float(ellipse_a) if domain == 'ellipse' else None,
        'ellipse_b': float(ellipse_b) if domain == 'ellipse' else None,
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")
    
    # Import and run
    from test_rmse_truncated_conformal import run_single_seed_conformal
    
    result = run_single_seed_conformal(
        seed=seed,
        domain=domain,
        test_case=test_case,
        r_min=r_min,
        r_max=r_max,
        sigma_noise=sigma_noise,
        n_sensors=n_sensors,
        n_restarts=n_restarts,
        random_rho_min=random_rho_min,
        rho_min_low=rho_min_low,
        rho_min_high=rho_min_high,
        intensity_low=intensity_low,
        intensity_high=intensity_high,
        ellipse_a=ellipse_a,
        ellipse_b=ellipse_b,
    )
    
    # Update config with actual values
    config['rho_min_actual'] = result.get('rho_min')
    config['n_star_actual'] = result.get('n_star')
    config['N_max_actual'] = result.get('N_max')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved results: {results_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Seed {seed}, Domain: {domain}, Test Case: {test_case}")
    print(f"  n* = {result['n_star']}, N_max = {result['N_max']:.2f}")
    print(f"  Time: {result['time_seconds']:.1f}s")
    for nr in result['results_by_N']:
        status = "OK" if nr['rmse_position'] < 0.1 else "FAIL"
        print(f"    N={nr['N']:2d}: pos={nr['rmse_position']:.4f} "
              f"int={nr['rmse_intensity']:.4f} tot={nr['rmse_total']:.4f} [{status}]")
    print(f"{'='*70}\n")
    
    return run_hash, output_dir, result


def main():
    parser = argparse.ArgumentParser(
        description="Run truncated RMSE test for conformal domain"
    )
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--domain', type=str, required=True,
                        choices=['ellipse', 'brain'])
    parser.add_argument('--test-case', type=str, default='general_random_intensity',
                        choices=['same_radius', 'same_angle', 'general',
                                 'general_random_intensity'])
    parser.add_argument('--r-min', type=float, default=0.5)
    parser.add_argument('--r-max', type=float, default=0.7)
    parser.add_argument('--sigma-noise', type=float, default=0.001)
    parser.add_argument('--sensors', type=int, default=100)
    parser.add_argument('--n-restarts', type=int, default=15)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory (auto-generated if not specified)')
    parser.add_argument('--random-rho-min', action='store_true')
    parser.add_argument('--rho-min-low', type=float, default=0.5)
    parser.add_argument('--rho-min-high', type=float, default=0.7)
    parser.add_argument('--intensity-low', type=float, default=0.5)
    parser.add_argument('--intensity-high', type=float, default=2.0)
    parser.add_argument('--ellipse-a', type=float, default=1.5)
    parser.add_argument('--ellipse-b', type=float, default=0.8)
    
    args = parser.parse_args()
    
    # Auto-generate output dir if not specified
    if args.output_dir is None:
        suffix = '_random_rho' if args.random_rho_min else ''
        args.output_dir = f"rmse_truncated_results_{args.domain}_{args.test_case}{suffix}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_experiment(
        seed=args.seed,
        domain=args.domain,
        test_case=args.test_case,
        r_min=args.r_min,
        r_max=args.r_max,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_restarts=args.n_restarts,
        base_output_dir=args.output_dir,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
        intensity_low=args.intensity_low,
        intensity_high=args.intensity_high,
        ellipse_a=args.ellipse_a,
        ellipse_b=args.ellipse_b,
    )


if __name__ == '__main__':
    main()
