#!/usr/bin/env python3
"""
Wrapper for test_statistical_validation_multidomain.py
=======================================================

Extends run_statistical_experiment.py with --domain flag.
Supports: disk, ellipse, brain

Output folder: stat_results_{domain}_{test_case}[_random_rho]/seed{seed:03d}_{hash}/
Saves: config.json, results.json

Usage:
    python run_statistical_experiment_multidomain.py --seed 0 --test-case same_radius --domain disk
    python run_statistical_experiment_multidomain.py --seed 0 --test-case general --domain ellipse
    python run_statistical_experiment_multidomain.py --seed 0 --test-case same_radius --domain brain
"""

import os
import sys
import json
import hashlib
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SUPPORTED_DOMAINS = ['disk', 'ellipse', 'brain']


def generate_hash(length=8):
    """Generate unique run hash."""
    data = f"{datetime.now().isoformat()}{os.urandom(8)}".encode()
    return hashlib.sha256(data).hexdigest()[:length]


def get_formula_description(test_case):
    formulas = {
        'same_radius': 'N_max = n*',
        'same_angle': 'N_max = (1/2) n*',
        'general': 'N_max = (2/3) n*',
        'general_random_intensity': 'N_max = (2/3) n*',
    }
    return formulas.get(test_case, 'unknown')


def run_experiment(seed, test_case, domain, rho, r_min, r_max, theta_0,
                   sigma_noise, n_sensors, n_restarts, base_output_dir,
                   random_rho_min=False, rho_min_low=0.5, rho_min_high=0.6):
    """Run statistical validation for one seed on specified domain."""

    run_hash = generate_hash()

    folder_name = f"seed{seed:03d}_{run_hash}"
    output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"STATISTICAL VALIDATION (MULTI-DOMAIN): Seed {seed}")
    print(f"DOMAIN={domain}  TEST_CASE={test_case}")
    if random_rho_min:
        print(f"RANDOM_RHO_MIN=[{rho_min_low}, {rho_min_high}]")
    print(f"RUN_HASH={run_hash}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"{'='*70}")

    # Reference rho_min for config
    if random_rho_min:
        rho_min_config = f"random [{rho_min_low}, {rho_min_high}]"
        rho_min_ref = (rho_min_low + rho_min_high) / 2
    else:
        if test_case == 'same_radius':
            rho_min_config = rho
            rho_min_ref = rho
        else:
            rho_min_config = r_min
            rho_min_ref = r_min

    sigma_four = sigma_noise / np.sqrt(n_sensors)
    n_star_sigma_four = np.log(sigma_four) / np.log(rho_min_ref)

    if test_case == 'same_radius':
        N_max_sigma_four = n_star_sigma_four
    elif test_case == 'same_angle':
        N_max_sigma_four = 0.5 * n_star_sigma_four
    else:
        N_max_sigma_four = (2.0 / 3.0) * n_star_sigma_four

    config = {
        'run_hash': run_hash,
        'timestamp': datetime.now().isoformat(),
        'seed': int(seed),
        'test_case': test_case,
        'domain': domain,
        'formula': get_formula_description(test_case),
        'rho': float(rho),
        'r_min': float(r_min),
        'r_max': float(r_max),
        'theta_0': float(theta_0),
        'rho_min': rho_min_config if isinstance(rho_min_config, str) else float(rho_min_config),
        'random_rho_min': random_rho_min,
        'rho_min_low': float(rho_min_low) if random_rho_min else None,
        'rho_min_high': float(rho_min_high) if random_rho_min else None,
        'sigma_noise': float(sigma_noise),
        'n_sensors': int(n_sensors),
        'n_restarts': int(n_restarts),
        'N_values_tested': 'dynamic',
        'sigma_four': float(sigma_four),
        'n_star_sigma_four': float(n_star_sigma_four),
        'N_max_sigma_four': float(N_max_sigma_four),
    }

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    # Import and run — uses same solver flow as test_bound_theory.py
    from test_statistical_validation_multidomain import run_single_seed_validation

    result = run_single_seed_validation(
        seed=seed,
        test_case=test_case,
        domain=domain,
        rho=rho,
        r_min=r_min,
        r_max=r_max,
        theta_0=theta_0,
        sigma_noise=sigma_noise,
        n_sensors=n_sensors,
        n_restarts=n_restarts,
        use_dynamic_N=True,
        random_rho_min=random_rho_min,
        rho_min_low=rho_min_low,
        rho_min_high=rho_min_high
    )

    # Update config
    config['N_values_tested'] = result.get('N_values_tested', [])
    config['rho_min_actual'] = result.get('rho_min')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved results: {results_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Seed {seed}, Domain: {domain}, Test Case: {test_case}")
    print(f"{'='*70}")
    print(f"  Formula: {get_formula_description(test_case)}")
    print(f"  rho_min = {result['rho_min']:.4f}" + (" (random)" if random_rho_min else ""))
    print(f"  n*_actual (max usable mode) = {result['n_star_max']}")
    print(f"  N_max_predicted = {result['N_max_predicted']:.2f}")
    print(f"  N values tested: {result.get('N_values_tested', 'N/A')}")
    print(f"  Time: {result['time_seconds']:.1f}s")
    print(f"{'='*70}\n")

    return run_hash, output_dir, result


def main():
    parser = argparse.ArgumentParser(description="Run multi-domain statistical validation experiment")
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--test-case', type=str, default='same_radius',
                       choices=['same_radius', 'same_angle', 'general', 'general_random_intensity'])
    parser.add_argument('--domain', type=str, default='disk',
                       choices=SUPPORTED_DOMAINS,
                       help='Domain geometry (default: disk)')
    parser.add_argument('--rho', type=float, default=0.7)
    parser.add_argument('--r-min', type=float, default=0.5)
    parser.add_argument('--r-max', type=float, default=0.9)
    parser.add_argument('--theta-0', type=float, default=0.0)
    parser.add_argument('--sigma-noise', type=float, default=0.001)
    parser.add_argument('--sensors', type=int, default=100)
    parser.add_argument('--restarts', type=int, default=15)
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--random-rho-min', action='store_true')
    parser.add_argument('--rho-min-low', type=float, default=0.5)
    parser.add_argument('--rho-min-high', type=float, default=0.6)

    args = parser.parse_args()

    # Auto-generate output dir if not specified
    if args.output_dir is None:
        suffix = "_random_rho" if args.random_rho_min else ""
        args.output_dir = f"stat_results_{args.domain}_{args.test_case}{suffix}"

    os.makedirs(args.output_dir, exist_ok=True)

    run_experiment(
        seed=args.seed,
        test_case=args.test_case,
        domain=args.domain,
        rho=args.rho,
        r_min=args.r_min,
        r_max=args.r_max,
        theta_0=args.theta_0,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_restarts=args.restarts,
        base_output_dir=args.output_dir,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high
    )


if __name__ == '__main__':
    main()
