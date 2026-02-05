#!/usr/bin/env python3
"""
Wrapper for test_perturbation.py (Test A)
==========================================

Runs the perturbation test for one seed with output organisation.

Output folder: perturbation_results_{test_case}/seed{seed:03d}_{hash}/
Saves: config.json, results.json

Usage:
    python run_perturbation_experiment.py --seed 0 --test-case general_random_intensity
    python run_perturbation_experiment.py --seed 42 --test-case general --random-rho-min
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
        'general_random_intensity': 'N_max = (2/3) n*',
    }
    return formulas.get(test_case, 'unknown')


def run_experiment(seed, test_case, rho, r_min, r_max, theta_0,
                   sigma_noise, n_sensors, n_directions, base_output_dir,
                   random_rho_min=False, rho_min_low=0.5, rho_min_high=0.7,
                   intensity_low=0.5, intensity_high=2.0):
    """Run perturbation test for one seed with output organisation."""

    run_hash = generate_hash()

    # Create output folder
    folder_name = f"seed{seed:03d}_{run_hash}"
    output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"PERTURBATION TEST (Test A): Seed {seed}")
    print(f"TEST_CASE={test_case}")
    if random_rho_min:
        print(f"RANDOM_RHO_MIN=[{rho_min_low}, {rho_min_high}]")
    print(f"RUN_HASH={run_hash}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"{'='*70}")

    # Determine rho_min for config
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

    # Compute reference values
    sigma_four = sigma_noise / np.sqrt(n_sensors)
    n_star_sigma_four = np.log(sigma_four) / np.log(rho_min_ref)

    # Save config
    config = {
        'test_type': 'perturbation',
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
        'rho_min': rho_min_config if isinstance(rho_min_config, str) else float(rho_min_config),
        'random_rho_min': random_rho_min,
        'rho_min_low': float(rho_min_low) if random_rho_min else None,
        'rho_min_high': float(rho_min_high) if random_rho_min else None,
        'sigma_noise': float(sigma_noise),
        'n_sensors': int(n_sensors),
        'n_directions': int(n_directions),
        'N_values': [2, 4, 6, 8, 10, 12, 14, 16],
        'sigma_four': float(sigma_four),
        'n_star_sigma_four': float(n_star_sigma_four),
    }

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    # Import and run the actual test
    from test_perturbation import run_single_seed_perturbation

    result = run_single_seed_perturbation(
        seed=seed,
        test_case=test_case,
        rho=rho,
        r_min=r_min,
        r_max=r_max,
        theta_0=theta_0,
        sigma_noise=sigma_noise,
        n_sensors=n_sensors,
        n_directions=n_directions,
        random_rho_min=random_rho_min,
        rho_min_low=rho_min_low,
        rho_min_high=rho_min_high,
        intensity_low=intensity_low,
        intensity_high=intensity_high,
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

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Seed {seed}, Test Case: {test_case}")
    print(f"{'='*70}")
    print(f"  Formula: {get_formula_description(test_case)}")
    print(f"  rho_min = {result['rho_min']:.4f}" + (" (random)" if random_rho_min else ""))
    print(f"  n* = {result['n_star']}")
    print(f"  N_max = {result['N_max']:.2f}")
    print(f"  Time: {result['time_seconds']:.1f}s")
    print(f"{'='*70}\n")

    return run_hash, output_dir, result


def main():
    parser = argparse.ArgumentParser(
        description="Run perturbation test (Test A) for one seed"
    )
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--test-case', type=str, default='general_random_intensity',
                        choices=['same_radius', 'same_angle', 'general',
                                 'general_random_intensity'],
                        help='Test case type')
    parser.add_argument('--rho', type=float, default=0.7,
                        help='Common radius for same_radius case')
    parser.add_argument('--r-min', type=float, default=0.5,
                        help='Min radius')
    parser.add_argument('--r-max', type=float, default=0.9,
                        help='Max radius')
    parser.add_argument('--theta-0', type=float, default=0.0,
                        help='Common angle for same_angle case')
    parser.add_argument('--sigma-noise', type=float, default=0.001,
                        help='Noise std dev')
    parser.add_argument('--sensors', type=int, default=100,
                        help='Number of sensors')
    parser.add_argument('--n-directions', type=int, default=200,
                        help='Number of perturbation directions')
    parser.add_argument('--output-dir', type=str, default='perturbation_results',
                        help='Output directory')
    parser.add_argument('--random-rho-min', action='store_true',
                        help='Randomly sample rho_min per seed')
    parser.add_argument('--rho-min-low', type=float, default=0.5,
                        help='Lower bound for random rho_min')
    parser.add_argument('--rho-min-high', type=float, default=0.7,
                        help='Upper bound for random rho_min')
    parser.add_argument('--intensity-low', type=float, default=0.5,
                        help='Min intensity magnitude')
    parser.add_argument('--intensity-high', type=float, default=2.0,
                        help='Max intensity magnitude')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_hash, output_dir, result = run_experiment(
        seed=args.seed,
        test_case=args.test_case,
        rho=args.rho,
        r_min=args.r_min,
        r_max=args.r_max,
        theta_0=args.theta_0,
        sigma_noise=args.sigma_noise,
        n_sensors=args.sensors,
        n_directions=args.n_directions,
        base_output_dir=args.output_dir,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
        intensity_low=args.intensity_low,
        intensity_high=args.intensity_high,
    )


if __name__ == '__main__':
    main()
