#!/usr/bin/env python3
"""
SLURM Job Generator for Perturbation Test (Test A)
====================================================

Generates SLURM jobs for the perturbation test (direct IFT validation).
Much faster than the RMSE test (no inverse solver), so shorter walltime.

Usage:
    python generate_perturbation_slurm.py --test-case general_random_intensity --seeds 100
    python generate_perturbation_slurm.py --all-cases --seeds 50
"""

import os
import argparse
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# JASMIN settings
ACCOUNT = "eocis_chuk"
PARTITION = "standard"
QOS = "standard"
TIME = "00:30:00"  # 30 min — no inverse solver, just forward evals + SVD
MEMORY = "8G"
CPUS = 1

# Paths
PROJECT_DIR = "/home/users/shaerdan/inverse_source_project"
CONDA_INIT = "source ~/miniforge3/bin/activate"
CONDA_ENV = "inverse_source"

# Experiment parameters
DEFAULT_SEEDS = 100
SIGMA_NOISE = 0.001
N_SENSORS = 100
N_DIRECTIONS = 200

# Random rho_min defaults
RHO_MIN_LOW = 0.5
RHO_MIN_HIGH = 0.7

# Source geometry defaults
RHO = 0.7
R_MIN = 0.5
R_MAX = 0.9
THETA_0 = 0.0

# Test cases
TEST_CASES = ['general_random_intensity']  # Primary focus for paper

FORMULAS = {
    'same_radius': 'N_max = n*',
    'same_angle': 'N_max = (1/2) n*',
    'general': 'N_max = (2/3) n*',
    'general_random_intensity': 'N_max = (2/3) n*',
}

# =============================================================================
# SLURM TEMPLATE
# =============================================================================

SLURM_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --mem={memory}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --cpus-per-task={cpus}
#SBATCH --chdir={project_dir}
#SBATCH -o {project_dir}/logs/pert_{test_case}_seed{seed:03d}_%j.out
#SBATCH -e {project_dir}/logs/pert_{test_case}_seed{seed:03d}_%j.err
#SBATCH --exclude=host1117

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Test: Perturbation (Test A)"
echo "Test Case: {test_case}"
echo "Seed: {seed}"
echo "Start: $(date)"
echo "=========================================="

# Activate conda environment
{conda_init}
conda activate {conda_env}

# Ensure directories exist
mkdir -p {project_dir}/logs
mkdir -p {project_dir}/{results_dir}

# Run experiment
cd {project_dir}/src
python run_perturbation_experiment.py \\
    --seed {seed} \\
    --test-case {test_case} \\
    --rho {rho} \\
    --r-min {r_min} \\
    --r-max {r_max} \\
    --theta-0 {theta_0} \\
    --sigma-noise {sigma_noise} \\
    --sensors {n_sensors} \\
    --n-directions {n_directions} \\
    --output-dir {project_dir}/{results_dir}{random_rho_args}

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
'''

# =============================================================================
# GENERATOR
# =============================================================================

def generate_slurm_scripts(test_cases, n_seeds, output_base_dir='slurm_perturbation',
                           random_rho_min=False, rho_min_low=0.5, rho_min_high=0.7):
    """Generate SLURM scripts for perturbation tests."""

    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Generating SLURM scripts for perturbation test (Test A)...")
    print(f"  Test cases: {test_cases}")
    print(f"  Seeds per case: {n_seeds}")
    print(f"  Total jobs: {len(test_cases) * n_seeds}")
    if random_rho_min:
        print(f"  Random rho_min: [{rho_min_low}, {rho_min_high}]")
    print()

    # Build random_rho_args string
    if random_rho_min:
        random_rho_args = (f" \\\n    --random-rho-min"
                           f" \\\n    --rho-min-low {rho_min_low}"
                           f" \\\n    --rho-min-high {rho_min_high}")
    else:
        random_rho_args = ""

    all_job_scripts = []
    jobs_by_case = {tc: [] for tc in test_cases}

    for test_case in test_cases:
        # Results directory per test case
        results_dir = f"perturbation_results_{test_case}"
        if random_rho_min:
            results_dir += "_random_rho"

        for seed in range(n_seeds):
            abbrev = {
                'same_radius': 'psr', 'same_angle': 'psa',
                'general': 'pgn', 'general_random_intensity': 'pgri',
            }.get(test_case, 'p' + test_case[:2])
            job_name = f"{abbrev}_{seed:03d}"

            script_content = SLURM_TEMPLATE.format(
                job_name=job_name,
                seed=seed,
                test_case=test_case,
                time=TIME,
                memory=MEMORY,
                account=ACCOUNT,
                partition=PARTITION,
                qos=QOS,
                cpus=CPUS,
                project_dir=PROJECT_DIR,
                conda_init=CONDA_INIT,
                conda_env=CONDA_ENV,
                results_dir=results_dir,
                rho=RHO,
                r_min=R_MIN,
                r_max=R_MAX,
                theta_0=THETA_0,
                sigma_noise=SIGMA_NOISE,
                n_sensors=N_SENSORS,
                n_directions=N_DIRECTIONS,
                random_rho_args=random_rho_args,
            )

            script_name = f"run_pert_{test_case}_seed{seed:03d}.sh"
            script_path = os.path.join(output_base_dir, script_name)

            with open(script_path, 'w') as f:
                f.write(script_content)

            all_job_scripts.append(script_name)
            jobs_by_case[test_case].append(script_name)

    print(f"  Created {len(all_job_scripts)} job scripts")

    # Create submit_all.sh
    submit_all = os.path.join(output_base_dir, "submit_all.sh")
    with open(submit_all, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all perturbation test (Test A) jobs\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total jobs: {len(all_job_scripts)}\n\n")
        f.write("echo 'Submitting all perturbation test jobs...'\n\n")
        for script in all_job_scripts:
            f.write(f"sbatch {script}\n")
        f.write("\necho 'All jobs submitted!'\n")
        f.write("echo 'Check status: squeue -u $USER'\n")
    os.chmod(submit_all, 0o755)
    print(f"  Created: submit_all.sh")

    # Create per-test-case submit scripts
    for test_case, scripts in jobs_by_case.items():
        submit_case = os.path.join(output_base_dir, f"submit_{test_case}.sh")
        with open(submit_case, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Submit perturbation test: {test_case} ({len(scripts)} jobs)\n")
            f.write(f"# Expected bound: {FORMULAS.get(test_case, 'unknown')}\n\n")
            f.write(f"echo 'Submitting {test_case} perturbation jobs...'\n\n")
            for script in scripts:
                f.write(f"sbatch {script}\n")
            f.write(f"\necho 'Submitted {len(scripts)} {test_case} perturbation jobs'\n")
        os.chmod(submit_case, 0o755)
        print(f"  Created: submit_{test_case}.sh ({len(scripts)} jobs)")

    # Quick-test scripts (first 10 seeds)
    for test_case in test_cases:
        scripts = jobs_by_case[test_case]
        batch_size = min(10, len(scripts))
        submit_test = os.path.join(output_base_dir, f"submit_{test_case}_first{batch_size}.sh")
        with open(submit_test, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Quick test: first {batch_size} seeds for {test_case}\n\n")
            for script in scripts[:batch_size]:
                f.write(f"sbatch {script}\n")
            f.write(f"\necho 'Submitted {batch_size} test jobs for {test_case}'\n")
        os.chmod(submit_test, 0o755)

    print(f"  Created quick-test scripts")

    # Job summary
    summary_path = os.path.join(output_base_dir, "job_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Perturbation Test (Test A) Job Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total jobs: {len(all_job_scripts)}\n\n")

        f.write("Test Cases:\n")
        f.write("-" * 40 + "\n")
        for tc in test_cases:
            f.write(f"  {tc}: {len(jobs_by_case[tc])} jobs\n")
            f.write(f"    Expected bound: {FORMULAS.get(tc, 'unknown')}\n")
        f.write("\n")

        f.write("Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Domain: disk\n")
        f.write(f"  sigma_noise: {SIGMA_NOISE}\n")
        f.write(f"  Sensors: {N_SENSORS}\n")
        f.write(f"  N values: [2, 4, 6, 8, 10, 12, 14, 16]\n")
        f.write(f"  Perturbation directions: {N_DIRECTIONS}\n")
        f.write(f"  Epsilons: [1e-7, 0.01, 0.05]\n")
        f.write(f"  Seeds per case: {n_seeds}\n")
        if random_rho_min:
            f.write(f"\n  RANDOM rho_min: [{rho_min_low}, {rho_min_high}]\n")
        f.write("\n")

        f.write("Output directories:\n")
        for tc in test_cases:
            suffix = "_random_rho" if random_rho_min else ""
            f.write(f"  {tc}: perturbation_results_{tc}{suffix}/\n")

        f.write("\nSubmit options:\n")
        f.write("-" * 40 + "\n")
        f.write("  All jobs:        bash submit_all.sh\n")
        for tc in test_cases:
            f.write(f"  {tc} only:  bash submit_{tc}.sh\n")

    print(f"  Created: job_summary.txt")

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"\nTo submit all {len(all_job_scripts)} jobs:")
    print(f"  cd {output_base_dir}")
    print(f"  bash submit_all.sh")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs for perturbation test (Test A)"
    )
    parser.add_argument('--all-cases', action='store_true',
                        help='Run all test cases')
    parser.add_argument('--test-case', type=str,
                        choices=['same_radius', 'same_angle', 'general',
                                 'general_random_intensity'],
                        help='Run specific test case')
    parser.add_argument('--seeds', type=int, default=DEFAULT_SEEDS,
                        help=f'Number of seeds (default: {DEFAULT_SEEDS})')
    parser.add_argument('--output-dir', type=str, default='slurm_perturbation',
                        help='Output directory for SLURM scripts')
    parser.add_argument('--random-rho-min', action='store_true',
                        help='Enable random rho_min per seed')
    parser.add_argument('--rho-min-low', type=float, default=RHO_MIN_LOW,
                        help=f'Lower bound for random rho_min (default: {RHO_MIN_LOW})')
    parser.add_argument('--rho-min-high', type=float, default=RHO_MIN_HIGH,
                        help=f'Upper bound for random rho_min (default: {RHO_MIN_HIGH})')

    args = parser.parse_args()

    if args.all_cases:
        test_cases = ['same_radius', 'same_angle', 'general', 'general_random_intensity']
    elif args.test_case:
        test_cases = [args.test_case]
    else:
        test_cases = TEST_CASES
        print(f"No test case specified, defaulting to: {test_cases}")
        print()

    generate_slurm_scripts(
        test_cases,
        args.seeds,
        args.output_dir,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
    )


if __name__ == '__main__':
    main()
