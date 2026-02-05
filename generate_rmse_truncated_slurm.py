#!/usr/bin/env python3
"""
SLURM Job Generator for Truncated RMSE Test (Test B)
======================================================

Generates SLURM jobs for the truncated RMSE test (inverse recovery
on the truncated Fourier system). Longer walltime than Test A because
each seed runs the nonlinear inverse solver for multiple N values.

Usage:
    python generate_rmse_truncated_slurm.py --test-case general_random_intensity --seeds 100
    python generate_rmse_truncated_slurm.py --all-cases --seeds 50
"""

import os
import argparse
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

ACCOUNT = "eocis_chuk"
PARTITION = "standard"
QOS = "standard"
TIME = "04:00:00"  # 4 hours — inverse solver is expensive
MEMORY = "8G"
CPUS = 1

PROJECT_DIR = "/home/users/shaerdan/inverse_source_project"
CONDA_INIT = "source ~/miniforge3/bin/activate"
CONDA_ENV = "inverse_source"

DEFAULT_SEEDS = 100
SIGMA_NOISE = 0.001
N_SENSORS = 100
N_RESTARTS = 15

RHO_MIN_LOW = 0.5
RHO_MIN_HIGH = 0.7
RHO = 0.7
R_MIN = 0.5
R_MAX = 0.9
THETA_0 = 0.0

TEST_CASES = ['general_random_intensity']

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
#SBATCH -o {project_dir}/logs/trunc_{test_case}_seed{seed:03d}_%j.out
#SBATCH -e {project_dir}/logs/trunc_{test_case}_seed{seed:03d}_%j.err
#SBATCH --exclude=host1117

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Test: Truncated RMSE (Test B)"
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
python run_rmse_truncated_experiment.py \\
    --seed {seed} \\
    --test-case {test_case} \\
    --rho {rho} \\
    --r-min {r_min} \\
    --r-max {r_max} \\
    --theta-0 {theta_0} \\
    --sigma-noise {sigma_noise} \\
    --sensors {n_sensors} \\
    --n-restarts {n_restarts} \\
    --output-dir {project_dir}/{results_dir}{random_rho_args}

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
'''


def generate_slurm_scripts(test_cases, n_seeds, output_base_dir='slurm_rmse_truncated',
                           random_rho_min=False, rho_min_low=0.5, rho_min_high=0.7):
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Generating SLURM scripts for truncated RMSE test (Test B)...")
    print(f"  Test cases: {test_cases}")
    print(f"  Seeds per case: {n_seeds}")
    print(f"  Total jobs: {len(test_cases) * n_seeds}")
    if random_rho_min:
        print(f"  Random rho_min: [{rho_min_low}, {rho_min_high}]")
    print()

    if random_rho_min:
        random_rho_args = (f" \\\n    --random-rho-min"
                           f" \\\n    --rho-min-low {rho_min_low}"
                           f" \\\n    --rho-min-high {rho_min_high}")
    else:
        random_rho_args = ""

    all_job_scripts = []
    jobs_by_case = {tc: [] for tc in test_cases}

    for test_case in test_cases:
        results_dir = f"rmse_truncated_results_{test_case}"
        if random_rho_min:
            results_dir += "_random_rho"

        for seed in range(n_seeds):
            abbrev = {
                'same_radius': 'tsr', 'same_angle': 'tsa',
                'general': 'tgn', 'general_random_intensity': 'tgri',
            }.get(test_case, 't' + test_case[:2])
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
                n_restarts=N_RESTARTS,
                random_rho_args=random_rho_args,
            )

            script_name = f"run_trunc_{test_case}_seed{seed:03d}.sh"
            script_path = os.path.join(output_base_dir, script_name)
            with open(script_path, 'w') as f:
                f.write(script_content)

            all_job_scripts.append(script_name)
            jobs_by_case[test_case].append(script_name)

    print(f"  Created {len(all_job_scripts)} job scripts")

    # submit_all.sh
    submit_all = os.path.join(output_base_dir, "submit_all.sh")
    with open(submit_all, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit all truncated RMSE test (Test B) jobs\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total jobs: {len(all_job_scripts)}\n\n")
        for script in all_job_scripts:
            f.write(f"sbatch {script}\n")
        f.write("\necho 'All jobs submitted!'\n")
    os.chmod(submit_all, 0o755)

    # Per-case submit scripts
    for test_case, scripts in jobs_by_case.items():
        submit_case = os.path.join(output_base_dir, f"submit_{test_case}.sh")
        with open(submit_case, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Submit truncated RMSE: {test_case} ({len(scripts)} jobs)\n\n")
            for script in scripts:
                f.write(f"sbatch {script}\n")
            f.write(f"\necho 'Submitted {len(scripts)} {test_case} jobs'\n")
        os.chmod(submit_case, 0o755)

    # Quick-test (first 5 seeds)
    for test_case in test_cases:
        scripts = jobs_by_case[test_case]
        batch = min(5, len(scripts))
        submit_test = os.path.join(output_base_dir, f"submit_{test_case}_first{batch}.sh")
        with open(submit_test, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Quick test: first {batch} seeds for {test_case}\n\n")
            for script in scripts[:batch]:
                f.write(f"sbatch {script}\n")
        os.chmod(submit_test, 0o755)

    # Job summary
    summary_path = os.path.join(output_base_dir, "job_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Truncated RMSE Test (Test B) Job Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total jobs: {len(all_job_scripts)}\n")
        f.write(f"Walltime per job: {TIME}\n\n")
        for tc in test_cases:
            f.write(f"  {tc}: {len(jobs_by_case[tc])} jobs\n")
            f.write(f"    Expected bound: {FORMULAS.get(tc, 'unknown')}\n")
        f.write(f"\nParameters:\n")
        f.write(f"  sigma_noise: {SIGMA_NOISE}, sensors: {N_SENSORS}\n")
        f.write(f"  n_restarts: {N_RESTARTS}\n")
        f.write(f"  Dynamic N values: ±6 around predicted N_max, step=2\n")
        if random_rho_min:
            f.write(f"  Random rho_min: [{rho_min_low}, {rho_min_high}]\n")
        f.write(f"\nSubmit: cd {output_base_dir} && bash submit_all.sh\n")

    print(f"  Created submit scripts and summary")
    print(f"\nTo submit: cd {output_base_dir} && bash submit_all.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs for truncated RMSE test (Test B)"
    )
    parser.add_argument('--all-cases', action='store_true')
    parser.add_argument('--test-case', type=str,
                        choices=['same_radius', 'same_angle', 'general',
                                 'general_random_intensity'])
    parser.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--output-dir', type=str, default='slurm_rmse_truncated')
    parser.add_argument('--random-rho-min', action='store_true')
    parser.add_argument('--rho-min-low', type=float, default=RHO_MIN_LOW)
    parser.add_argument('--rho-min-high', type=float, default=RHO_MIN_HIGH)

    args = parser.parse_args()

    if args.all_cases:
        test_cases = ['same_radius', 'same_angle', 'general', 'general_random_intensity']
    elif args.test_case:
        test_cases = [args.test_case]
    else:
        test_cases = TEST_CASES

    generate_slurm_scripts(
        test_cases, args.seeds, args.output_dir,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
    )


if __name__ == '__main__':
    main()
