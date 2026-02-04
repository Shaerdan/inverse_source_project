#!/usr/bin/env python3
"""
SLURM Job Generator for Source Configuration Experiments (v2)
==============================================================

Generates SLURM jobs for testing different source configurations:
  - same_radius: All sources at same r, random angles → N_max = n*
  - same_angle: All sources at same θ, random radii → N_max = (1/2)n*
  - general: Random radii and angles → N_max = (2/3)n*

Can run all test cases or select specific ones.

Usage:
    # Generate for all test cases (50 seeds each)
    python generate_statistical_slurm.py --all-cases

    # Generate for specific test case
    python generate_statistical_slurm.py --test-case same_radius --seeds 100

    # Quick test (10 seeds each)
    python generate_statistical_slurm.py --all-cases --seeds 10
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
TIME = "02:00:00"  # 2 hours for dynamic N values
MEMORY = "16G"
CPUS = 1

# Paths
PROJECT_DIR = "/home/users/shaerdan/inverse_source_project"
CONDA_INIT = "source ~/miniforge3/bin/activate"
CONDA_ENV = "inverse_source"

# Experiment parameters
DEFAULT_SEEDS = 50  # Default number of seeds per test case
RHO = 0.7           # Common radius for same_radius case
R_MIN = 0.5         # Min radius for same_angle and general cases
R_MAX = 0.9         # Max radius for same_angle and general cases
THETA_0 = 0.0       # Common angle for same_angle case
SIGMA_NOISE = 0.001
N_SENSORS = 100
N_RESTARTS = 15

# Test cases
TEST_CASES = ['same_radius', 'same_angle', 'general']

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
#SBATCH -o {project_dir}/logs/{test_case}_seed{seed:03d}_%j.out
#SBATCH -e {project_dir}/logs/{test_case}_seed{seed:03d}_%j.err

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
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
python run_statistical_experiment.py \\
    --seed {seed} \\
    --test-case {test_case} \\
    --rho {rho} \\
    --r-min {r_min} \\
    --r-max {r_max} \\
    --theta-0 {theta_0} \\
    --sigma-noise {sigma_noise} \\
    --sensors {n_sensors} \\
    --restarts {n_restarts} \\
    --output-dir {project_dir}/{results_dir}

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
'''

# =============================================================================
# GENERATOR
# =============================================================================

def generate_slurm_scripts(test_cases, n_seeds, output_base_dir='slurm_statistical'):
    """Generate SLURM scripts for specified test cases."""
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Generating SLURM scripts for source configuration experiments...")
    print(f"  Test cases: {test_cases}")
    print(f"  Seeds per case: {n_seeds}")
    print(f"  Total jobs: {len(test_cases) * n_seeds}")
    print()
    
    all_job_scripts = []
    jobs_by_case = {tc: [] for tc in test_cases}
    
    for test_case in test_cases:
        # Results directory per test case
        results_dir = f"stat_results_{test_case}"
        
        for seed in range(n_seeds):
            job_name = f"{test_case[:3]}_{seed:03d}"  # e.g., "sam_000", "gen_042"
            
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
            )
            
            script_name = f"run_{test_case}_seed{seed:03d}.sh"
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
        f.write("# Submit all source configuration experiment jobs\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total jobs: {len(all_job_scripts)}\n\n")
        
        f.write("echo 'Submitting all jobs...'\n\n")
        
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
            f.write(f"# Submit {test_case} jobs ({len(scripts)} total)\n")
            f.write(f"# Expected N_max formula: {get_formula_for_case(test_case)}\n\n")
            
            f.write(f"echo 'Submitting {test_case} jobs...'\n\n")
            
            for script in scripts:
                f.write(f"sbatch {script}\n")
            
            f.write(f"\necho 'Submitted {len(scripts)} {test_case} jobs'\n")
        
        os.chmod(submit_case, 0o755)
        print(f"  Created: submit_{test_case}.sh ({len(scripts)} jobs)")
    
    # Create batch scripts for testing (10 seeds each)
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
    
    print(f"  Created quick-test scripts (submit_<case>_first10.sh)")
    
    # Create job summary
    summary_path = os.path.join(output_base_dir, "job_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Source Configuration Experiment Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total jobs: {len(all_job_scripts)}\n\n")
        
        f.write("Test Cases:\n")
        f.write("-" * 40 + "\n")
        for tc in test_cases:
            formula = get_formula_for_case(tc)
            f.write(f"  {tc}: {len(jobs_by_case[tc])} jobs\n")
            f.write(f"    Expected bound: {formula}\n")
        f.write("\n")
        
        f.write("Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Domain: disk\n")
        f.write(f"  sigma_noise: {SIGMA_NOISE}\n")
        f.write(f"  Sensors: {N_SENSORS}\n")
        f.write(f"  Restarts: {N_RESTARTS}\n")
        f.write(f"  Seeds per case: {n_seeds}\n\n")
        
        f.write("  same_radius config:\n")
        f.write(f"    rho = {RHO}\n")
        f.write(f"    rho_min = {RHO}\n\n")
        
        f.write("  same_angle config:\n")
        f.write(f"    theta_0 = {THETA_0}\n")
        f.write(f"    r_min = {R_MIN}, r_max = {R_MAX}\n")
        f.write(f"    rho_min = {R_MIN}\n\n")
        
        f.write("  general config:\n")
        f.write(f"    r_min = {R_MIN}, r_max = {R_MAX}\n")
        f.write(f"    rho_min = {R_MIN}\n\n")
        
        f.write("N values: DYNAMIC\n")
        f.write("  Centered around N_predicted with range ±6, step 2\n")
        f.write("  N = [N_pred-6, N_pred-4, N_pred-2, N_pred, N_pred+2, N_pred+4, N_pred+6]\n\n")
        
        f.write("Output directories:\n")
        for tc in test_cases:
            f.write(f"  {tc}: stat_results_{tc}/\n")
        
        f.write("\nSubmit options:\n")
        f.write("-" * 40 + "\n")
        f.write("  All jobs:        bash submit_all.sh\n")
        for tc in test_cases:
            f.write(f"  {tc} only:  bash submit_{tc}.sh\n")
        f.write("\nQuick tests (10 seeds):\n")
        for tc in test_cases:
            f.write(f"  bash submit_{tc}_first10.sh\n")
    
    print(f"  Created: job_summary.txt")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"\nTo submit all {len(all_job_scripts)} jobs:")
    print(f"  cd {output_base_dir}")
    print(f"  bash submit_all.sh")
    print(f"\nOr submit by test case:")
    for tc in test_cases:
        print(f"  bash submit_{tc}.sh  ({len(jobs_by_case[tc])} jobs)")
    print(f"\nQuick test (10 seeds each):")
    for tc in test_cases:
        print(f"  bash submit_{tc}_first10.sh")


def get_formula_for_case(test_case):
    """Return the expected N_max formula for each test case."""
    formulas = {
        'same_radius': 'N_max = n*',
        'same_angle': 'N_max = (1/2) n*',
        'general': 'N_max = (2/3) n*',
    }
    return formulas.get(test_case, 'unknown')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate SLURM jobs for source config experiments")
    parser.add_argument('--all-cases', action='store_true',
                       help='Run all 3 test cases')
    parser.add_argument('--test-case', type=str, choices=TEST_CASES,
                       help='Run specific test case')
    parser.add_argument('--seeds', type=int, default=DEFAULT_SEEDS,
                       help=f'Number of seeds per test case (default: {DEFAULT_SEEDS})')
    parser.add_argument('--output-dir', type=str, default='slurm_statistical',
                       help='Output directory for SLURM scripts')
    
    args = parser.parse_args()
    
    # Determine which test cases to run
    if args.all_cases:
        test_cases = TEST_CASES
    elif args.test_case:
        test_cases = [args.test_case]
    else:
        # Default: all cases
        test_cases = TEST_CASES
        print("No test case specified, defaulting to all cases.")
        print("Use --test-case <case> to run a specific case.")
        print()
    
    generate_slurm_scripts(test_cases, args.seeds, args.output_dir)


if __name__ == '__main__':
    main()
