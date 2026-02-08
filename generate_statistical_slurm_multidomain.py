#!/usr/bin/env python3
"""
SLURM Job Generator for Full-System Statistical Test - Multi-Domain
====================================================================

Generates SLURM jobs for the FULL SYSTEM (not truncated) statistical 
validation test across all domains: disk, ellipse, brain.

This is the full-system counterpart to the truncated tests. 
Uses AnalyticalNonlinearInverseSolver for disk, ConformalNonlinearInverseSolver
for ellipse/brain.

Objective: min ||u_est - u_measured||^2 (full boundary potential)
vs truncated: min ||h_trunc(s) - d_trunc||^2 (first n* Fourier modes)

Usage:
    # All domains with general_random_intensity (to match truncated tests)
    python generate_statistical_slurm_multidomain.py --all-domains --seeds 100 --random-rho-min --rho-min-high 0.7

    # Single domain
    python generate_statistical_slurm_multidomain.py --domain ellipse --seeds 100 --random-rho-min

    # All test cases for one domain
    python generate_statistical_slurm_multidomain.py --domain disk --all-cases --seeds 50
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
TIME = "04:00:00"  # 4 hours — full system is faster than truncated
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
RHO_MIN_HIGH = 0.7  # Match truncated tests
R_MIN = 0.5
R_MAX = 0.7  # Match truncated tests

DEFAULT_TEST_CASE = 'general_random_intensity'
TEST_CASES = ['same_radius', 'same_angle', 'general', 'general_random_intensity']
DOMAINS = ['disk', 'ellipse', 'brain']

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
#SBATCH -o {project_dir}/logs/full_{domain}_{test_case}_seed{seed:03d}_%j.out
#SBATCH -e {project_dir}/logs/full_{domain}_{test_case}_seed{seed:03d}_%j.err
#SBATCH --exclude=host1117

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Test: FULL SYSTEM Statistical Validation"
echo "Domain: {domain}"
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
python run_statistical_experiment_multidomain.py \\
    --seed {seed} \\
    --domain {domain} \\
    --test-case {test_case} \\
    --r-min {r_min} \\
    --r-max {r_max} \\
    --sigma-noise {sigma_noise} \\
    --sensors {n_sensors} \\
    --restarts {n_restarts} \\
    --output-dir {project_dir}/{results_dir}{random_rho_args}

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
'''


def generate_slurm_scripts(domains, test_cases, n_seeds, 
                           output_base_dir='slurm_statistical_multidomain',
                           random_rho_min=False, rho_min_low=0.5, rho_min_high=0.7):
    """Generate SLURM scripts for full-system statistical tests."""
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    total_jobs = len(domains) * len(test_cases) * n_seeds
    
    print(f"Generating SLURM scripts for FULL SYSTEM statistical test...")
    print(f"  Domains: {domains}")
    print(f"  Test cases: {test_cases}")
    print(f"  Seeds per (domain, test_case): {n_seeds}")
    print(f"  Total jobs: {total_jobs}")
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
    jobs_by_domain = {d: [] for d in domains}
    jobs_by_case = {c: [] for c in test_cases}
    jobs_by_domain_case = {(d, c): [] for d in domains for c in test_cases}
    
    for domain in domains:
        for test_case in test_cases:
            results_dir = f"stat_results_{domain}_{test_case}"
            if random_rho_min:
                results_dir += "_random_rho"
            
            for seed in range(n_seeds):
                # Job name abbreviations
                domain_abbrev = {'disk': 'fdk', 'ellipse': 'fel', 'brain': 'fbr'}[domain]
                case_abbrev = {'same_radius': 'sr', 'same_angle': 'sa', 
                               'general': 'gn', 'general_random_intensity': 'gri'}[test_case]
                job_name = f"{domain_abbrev}_{case_abbrev}_{seed:03d}"
                
                script_content = SLURM_TEMPLATE.format(
                    job_name=job_name,
                    seed=seed,
                    domain=domain,
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
                    r_min=R_MIN,
                    r_max=R_MAX,
                    sigma_noise=SIGMA_NOISE,
                    n_sensors=N_SENSORS,
                    n_restarts=N_RESTARTS,
                    random_rho_args=random_rho_args,
                )
                
                script_name = f"run_full_{domain}_{test_case}_seed{seed:03d}.sh"
                script_path = os.path.join(output_base_dir, script_name)
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                all_job_scripts.append(script_name)
                jobs_by_domain[domain].append(script_name)
                jobs_by_case[test_case].append(script_name)
                jobs_by_domain_case[(domain, test_case)].append(script_name)
    
    print(f"  Created {len(all_job_scripts)} job scripts")
    
    # submit_all.sh
    submit_all = os.path.join(output_base_dir, "submit_all.sh")
    with open(submit_all, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit ALL full-system statistical jobs\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total jobs: {len(all_job_scripts)}\n\n")
        for script in all_job_scripts:
            f.write(f"sbatch {script}\n")
        f.write("\necho 'All jobs submitted!'\n")
    os.chmod(submit_all, 0o755)
    
    # Per-domain submit scripts
    for domain, scripts in jobs_by_domain.items():
        if scripts:
            submit_domain = os.path.join(output_base_dir, f"submit_{domain}.sh")
            with open(submit_domain, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"# Submit full-system: {domain} ({len(scripts)} jobs)\n\n")
                for script in scripts:
                    f.write(f"sbatch {script}\n")
                f.write(f"\necho 'Submitted {len(scripts)} {domain} jobs'\n")
            os.chmod(submit_domain, 0o755)
    
    # Per-test-case submit scripts
    for test_case, scripts in jobs_by_case.items():
        if scripts:
            submit_case = os.path.join(output_base_dir, f"submit_{test_case}.sh")
            with open(submit_case, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"# Submit full-system: {test_case} ({len(scripts)} jobs)\n\n")
                for script in scripts:
                    f.write(f"sbatch {script}\n")
                f.write(f"\necho 'Submitted {len(scripts)} {test_case} jobs'\n")
            os.chmod(submit_case, 0o755)
    
    # Per (domain, test_case) submit scripts
    for (domain, test_case), scripts in jobs_by_domain_case.items():
        if scripts:
            submit_dt = os.path.join(output_base_dir, f"submit_{domain}_{test_case}.sh")
            with open(submit_dt, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"# Submit full-system: {domain} + {test_case} ({len(scripts)} jobs)\n\n")
                for script in scripts:
                    f.write(f"sbatch {script}\n")
                f.write(f"\necho 'Submitted {len(scripts)} jobs'\n")
            os.chmod(submit_dt, 0o755)
    
    # Quick-test scripts (first 5 seeds)
    for domain in domains:
        for test_case in test_cases:
            scripts = jobs_by_domain_case[(domain, test_case)]
            if scripts:
                batch = min(5, len(scripts))
                submit_test = os.path.join(output_base_dir, f"submit_{domain}_{test_case}_first{batch}.sh")
                with open(submit_test, 'w') as f:
                    f.write("#!/bin/bash\n")
                    f.write(f"# Quick test: first {batch} seeds for {domain} + {test_case}\n\n")
                    for script in scripts[:batch]:
                        f.write(f"sbatch {script}\n")
                os.chmod(submit_test, 0o755)
    
    # Job summary
    summary_path = os.path.join(output_base_dir, "job_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Full-System Statistical Test Job Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Domains: {domains}\n")
        f.write(f"  Test cases: {test_cases}\n")
        f.write(f"  Seeds per (domain, case): {n_seeds}\n")
        f.write(f"  Total jobs: {total_jobs}\n")
        f.write(f"  Time per job: {TIME}\n")
        f.write(f"  Memory: {MEMORY}\n\n")
        
        if random_rho_min:
            f.write(f"  RANDOM rho_min: [{rho_min_low}, {rho_min_high}]\n")
        else:
            f.write(f"  FIXED r_min={R_MIN}, r_max={R_MAX}\n")
        
        f.write(f"\nNoise: sigma = {SIGMA_NOISE}\n")
        f.write(f"Sensors: {N_SENSORS}\n")
        f.write(f"Restarts: {N_RESTARTS}\n\n")
        
        f.write("Formulas:\n")
        for tc in test_cases:
            f.write(f"  {tc}: {FORMULAS[tc]}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Submit Scripts:\n")
        f.write("  submit_all.sh                    - All jobs\n")
        for domain in domains:
            f.write(f"  submit_{domain}.sh               - All {domain} jobs\n")
        for test_case in test_cases:
            f.write(f"  submit_{test_case}.sh   - All {test_case} jobs\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Jobs by Domain:\n")
        for domain, scripts in jobs_by_domain.items():
            f.write(f"  {domain}: {len(scripts)} jobs\n")
        
        f.write("\nJobs by Test Case:\n")
        for test_case, scripts in jobs_by_case.items():
            f.write(f"  {test_case}: {len(scripts)} jobs\n")
    
    print(f"\nOutput directory: {output_base_dir}/")
    print(f"To submit all jobs: cd {output_base_dir} && bash submit_all.sh")
    
    return output_base_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs for full-system multi-domain statistical tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All domains with general_random_intensity (matches truncated tests):
  python generate_statistical_slurm_multidomain.py --all-domains --seeds 100 --random-rho-min --rho-min-high 0.7

  # Single domain, all test cases:
  python generate_statistical_slurm_multidomain.py --domain disk --all-cases --seeds 50

  # Specific domain and test case:
  python generate_statistical_slurm_multidomain.py --domain ellipse --test-case general_random_intensity --seeds 100
        """
    )
    
    # Domain selection
    domain_group = parser.add_mutually_exclusive_group(required=True)
    domain_group.add_argument('--domain', type=str, choices=DOMAINS,
                              help='Single domain to test')
    domain_group.add_argument('--all-domains', action='store_true',
                              help='Test all domains (disk, ellipse, brain)')
    
    # Test case selection
    case_group = parser.add_mutually_exclusive_group()
    case_group.add_argument('--test-case', type=str, choices=TEST_CASES,
                            default=DEFAULT_TEST_CASE,
                            help=f'Test case (default: {DEFAULT_TEST_CASE})')
    case_group.add_argument('--all-cases', action='store_true',
                            help='Run all test cases')
    
    # Other parameters
    parser.add_argument('--seeds', type=int, default=DEFAULT_SEEDS,
                        help=f'Number of seeds (default: {DEFAULT_SEEDS})')
    parser.add_argument('--random-rho-min', action='store_true',
                        help='Enable random rho_min sampling per seed')
    parser.add_argument('--rho-min-low', type=float, default=RHO_MIN_LOW,
                        help=f'Lower bound for random rho_min (default: {RHO_MIN_LOW})')
    parser.add_argument('--rho-min-high', type=float, default=RHO_MIN_HIGH,
                        help=f'Upper bound for random rho_min (default: {RHO_MIN_HIGH})')
    parser.add_argument('--output-dir', type=str, default='slurm_statistical_multidomain',
                        help='Output directory for SLURM scripts')
    
    args = parser.parse_args()
    
    # Determine domains
    if args.all_domains:
        domains = DOMAINS
    else:
        domains = [args.domain]
    
    # Determine test cases
    if args.all_cases:
        test_cases = TEST_CASES
    else:
        test_cases = [args.test_case]
    
    generate_slurm_scripts(
        domains=domains,
        test_cases=test_cases,
        n_seeds=args.seeds,
        output_base_dir=args.output_dir,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
    )


if __name__ == '__main__':
    main()
