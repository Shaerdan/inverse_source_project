#!/usr/bin/env python3
"""
SLURM Job Generator for Truncated RMSE Test - Conformal Domains
=================================================================

Generates SLURM jobs for ellipse and brain domains.

Usage:
    python generate_rmse_truncated_conformal_slurm.py --domain ellipse --seeds 100
    python generate_rmse_truncated_conformal_slurm.py --domain brain --seeds 100 --random-rho-min
    python generate_rmse_truncated_conformal_slurm.py --all-domains --seeds 50
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
TIME = "06:00:00"  # 6 hours — conformal map adds overhead
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
R_MIN = 0.5
R_MAX = 0.7

TEST_CASE = 'general_random_intensity'

DOMAINS = ['ellipse', 'brain']

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
#SBATCH -o {project_dir}/logs/trunc_{domain}_{test_case}_seed{seed:03d}_%j.out
#SBATCH -e {project_dir}/logs/trunc_{domain}_{test_case}_seed{seed:03d}_%j.err
#SBATCH --exclude=host1117

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Test: Truncated RMSE (Conformal)"
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
python run_rmse_truncated_conformal.py \\
    --seed {seed} \\
    --domain {domain} \\
    --test-case {test_case} \\
    --r-min {r_min} \\
    --r-max {r_max} \\
    --sigma-noise {sigma_noise} \\
    --sensors {n_sensors} \\
    --n-restarts {n_restarts} \\
    --output-dir {project_dir}/{results_dir}{random_rho_args}{domain_args}

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
'''


def generate_slurm_scripts(domains, test_case, n_seeds, output_base_dir='slurm_rmse_truncated_conformal',
                           random_rho_min=False, rho_min_low=0.5, rho_min_high=0.7,
                           ellipse_a=1.5, ellipse_b=0.8):
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Generating SLURM scripts for truncated RMSE test (Conformal)...")
    print(f"  Domains: {domains}")
    print(f"  Test case: {test_case}")
    print(f"  Seeds per domain: {n_seeds}")
    print(f"  Total jobs: {len(domains) * n_seeds}")
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
    
    for domain in domains:
        results_dir = f"rmse_truncated_results_{domain}_{test_case}"
        if random_rho_min:
            results_dir += "_random_rho"
        
        # Domain-specific args
        if domain == 'ellipse':
            domain_args = f" \\\n    --ellipse-a {ellipse_a} \\\n    --ellipse-b {ellipse_b}"
        else:
            domain_args = ""
        
        for seed in range(n_seeds):
            abbrev = {'ellipse': 'tel', 'brain': 'tbr'}[domain]
            job_name = f"{abbrev}_{seed:03d}"
            
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
                domain_args=domain_args,
            )
            
            script_name = f"run_trunc_{domain}_seed{seed:03d}.sh"
            script_path = os.path.join(output_base_dir, script_name)
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            all_job_scripts.append(script_name)
            jobs_by_domain[domain].append(script_name)
    
    print(f"  Created {len(all_job_scripts)} job scripts")
    
    # submit_all.sh
    submit_all = os.path.join(output_base_dir, "submit_all.sh")
    with open(submit_all, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit all truncated RMSE conformal jobs\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total jobs: {len(all_job_scripts)}\n\n")
        for script in all_job_scripts:
            f.write(f"sbatch {script}\n")
        f.write("\necho 'All jobs submitted!'\n")
    os.chmod(submit_all, 0o755)
    
    # Per-domain submit scripts
    for domain, scripts in jobs_by_domain.items():
        submit_domain = os.path.join(output_base_dir, f"submit_{domain}.sh")
        with open(submit_domain, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Submit truncated RMSE: {domain} ({len(scripts)} jobs)\n\n")
            for script in scripts:
                f.write(f"sbatch {script}\n")
            f.write(f"\necho 'Submitted {len(scripts)} {domain} jobs'\n")
        os.chmod(submit_domain, 0o755)
    
    # Quick-test (first 5 seeds per domain)
    for domain in domains:
        scripts = jobs_by_domain[domain]
        batch = min(5, len(scripts))
        submit_test = os.path.join(output_base_dir, f"submit_{domain}_first{batch}.sh")
        with open(submit_test, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Quick test: first {batch} seeds for {domain}\n\n")
            for script in scripts[:batch]:
                f.write(f"sbatch {script}\n")
        os.chmod(submit_test, 0o755)
    
    # Job summary
    summary_path = os.path.join(output_base_dir, "job_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Truncated RMSE Test (Conformal Domains) Job Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total jobs: {len(all_job_scripts)}\n")
        f.write(f"Walltime per job: {TIME}\n\n")
        for domain in domains:
            f.write(f"  {domain}: {len(jobs_by_domain[domain])} jobs\n")
        f.write(f"\nTest case: {test_case}\n")
        f.write(f"Expected bound: {FORMULAS.get(test_case, 'unknown')}\n")
        f.write(f"\nParameters:\n")
        f.write(f"  sigma_noise: {SIGMA_NOISE}, sensors: {N_SENSORS}\n")
        f.write(f"  n_restarts: {N_RESTARTS}\n")
        f.write(f"  Dynamic N values: ±6 around predicted N_max, step=2\n")
        if random_rho_min:
            f.write(f"  Random rho_min: [{rho_min_low}, {rho_min_high}]\n")
        if 'ellipse' in domains:
            f.write(f"  Ellipse: a={ellipse_a}, b={ellipse_b}\n")
        f.write(f"\nSubmit: cd {output_base_dir} && bash submit_all.sh\n")
    
    print(f"  Created submit scripts and summary")
    print(f"\nTo submit: cd {output_base_dir} && bash submit_all.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs for truncated RMSE test (conformal domains)"
    )
    parser.add_argument('--domain', type=str, choices=['ellipse', 'brain'])
    parser.add_argument('--all-domains', action='store_true')
    parser.add_argument('--test-case', type=str, default=TEST_CASE,
                        choices=['same_radius', 'same_angle', 'general',
                                 'general_random_intensity'])
    parser.add_argument('--seeds', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--output-dir', type=str, default='slurm_rmse_truncated_conformal')
    parser.add_argument('--random-rho-min', action='store_true')
    parser.add_argument('--rho-min-low', type=float, default=RHO_MIN_LOW)
    parser.add_argument('--rho-min-high', type=float, default=RHO_MIN_HIGH)
    parser.add_argument('--ellipse-a', type=float, default=1.5)
    parser.add_argument('--ellipse-b', type=float, default=0.8)
    
    args = parser.parse_args()
    
    if args.all_domains:
        domains = DOMAINS
    elif args.domain:
        domains = [args.domain]
    else:
        print("Specify --domain or --all-domains")
        parser.print_help()
        return
    
    generate_slurm_scripts(
        domains, args.test_case, args.seeds, args.output_dir,
        random_rho_min=args.random_rho_min,
        rho_min_low=args.rho_min_low,
        rho_min_high=args.rho_min_high,
        ellipse_a=args.ellipse_a,
        ellipse_b=args.ellipse_b,
    )


if __name__ == '__main__':
    main()
