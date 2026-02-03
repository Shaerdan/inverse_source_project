#!/usr/bin/env python3
"""
SLURM Job Generator for Bound Theory Validation

Generates one SLURM script per (domain, rho, seed) combination.
Creates submit_all.sh to submit all jobs at once.

Usage:
    python generate_slurm_jobs.py
    cd slurm_jobs/
    bash submit_all.sh

Configuration (edit below):
    - rho values: 0.6 to 0.8, 10 points
    - domains: disk, ellipse, brain
    - seeds: 0, 1, 2, 3, 4 (5 seeds for variance estimation)
    - Total jobs: 3 × 10 × 5 = 150
"""

import os
import numpy as np
from datetime import datetime

# =============================================================================
# CONFIGURATION - Edit these as needed
# =============================================================================

# JASMIN settings
ACCOUNT = "eocis_chuk"
PARTITION = "standard"
QOS = "standard"
TIME = "04:00:00"
MEMORY = "32G"
CPUS = 1

# Paths
PROJECT_DIR = "/home/users/shaerdan/inverse_source_project"
CONDA_INIT = "source ~/miniforge3/bin/activate"
CONDA_ENV = "inverse_source"

# Experiment parameters
DOMAINS = ['disk', 'ellipse', 'brain']
RHO_MIN = 0.60
RHO_MAX = 0.80
RHO_STEPS = 10  # 10 points from 0.6 to 0.8

# Multiple seeds for variance estimation
SEEDS = [0, 1, 2, 3, 4]  # 5 seeds

SIGMA_NOISE = 0.001
N_SENSORS = 100
N_RESTARTS = 15
THRESHOLD = 0.05

# Output
SLURM_OUTPUT_DIR = "slurm_jobs"
RESULTS_DIR = "results"  # relative to PROJECT_DIR

# =============================================================================
# SLURM TEMPLATE
# =============================================================================

SLURM_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=bound_{domain}_{rho_str}_s{seed}
#SBATCH --time={time}
#SBATCH --mem={memory}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --cpus-per-task={cpus}
#SBATCH --chdir={project_dir}
#SBATCH -o {project_dir}/logs/bound_{domain}_{rho_str}_s{seed}_%j.out
#SBATCH -e {project_dir}/logs/bound_{domain}_{rho_str}_s{seed}_%j.err

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Domain: {domain}"
echo "Rho: {rho}"
echo "Seed: {seed}"
echo "Start: $(date)"
echo "=========================================="

# Activate conda environment
{conda_init}
conda activate {conda_env}

# Ensure logs and results directories exist
mkdir -p {project_dir}/logs
mkdir -p {project_dir}/{results_dir}

# Run experiment
cd {project_dir}/src
python run_bound_experiment.py \\
    --domain {domain} \\
    --rho {rho} \\
    --sigma-noise {sigma_noise} \\
    --sensors {n_sensors} \\
    --restarts {n_restarts} \\
    --seed {seed} \\
    --threshold {threshold} \\
    --output-dir {project_dir}/{results_dir}

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
'''

# =============================================================================
# GENERATOR
# =============================================================================

def generate_slurm_scripts():
    """Generate all SLURM scripts."""
    
    # Create output directory
    os.makedirs(SLURM_OUTPUT_DIR, exist_ok=True)
    
    # Generate rho values
    rho_values = np.linspace(RHO_MIN, RHO_MAX, RHO_STEPS)
    
    total_jobs = len(DOMAINS) * len(rho_values) * len(SEEDS)
    
    print(f"Generating SLURM scripts...")
    print(f"  Domains: {DOMAINS}")
    print(f"  Rho values: {[f'{r:.2f}' for r in rho_values]}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total jobs: {len(DOMAINS)} × {len(rho_values)} × {len(SEEDS)} = {total_jobs}")
    print()
    
    job_scripts = []
    
    for domain in DOMAINS:
        for rho in rho_values:
            for seed in SEEDS:
                rho_str = f"{rho:.2f}".replace('.', 'p')  # 0.70 -> 0p70 for filename
                
                # Generate script content
                script_content = SLURM_TEMPLATE.format(
                    domain=domain,
                    rho=rho,
                    rho_str=rho_str,
                    seed=seed,
                    time=TIME,
                    memory=MEMORY,
                    account=ACCOUNT,
                    partition=PARTITION,
                    qos=QOS,
                    cpus=CPUS,
                    project_dir=PROJECT_DIR,
                    conda_init=CONDA_INIT,
                    conda_env=CONDA_ENV,
                    results_dir=RESULTS_DIR,
                    sigma_noise=SIGMA_NOISE,
                    n_sensors=N_SENSORS,
                    n_restarts=N_RESTARTS,
                    threshold=THRESHOLD,
                )
                
                # Write script
                script_name = f"run_{domain}_{rho_str}_s{seed}.sh"
                script_path = os.path.join(SLURM_OUTPUT_DIR, script_name)
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                job_scripts.append(script_name)
    
    print(f"  Created {len(job_scripts)} job scripts")
    
    # Create submit_all.sh
    submit_script = os.path.join(SLURM_OUTPUT_DIR, "submit_all.sh")
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all bound validation jobs\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total jobs: {len(job_scripts)}\n")
        f.write(f"# Domains: {DOMAINS}\n")
        f.write(f"# Rho: {RHO_MIN} to {RHO_MAX} ({RHO_STEPS} points)\n")
        f.write(f"# Seeds: {SEEDS}\n\n")
        
        f.write("echo 'Submitting all bound validation jobs...'\n")
        f.write(f"echo 'Total: {len(job_scripts)} jobs'\n\n")
        
        for script in job_scripts:
            f.write(f"sbatch {script}\n")
        
        f.write("\necho 'All jobs submitted!'\n")
        f.write("echo 'Check status with: squeue -u $USER'\n")
    
    os.chmod(submit_script, 0o755)
    print(f"  Created: {submit_script}")
    
    # Create submit by domain scripts (for partial submission)
    for domain in DOMAINS:
        domain_script = os.path.join(SLURM_OUTPUT_DIR, f"submit_{domain}.sh")
        domain_jobs = [s for s in job_scripts if s.startswith(f"run_{domain}_")]
        with open(domain_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Submit {domain} jobs only\n")
            f.write(f"# Total: {len(domain_jobs)} jobs\n\n")
            for script in domain_jobs:
                f.write(f"sbatch {script}\n")
            f.write(f"\necho '{domain}: {len(domain_jobs)} jobs submitted'\n")
        os.chmod(domain_script, 0o755)
        print(f"  Created: {domain_script}")
    
    # Create job summary
    summary_path = os.path.join(SLURM_OUTPUT_DIR, "job_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Bound Validation Job Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total jobs: {len(job_scripts)}\n\n")
        f.write("Parameters:\n")
        f.write(f"  Domains: {DOMAINS}\n")
        f.write(f"  Rho range: [{RHO_MIN}, {RHO_MAX}] with {RHO_STEPS} points\n")
        f.write(f"  Seeds: {SEEDS}\n")
        f.write(f"  sigma_noise: {SIGMA_NOISE}\n")
        f.write(f"  Sensors: {N_SENSORS}\n")
        f.write(f"  Restarts: {N_RESTARTS}\n")
        f.write(f"  Threshold: {THRESHOLD}\n\n")
        f.write("Submit options:\n")
        f.write(f"  All jobs:    bash submit_all.sh\n")
        for domain in DOMAINS:
            f.write(f"  {domain:10s}: bash submit_{domain}.sh\n")
    
    print(f"  Created: {summary_path}")
    
    print(f"\n{'='*50}")
    print("DONE!")
    print(f"{'='*50}")
    print(f"\nTo submit all {total_jobs} jobs:")
    print(f"  cd {SLURM_OUTPUT_DIR}")
    print(f"  bash submit_all.sh")
    print(f"\nOr submit by domain:")
    for domain in DOMAINS:
        n = len(rho_values) * len(SEEDS)
        print(f"  bash submit_{domain}.sh  ({n} jobs)")
    

if __name__ == '__main__':
    generate_slurm_scripts()
