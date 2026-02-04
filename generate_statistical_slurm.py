#!/usr/bin/env python3
"""
SLURM Job Generator for Statistical Validation

Generates one SLURM script per seed (100 total).
Each job runs 8 N values for that seed.

Usage:
    python generate_statistical_slurm.py
    cd slurm_statistical/
    bash submit_all.sh

Expected runtime: ~5-10 min per seed (8 inverse solves)
Total: 100 parallel jobs
"""

import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# JASMIN settings
ACCOUNT = "eocis_chuk"
PARTITION = "standard"
QOS = "standard"
TIME = "01:00:00"  # 1 hour should be plenty per seed
MEMORY = "16G"
CPUS = 1

# Paths
PROJECT_DIR = "/home/users/shaerdan/inverse_source_project"
CONDA_INIT = "source ~/miniforge3/bin/activate"
CONDA_ENV = "inverse_source"

# Experiment parameters (from spec)
SEEDS = list(range(100))  # 0 to 99
RHO = 0.7
SIGMA_NOISE = 0.001
N_SENSORS = 100
N_RESTARTS = 15

# Output
SLURM_OUTPUT_DIR = "slurm_statistical"
RESULTS_DIR = "stat_results"

# =============================================================================
# SLURM TEMPLATE
# =============================================================================

SLURM_TEMPLATE = '''#!/bin/bash
#SBATCH --job-name=stat_seed{seed:03d}
#SBATCH --time={time}
#SBATCH --mem={memory}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --cpus-per-task={cpus}
#SBATCH --chdir={project_dir}
#SBATCH -o {project_dir}/logs/stat_seed{seed:03d}_%j.out
#SBATCH -e {project_dir}/logs/stat_seed{seed:03d}_%j.err

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Statistical Validation - Seed {seed}"
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
    --rho {rho} \\
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

def generate_slurm_scripts():
    """Generate all SLURM scripts for statistical validation."""
    
    os.makedirs(SLURM_OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating SLURM scripts for statistical validation...")
    print(f"  Seeds: 0 to {len(SEEDS)-1} ({len(SEEDS)} total)")
    print(f"  N values per seed: [10, 12, 14, 16, 18, 20, 22, 24]")
    print(f"  Total inverse solves: {len(SEEDS)} × 8 = {len(SEEDS) * 8}")
    print()
    
    job_scripts = []
    
    for seed in SEEDS:
        script_content = SLURM_TEMPLATE.format(
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
            rho=RHO,
            sigma_noise=SIGMA_NOISE,
            n_sensors=N_SENSORS,
            n_restarts=N_RESTARTS,
        )
        
        script_name = f"run_seed{seed:03d}.sh"
        script_path = os.path.join(SLURM_OUTPUT_DIR, script_name)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        job_scripts.append(script_name)
    
    print(f"  Created {len(job_scripts)} job scripts")
    
    # Create submit_all.sh
    submit_script = os.path.join(SLURM_OUTPUT_DIR, "submit_all.sh")
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all statistical validation jobs\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total jobs: {len(job_scripts)}\n")
        f.write(f"# Seeds: 0 to {len(SEEDS)-1}\n\n")
        
        f.write("echo 'Submitting statistical validation jobs...'\n")
        f.write(f"echo 'Total: {len(job_scripts)} jobs'\n\n")
        
        for script in job_scripts:
            f.write(f"sbatch {script}\n")
        
        f.write("\necho 'All jobs submitted!'\n")
        f.write("echo 'Check status with: squeue -u $USER'\n")
    
    os.chmod(submit_script, 0o755)
    print(f"  Created: {submit_script}")
    
    # Create submit in batches (for testing)
    batch_size = 10
    for batch_start in range(0, len(SEEDS), batch_size):
        batch_end = min(batch_start + batch_size, len(SEEDS))
        batch_script = os.path.join(SLURM_OUTPUT_DIR, f"submit_seeds_{batch_start:03d}_{batch_end-1:03d}.sh")
        
        with open(batch_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Submit seeds {batch_start} to {batch_end-1}\n\n")
            for seed in range(batch_start, batch_end):
                f.write(f"sbatch run_seed{seed:03d}.sh\n")
            f.write(f"\necho 'Submitted seeds {batch_start}-{batch_end-1}'\n")
        
        os.chmod(batch_script, 0o755)
    
    print(f"  Created batch submit scripts (10 seeds each)")
    
    # Create job summary
    summary_path = os.path.join(SLURM_OUTPUT_DIR, "job_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Statistical Validation Job Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total jobs: {len(job_scripts)}\n\n")
        f.write("Parameters:\n")
        f.write(f"  Domain: disk\n")
        f.write(f"  Rho: {RHO}\n")
        f.write(f"  sigma_noise: {SIGMA_NOISE}\n")
        f.write(f"  Sensors: {N_SENSORS}\n")
        f.write(f"  Restarts: {N_RESTARTS}\n")
        f.write(f"  Seeds: 0 to {len(SEEDS)-1}\n")
        f.write(f"  N values: [10, 12, 14, 16, 18, 20, 22, 24]\n\n")
        f.write("Computed reference values:\n")
        sigma_four = SIGMA_NOISE / (N_SENSORS ** 0.5)
        import math
        n_star_pred = math.log(sigma_four) / math.log(RHO)
        N_max_pred = (2/3) * n_star_pred
        f.write(f"  sigma_four = {sigma_four}\n")
        f.write(f"  n*_predicted = {n_star_pred:.2f}\n")
        f.write(f"  N_max_predicted = {N_max_pred:.2f}\n\n")
        f.write("Submit options:\n")
        f.write(f"  All jobs: bash submit_all.sh\n")
        f.write(f"  First 10: bash submit_seeds_000_009.sh\n")
    
    print(f"  Created: {summary_path}")
    
    print(f"\n{'='*50}")
    print("DONE!")
    print(f"{'='*50}")
    print(f"\nTo submit all {len(SEEDS)} jobs:")
    print(f"  cd {SLURM_OUTPUT_DIR}")
    print(f"  bash submit_all.sh")
    print(f"\nOr test with first 10:")
    print(f"  bash submit_seeds_000_009.sh")


if __name__ == '__main__':
    generate_slurm_scripts()
