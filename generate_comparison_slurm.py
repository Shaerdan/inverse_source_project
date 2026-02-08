#!/usr/bin/env python3
"""
Generate SLURM job scripts for nonlinear vs linear comparison experiments.

Usage:
    python generate_comparison_slurm.py --domain disk --output-dir slurm_comparison

This generates one job per (N, domain) combination:
    N ∈ {6, 8, 10, 12}
    domain ∈ {disk, ellipse, brain}

Output structure:
    comparison_results_{hash}/
    ├── disk/
    │   ├── N6/
    │   │   ├── results_disk_N6.json
    │   │   ├── summary_disk_N6.txt
    │   │   └── *.png plots
    │   ├── N8/
    │   └── ...
    ├── ellipse/
    └── brain/
"""

import os
import argparse
import hashlib
from datetime import datetime

# Default parameters
N_VALUES = [6, 8, 10, 12]
DOMAINS = ['disk', 'ellipse', 'brain']
SEED = 42

# JASMIN HPC settings
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=cmp_{domain}_N{n_sources}
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --account=eocis_chuk
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --chdir={project_dir}
#SBATCH -o logs/cmp_{domain}_N{n_sources}_%j.out
#SBATCH -e logs/cmp_{domain}_N{n_sources}_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Domain: {domain}"
echo "N sources: {n_sources}"
echo "Start: $(date)"

source ~/miniforge3/bin/activate
conda activate inverse_source

mkdir -p logs
mkdir -p {job_output_dir}

cd src
python run_comparison_job.py \\
    --n-sources {n_sources} \\
    --domain {domain} \\
    --seed {seed} \\
    --output-dir ../{job_output_dir} \\
    --rho-min 0.6 \\
    --rho-max 0.8 \\
    --n-sensors 100 \\
    --margin 2

echo "End: $(date)"
"""


def generate_jobs(domains, n_values, output_dir, project_dir, results_base, seed):
    """Generate SLURM job scripts with hierarchical output structure."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate hash from timestamp for unique results folder
    timestamp = datetime.now().isoformat()
    hash_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    results_root = f"{results_base}_{hash_str}"
    
    job_files = []
    
    for domain in domains:
        for n_sources in n_values:
            # Hierarchical output: results_root/domain/N{n}
            job_output_dir = f"{results_root}/{domain}/N{n_sources}"
            
            filename = f"job_{domain}_N{n_sources}.sh"
            filepath = os.path.join(output_dir, filename)
            
            content = SLURM_TEMPLATE.format(
                domain=domain,
                n_sources=n_sources,
                seed=seed,
                project_dir=project_dir,
                job_output_dir=job_output_dir
            )
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            os.chmod(filepath, 0o755)
            job_files.append(filename)
            print(f"Created: {filepath}")
    
    # Create submission script
    submit_script = os.path.join(output_dir, "submit_all.sh")
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Submit all comparison jobs\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Results folder: {results_root}\n\n")
        
        for job_file in job_files:
            f.write(f"sbatch {job_file}\n")
        
        f.write(f"\necho 'Submitted {len(job_files)} jobs'\n")
        f.write(f"echo 'Results will be in: {results_root}'\n")
    
    os.chmod(submit_script, 0o755)
    print(f"\nCreated submission script: {submit_script}")
    
    # Create summary
    summary = os.path.join(output_dir, "README.txt")
    with open(summary, 'w') as f:
        f.write("COMPARISON EXPERIMENT SLURM JOBS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Domains: {domains}\n")
        f.write(f"N values: {n_values}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total jobs: {len(job_files)}\n\n")
        f.write("To submit all jobs:\n")
        f.write(f"  cd {output_dir}\n")
        f.write("  bash submit_all.sh\n\n")
        f.write("Results structure:\n")
        f.write(f"  {results_root}/\n")
        for domain in domains:
            f.write(f"    ├── {domain}/\n")
            for n in n_values:
                f.write(f"    │   ├── N{n}/\n")
        f.write("\nEach N folder contains:\n")
        f.write("  - results_{domain}_N{n}.json\n")
        f.write("  - summary_{domain}_N{n}.txt\n")
        f.write("  - recovery_nonlinear_{domain}_N{n}.png\n")
        f.write("  - heatmap_greens_{method}_{resolution}_{domain}_N{n}.png\n")
        f.write("  - lcurve_{method}_{resolution}_{domain}_N{n}.png\n")
        f.write("  - sources_true_{domain}_N{n}.png\n")
    
    print(f"Created README: {summary}")
    print(f"\nTotal jobs: {len(job_files)}")
    print(f"Results folder: {results_root}")
    
    return job_files, results_root


def main():
    parser = argparse.ArgumentParser(description='Generate SLURM jobs for comparison')
    parser.add_argument('--domain', type=str, nargs='+', default=DOMAINS,
                        choices=['disk', 'ellipse', 'brain', 'all'],
                        help='Domain(s) to run')
    parser.add_argument('--n-values', type=int, nargs='+', default=N_VALUES,
                        help='N source values')
    parser.add_argument('--output-dir', type=str, default='slurm_comparison',
                        help='Directory for SLURM scripts')
    parser.add_argument('--project-dir', type=str, 
                        default='/home/users/shaerdan/inverse_source_project',
                        help='Project directory on HPC')
    parser.add_argument('--results-base', type=str, default='comparison_results',
                        help='Results directory base name (hash will be appended)')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    domains = DOMAINS if 'all' in args.domain else args.domain
    
    generate_jobs(
        domains=domains,
        n_values=args.n_values,
        output_dir=args.output_dir,
        project_dir=args.project_dir,
        results_base=args.results_base,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
