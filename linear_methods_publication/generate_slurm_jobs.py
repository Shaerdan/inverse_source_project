#!/usr/bin/env python3
"""
Generate SLURM job scripts for boundary bias analysis.

Creates job arrays for:
1. Single-seed detailed figure generation (one per domain)
2. Multi-seed validation (parallel seeds)

Usage:
    python generate_slurm_jobs.py --output-dir ./slurm_jobs
    
Then on cluster:
    cd slurm_jobs
    sbatch submit_figures.sh
    sbatch submit_multiseed.sh
"""

import os
import argparse

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --cpus-per-task=1
#SBATCH --chdir={work_dir}
#SBATCH -o logs/{job_name}_%j.out
#SBATCH -e logs/{job_name}_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

{conda_init}
conda activate {conda_env}

mkdir -p logs
mkdir -p {output_dir}

{command}

echo "End: $(date)"
"""

SLURM_ARRAY_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{max_array}
#SBATCH --chdir={work_dir}
#SBATCH -o logs/{job_name}_%A_%a.out
#SBATCH -e logs/{job_name}_%A_%a.err

echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"

{conda_init}
conda activate {conda_env}

mkdir -p logs
mkdir -p {output_dir}

SEED=$SLURM_ARRAY_TASK_ID

{command}

echo "End: $(date)"
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='./slurm_boundary_bias')
    parser.add_argument('--work-dir', default='/home/users/shaerdan/inverse_source_project')
    parser.add_argument('--account', default='eocis_chuk')
    parser.add_argument('--partition', default='standard')
    parser.add_argument('--qos', default='standard')
    parser.add_argument('--conda-init', default='source ~/miniforge3/bin/activate')
    parser.add_argument('--conda-env', default='inverse_source')
    parser.add_argument('--n-seeds', type=int, default=50)
    parser.add_argument('--results-dir', default='results_boundary_bias')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Common settings
    common = {
        'account': args.account,
        'partition': args.partition,
        'qos': args.qos,
        'work_dir': args.work_dir,
        'conda_init': args.conda_init,
        'conda_env': args.conda_env,
    }
    
    # =========================================================================
    # Job 1: Single-seed detailed figures for each domain
    # =========================================================================
    
    domains = ['disk', 'ellipse', 'brain']
    
    submit_figures = []
    for domain in domains:
        job_name = f'bbias_fig_{domain}'
        script_path = os.path.join(args.output_dir, f'{job_name}.sh')
        
        output_subdir = os.path.join(args.results_dir, 'figures', domain)
        
        command = f"""python linear_methods_publication/generate_boundary_bias_figures.py \\
    --domain {domain} \\
    --seed 42 \\
    --n-sources 4 \\
    --rho-min 0.5 \\
    --rho-max 0.7 \\
    --output-dir {output_subdir}"""
        
        script = SLURM_TEMPLATE.format(
            job_name=job_name,
            time='02:00:00',
            mem='8G',
            output_dir=output_subdir,
            command=command,
            **common
        )
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        submit_figures.append(f'sbatch {job_name}.sh')
        print(f"Created: {script_path}")
    
    # Create combined submit script
    submit_all_figures = os.path.join(args.output_dir, 'submit_figures.sh')
    with open(submit_all_figures, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Submit all figure generation jobs\n\n')
        for cmd in submit_figures:
            f.write(cmd + '\n')
    os.chmod(submit_all_figures, 0o755)
    print(f"Created: {submit_all_figures}")
    
    # =========================================================================
    # Job 2: Multi-seed validation (array job)
    # =========================================================================
    
    for domain in domains:
        job_name = f'bbias_multi_{domain}'
        script_path = os.path.join(args.output_dir, f'{job_name}.sh')
        
        output_subdir = os.path.join(args.results_dir, 'multiseed', domain)
        
        # Each array task runs one seed
        command = f"""python linear_methods_publication/run_multiseed_validation.py \\
    --domain {domain} \\
    --n-seeds 1 \\
    --start-seed $SEED \\
    --n-sources 4 \\
    --rho-min 0.5 \\
    --rho-max 0.7 \\
    --output-dir {output_subdir}/seed_$SEED"""
        
        script = SLURM_ARRAY_TEMPLATE.format(
            job_name=job_name,
            time='01:00:00',
            mem='4G',
            max_array=args.n_seeds - 1,
            output_dir=output_subdir,
            command=command,
            **common
        )
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        print(f"Created: {script_path}")
    
    # Create aggregation script (runs after array jobs complete)
    agg_script = os.path.join(args.output_dir, 'aggregate_multiseed.py')
    with open(agg_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""Aggregate multi-seed results and generate summary figures."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--n-seeds', type=int, default=50)
    parser.add_argument('--output-dir', default='./figs_boundary_bias')
    args = parser.parse_args()
    
    all_results = []
    
    for seed in range(args.n_seeds):
        result_path = os.path.join(
            args.results_dir, 'multiseed', args.domain, 
            f'seed_{seed}', f'summary_multiseed_{args.domain}.json'
        )
        
        if os.path.exists(result_path):
            with open(result_path) as f:
                data = json.load(f)
                if data['all_results']:
                    all_results.append(data['all_results'][0])
    
    print(f"Loaded {len(all_results)} seeds")
    
    if len(all_results) == 0:
        print("No results found!")
        return
    
    # Generate Figure D: Box plots
    os.makedirs(args.output_dir, exist_ok=True)
    
    methods = ['L2', 'L1', 'TV']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        orig_vals = [r[f'{method}_original']['target_pct'] for r in all_results]
        weight_vals = [r[f'{method}_weighted']['target_pct'] for r in all_results]
        
        bp = ax.boxplot([orig_vals, weight_vals], positions=[0, 1], widths=0.6,
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('coral')
        bp['boxes'][1].set_facecolor('steelblue')
        
        orig_med = np.median(orig_vals)
        weight_med = np.median(weight_vals)
        stat, p_val = stats.mannwhitneyu(weight_vals, orig_vals, alternative='greater')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Original', 'Weighted'])
        ax.set_ylabel('Target Zone Intensity (%)')
        ax.set_title(f'{method}\\nMedian: {orig_med:.1f}% → {weight_med:.1f}%\\np={p_val:.2e}')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f'fig_D_boxplots_{args.domain}_aggregated.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Print summary
    print("\\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for method in methods:
        orig = [r[f'{method}_original']['target_pct'] for r in all_results]
        weighted = [r[f'{method}_weighted']['target_pct'] for r in all_results]
        print(f"{method}: {np.median(orig):.1f}% → {np.median(weighted):.1f}% "
              f"(improvement: +{np.median(weighted) - np.median(orig):.1f}%)")

if __name__ == '__main__':
    main()
''')
    os.chmod(agg_script, 0o755)
    print(f"Created: {agg_script}")
    
    # Submit script for multi-seed
    submit_multi = os.path.join(args.output_dir, 'submit_multiseed.sh')
    with open(submit_multi, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Submit all multi-seed validation jobs\n\n')
        for domain in domains:
            f.write(f'sbatch bbias_multi_{domain}.sh\n')
    os.chmod(submit_multi, 0o755)
    print(f"Created: {submit_multi}")
    
    print("\n" + "="*60)
    print("SLURM jobs generated!")
    print("="*60)
    print(f"\nTo run:")
    print(f"  cd {args.output_dir}")
    print(f"  bash submit_figures.sh     # Detailed figures (3 jobs)")
    print(f"  bash submit_multiseed.sh   # Statistical validation (3 × {args.n_seeds} jobs)")
    print(f"\nAfter completion, aggregate:")
    print(f"  python aggregate_multiseed.py --domain disk --results-dir ../{args.results_dir} --n-seeds {args.n_seeds}")


if __name__ == '__main__':
    main()
