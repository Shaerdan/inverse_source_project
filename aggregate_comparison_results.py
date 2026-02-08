#!/usr/bin/env python3
"""
Aggregate Comparison Results
============================

Compiles results from all comparison jobs into summary tables and plots.

Usage:
    python aggregate_comparison_results.py --results-dir comparison_results
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List


def load_all_results(results_dir: str) -> List[Dict]:
    """Load all results JSON files."""
    results = []
    
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.startswith('results_') and f.endswith('.json'):
                path = os.path.join(root, f)
                try:
                    with open(path, 'r') as fp:
                        data = json.load(fp)
                        data['_file'] = path
                        results.append(data)
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
    
    return results


def create_summary_table(results: List[Dict]) -> str:
    """Create markdown summary table."""
    
    lines = []
    lines.append("# Nonlinear vs Linear FEM Comparison Results\n")
    lines.append(f"Total experiments: {len(results)}\n")
    
    # Group by domain
    by_domain = {}
    for r in results:
        domain = r.get('domain', 'unknown')
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(r)
    
    for domain in sorted(by_domain.keys()):
        lines.append(f"\n## {domain.upper()} Domain\n")
        
        # Header
        lines.append("| N | n* | N_max | Nonlinear RMSE | Linear (fine) RMSE | Ratio | μ (fine) |")
        lines.append("|---|----|----|----------------|-------------------|-------|----------|")
        
        domain_results = sorted(by_domain[domain], key=lambda x: x.get('n_sources', 0))
        
        for r in domain_results:
            N = r.get('n_sources', '?')
            n_star = r.get('n_star_actual', '?')
            N_max = r.get('N_max_actual', '?')
            
            # Nonlinear
            nl = r.get('nonlinear', {})
            nl_rmse = nl.get('rmse_position', float('inf'))
            nl_str = f"{nl_rmse:.4f}" if nl_rmse < 100 else "FAILED"
            
            # Linear (fine resolution)
            lin = r.get('linear_fem', {}).get('fine', {})
            lin_rmse = lin.get('rmse_position', float('inf'))
            lin_str = f"{lin_rmse:.4f}" if lin_rmse < 100 else "FAILED"
            
            mu = lin.get('mutual_coherence', '?')
            mu_str = f"{mu:.4f}" if isinstance(mu, float) else str(mu)
            
            # Ratio
            if nl_rmse > 0 and nl_rmse < 100 and lin_rmse < 100:
                ratio = lin_rmse / nl_rmse
                ratio_str = f"{ratio:.1f}×"
            else:
                ratio_str = "N/A"
            
            lines.append(f"| {N} | {n_star} | {N_max} | {nl_str} | {lin_str} | {ratio_str} | {mu_str} |")
    
    return "\n".join(lines)


def create_comparison_plot(results: List[Dict], output_path: str):
    """Create bar chart comparing nonlinear vs linear RMSE."""
    
    # Organize data
    domains = sorted(set(r.get('domain', '') for r in results))
    n_values = sorted(set(r.get('n_sources', 0) for r in results))
    
    fig, axes = plt.subplots(1, len(domains), figsize=(5*len(domains), 5), sharey=True)
    if len(domains) == 1:
        axes = [axes]
    
    for ax, domain in zip(axes, domains):
        domain_results = [r for r in results if r.get('domain') == domain]
        domain_results = sorted(domain_results, key=lambda x: x.get('n_sources', 0))
        
        ns = []
        nl_rmse = []
        lin_rmse = []
        
        for r in domain_results:
            ns.append(r.get('n_sources', 0))
            
            nl = r.get('nonlinear', {})
            nl_rmse.append(nl.get('rmse_position', np.nan))
            
            lin = r.get('linear_fem', {}).get('fine', {})
            lin_rmse.append(lin.get('rmse_position', np.nan))
        
        x = np.arange(len(ns))
        width = 0.35
        
        ax.bar(x - width/2, nl_rmse, width, label='Nonlinear', color='green', alpha=0.7)
        ax.bar(x + width/2, lin_rmse, width, label='Linear FEM', color='red', alpha=0.7)
        
        ax.set_xlabel('Number of Sources (N)')
        ax.set_ylabel('Position RMSE')
        ax.set_title(f'{domain.capitalize()} Domain')
        ax.set_xticks(x)
        ax.set_xticklabels(ns)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Created: {output_path}")


def create_coherence_plot(results: List[Dict], output_path: str):
    """Plot mutual coherence vs grid resolution."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    resolutions = ['coarse', 'medium', 'fine', 'very_fine']
    
    for r in results:
        domain = r.get('domain', '')
        N = r.get('n_sources', 0)
        
        mus = []
        for res in resolutions:
            lin = r.get('linear_fem', {}).get(res, {})
            mu = lin.get('mutual_coherence', np.nan)
            mus.append(mu)
        
        label = f"{domain} N={N}"
        ax.plot(range(len(resolutions)), mus, 'o-', label=label, alpha=0.7)
    
    ax.set_xlabel('Grid Resolution')
    ax.set_ylabel('Mutual Coherence μ')
    ax.set_title('Mutual Coherence vs Grid Resolution')
    ax.set_xticks(range(len(resolutions)))
    ax.set_xticklabels(resolutions)
    ax.set_ylim(0.99, 1.001)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate comparison results')
    parser.add_argument('--results-dir', type=str, default='comparison_results',
                        help='Directory containing results')
    parser.add_argument('--output-dir', type=str, default='comparison_summary',
                        help='Output directory for summary')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)
    print(f"Found {len(results)} result files")
    
    if len(results) == 0:
        print("No results found!")
        return
    
    # Create summary table
    summary = create_summary_table(results)
    summary_path = os.path.join(args.output_dir, 'summary.md')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Created: {summary_path}")
    
    # Create plots
    create_comparison_plot(results, os.path.join(args.output_dir, 'rmse_comparison.png'))
    create_coherence_plot(results, os.path.join(args.output_dir, 'coherence_vs_resolution.png'))
    
    print(f"\nSummary written to {args.output_dir}/")


if __name__ == '__main__':
    main()
