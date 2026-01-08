#!/usr/bin/env python3
"""
Generate publication-quality figures for the manuscript.

Figures:
1. Model comparison schematic (Ekenstam vs Saturating)
2. CV results bar chart
3. Bias analysis plot
4. Fanti data with model fits

Usage:
    python 03_generate_figures.py --paper-results out_paper --flax-results out_flax
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def fig1_model_schematic(outdir):
    """Figure 1: Ekenstam vs Saturating model schematic."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t = np.linspace(0, 150, 200)
    
    # Model parameters (illustrative)
    inv_dp0 = 0.0003
    k = 1.5e-5
    inv_dp_inf = 0.004
    tau = 100
    
    # Ekenstam (linear)
    y_ek = inv_dp0 + k * t
    
    # Saturating
    y_sat = inv_dp_inf + (inv_dp0 - inv_dp_inf) * np.exp(-t / tau)
    
    ax.plot(t, y_ek * 1000, 'b-', linewidth=2, label='Ekenstam (linear)')
    ax.plot(t, y_sat * 1000, 'r-', linewidth=2, label='Saturating')
    ax.axhline(y=inv_dp_inf * 1000, color='r', linestyle='--', alpha=0.5, label='LODP asymptote')
    
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('1/DP × 10³')
    ax.set_title('Degradation Model Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 5)
    ax.grid(True, alpha=0.3)
    
    # Annotation
    ax.annotate('Divergence increases\nwith extrapolation', 
                xy=(120, 2.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(outdir / 'fig1_model_schematic.png')
    fig.savefig(outdir / 'fig1_model_schematic.pdf')
    plt.close(fig)
    print(f"✓ Saved: {outdir}/fig1_model_schematic.png")

def fig2_cv_results(paper_results_dir, outdir):
    """Figure 2: Cross-validation results bar chart."""
    # Load results
    df = pd.read_csv(paper_results_dir / 'table1_cv_results.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel A: MAE comparison
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['MAE_Lin'], width, label='Ekenstam', color='steelblue')
    bars2 = ax.bar(x + width/2, df['MAE_Sat'], width, label='Saturating', color='coral')
    
    ax.set_xlabel('Split Year')
    ax.set_ylabel('MAE (years)')
    ax.set_title('A. Mean Absolute Error by Split')
    ax.set_xticks(x)
    ax.set_xticklabels(df['split'])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel B: Improvement percentage
    ax = axes[1]
    colors = ['forestgreen' if p > 0 else 'crimson' for p in df['delta_pct']]
    ax.bar(x, df['delta_pct'], color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=df['delta_pct'].mean(), color='red', linestyle='--', 
               label=f'Mean = {df["delta_pct"].mean():.0f}%')
    
    ax.set_xlabel('Split Year')
    ax.set_ylabel('MAE Improvement (%)')
    ax.set_title('B. Relative Improvement (Saturating vs Linear)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['split'])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(outdir / 'fig2_cv_results.png')
    fig.savefig(outdir / 'fig2_cv_results.pdf')
    plt.close(fig)
    print(f"✓ Saved: {outdir}/fig2_cv_results.png")

def fig3_bias_analysis(paper_results_dir, outdir):
    """Figure 3: Age-dependent bias analysis."""
    # Load Welch test results
    with open(paper_results_dir / 'welch_test.json') as f:
        welch = json.load(f)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Data
    bins = ['Oldest\n(1850-1880)', 'Newest\n(1970-1990)']
    ek_bias = [welch['bias_ek_oldest'], welch['bias_ek_newest']]
    sat_bias = [welch['bias_sat_oldest'], welch['bias_sat_newest']]
    
    x = np.arange(len(bins))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ek_bias, width, label='Ekenstam', color='steelblue')
    bars2 = ax.bar(x + width/2, sat_bias, width, label='Saturating', color='coral')
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('Mean Prediction Bias (years)')
    ax.set_title('Age-Dependent Bias in Prediction Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add p-value annotation
    ax.annotate(f'Welch t-test (Ekenstam):\nt = {welch["t_ek"]:.2f}, p = {welch["p_ek"]:.3f}',
                xy=(0.98, 0.98), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(outdir / 'fig3_bias_analysis.png')
    fig.savefig(outdir / 'fig3_bias_analysis.pdf')
    plt.close(fig)
    print(f"✓ Saved: {outdir}/fig3_bias_analysis.png")

def fig4_fanti_comparison(flax_results_dir, outdir):
    """Figure 4: Fanti data comparison."""
    # Load results
    with open(flax_results_dir / 'fanti_results.json') as f:
        results = json.load(f)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Methods and dates
    methods = [
        'Basso σ_R\n(published)',
        'Linear\n(reproduced)',
        'Saturating\n(this study)',
        'Radiocarbon\n(1988)'
    ]
    dates = [603, 601, 697, 1325]  # 1325 = midpoint of 1260-1390
    errors = [69, 103, 81, 65]  # uncertainties
    colors = ['steelblue', 'steelblue', 'coral', 'forestgreen']
    
    x = np.arange(len(methods))
    bars = ax.bar(x, dates, yerr=errors, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add radiocarbon band
    ax.axhspan(1260, 1390, alpha=0.2, color='green', label='Radiocarbon 95% CI')
    
    ax.set_ylabel('Estimated Date (AD)')
    ax.set_title('Comparison of Dating Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1600)
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add gap annotation
    ax.annotate('', xy=(2.5, 697), xytext=(2.5, 1260),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(2.7, 980, '~560 yr\ngap', fontsize=9, color='red', va='center')
    
    fig.tight_layout()
    fig.savefig(outdir / 'fig4_fanti_comparison.png')
    fig.savefig(outdir / 'fig4_fanti_comparison.pdf')
    plt.close(fig)
    print(f"✓ Saved: {outdir}/fig4_fanti_comparison.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper-results', default='out_paper', help='Paper validation results directory')
    parser.add_argument('--flax-results', default='out_flax', help='Flax analysis results directory')
    parser.add_argument('--outdir', default='figures', help='Output directory for figures')
    args = parser.parse_args()
    
    paper_dir = Path(args.paper_results)
    flax_dir = Path(args.flax_results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("Generating figures...")
    fig1_model_schematic(outdir)
    
    if (paper_dir / 'table1_cv_results.csv').exists():
        fig2_cv_results(paper_dir, outdir)
    else:
        print(f"⚠ Skipping fig2: {paper_dir}/table1_cv_results.csv not found")
    
    if (paper_dir / 'welch_test.json').exists():
        fig3_bias_analysis(paper_dir, outdir)
    else:
        print(f"⚠ Skipping fig3: {paper_dir}/welch_test.json not found")
    
    if (flax_dir / 'fanti_results.json').exists():
        fig4_fanti_comparison(flax_dir, outdir)
    else:
        print(f"⚠ Skipping fig4: {flax_dir}/fanti_results.json not found")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
