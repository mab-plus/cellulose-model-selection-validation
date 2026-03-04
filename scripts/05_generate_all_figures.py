#!/usr/bin/env python3
"""
05_generate_all_figures.py
==========================
Generate ALL 6 publication-quality figures for the manuscript.

Figure map (manuscript → this script):
  fig1_model_schematic.png        → fig1_model_schematic()       [standalone]
  fig2_cv_results.png             → fig2_cv_results()            [needs 01 output]
  fig3_bias_analysis.png          → fig3_bias_analysis()         [needs 01 output]
  fig_fanti_fire_omission.png     → fig4_fire_omission()         [standalone]
  fig_fire_correction_article.png → fig5_fire_correction()       [standalone MC]
  fig_constraint_comparison.png   → fig6_constraint_comparison() [standalone MC]

Usage:
  python 05_generate_all_figures.py --outdir figures
  python 05_generate_all_figures.py --outdir figures --paper-results out_paper

Dependencies: numpy, matplotlib, scipy
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────
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
    'savefig.bbox': 'tight',
})

# ─────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────
R_GAS   = 8.314       # J/(mol·K)
E_A     = 110e3       # J/mol — Emsley & Stevens 1994: 111 ± 6 kJ/mol
T0_K    = 293.15      # 20 °C ambient reference
C14_MIN = 1260
C14_MAX = 1390
Y_FIRE  = 1532

# Basso calibration: ln(σ_R) = m·year + q
M_BASSO = 9.256e-4
Q_BASSO = 4.936
M_STD   = 0.5e-4
Q_STD   = 0.05


def basso_date(sigma_mpa, m=M_BASSO, q=Q_BASSO):
    """Invert Basso calibration → year AD."""
    return (np.log(sigma_mpa) - q) / m


def arrhenius_teq_years(dt_h, T_fire_C, Ea=E_A, T0=T0_K):
    """Equivalent aging time at ambient, in years."""
    T_fire_K = T_fire_C + 273.15
    factor = np.exp(Ea / R_GAS * (1.0 / T0 - 1.0 / T_fire_K))
    return dt_h * 3600.0 * factor / (3600.0 * 24.0 * 365.25)


# =====================================================================
# FIG 1 — Model comparison schematic (Ekenstam vs Saturating)
# =====================================================================
def fig1_model_schematic(outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    t = np.linspace(0, 2500, 500)

    # (A) Ekenstam — linear 1/DP
    ax = axes[0]
    inv_dp0 = 1 / 5000
    k = 3e-7
    y_ek = (inv_dp0 + k * t) * 1e3
    ax.plot(t, y_ek, 'b-', lw=2, label='Ekenstam (linear)')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('1/DP × 10³')
    ax.set_title('(A) Ekenstam model')
    ax.axhline(y=1 / 200 * 1e3, color='grey', ls=':', alpha=0.5, label='LODP ≈ 200')
    ax.fill_between([1500, 2500], 0, 1.2, alpha=0.08, color='red')
    ax.annotate('Unbounded\nextrapolation',
                xy=(2000, 0.85), fontsize=9, ha='center', color='red',
                bbox=dict(boxstyle='round', fc='mistyrose', ec='red', alpha=0.7))
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (B) Saturating
    ax = axes[1]
    inv_dp_inf = 1 / 200
    tau = 800
    y_sat = (inv_dp_inf + (inv_dp0 - inv_dp_inf) * np.exp(-t / tau)) * 1e3
    ax.plot(t, y_sat, 'r-', lw=2, label='Saturating model')
    ax.axhline(y=inv_dp_inf * 1e3, color='r', ls='--', alpha=0.5, label='LODP asymptote')
    ax.fill_betweenx([0, 6], 1500, 2500, alpha=0.08, color='green')
    ax.annotate('Physically bounded',
                xy=(2000, 3.8), fontsize=9, ha='center', color='green',
                bbox=dict(boxstyle='round', fc='honeydew', ec='green', alpha=0.7))
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('1/DP × 10³')
    ax.set_title('(B) Saturating model (LODP-aware)')
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 6)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Degradation Model Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'fig1_model_schematic.{ext}')
    plt.close(fig)
    print(f"✓ fig1_model_schematic")


# =====================================================================
# FIG 2 — Forward temporal cross-validation results
# =====================================================================
def fig2_cv_results(paper_dir: Path, outdir: Path):
    import pandas as pd
    csv_path = paper_dir / 'table1_cv_results.csv'
    if not csv_path.exists():
        print(f"⚠  Skipping fig2: {csv_path} not found (run 01_paper_validation.py first)")
        return

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(df))
    w = 0.35

    # (A) MAE comparison
    ax = axes[0]
    ax.bar(x - w / 2, df['MAE_Lin'], w, label='Ekenstam', color='steelblue')
    ax.bar(x + w / 2, df['MAE_Sat'], w, label='Saturating', color='coral')
    ax.set_xlabel('Split Year')
    ax.set_ylabel('MAE (years)')
    ax.set_title('(A) Mean Absolute Error by Split')
    ax.set_xticks(x)
    ax.set_xticklabels(df['split'])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # (B) ΔAICc or delta_pct
    ax = axes[1]
    col = 'delta_pct' if 'delta_pct' in df.columns else 'delta_AICc'
    colors = ['forestgreen' if v > 0 else 'crimson' for v in df[col]]
    ax.bar(x, df[col], color=colors, edgecolor='black', lw=0.5)
    ax.axhline(0, color='black', lw=0.5)
    m_val = df[col].mean()
    ax.axhline(m_val, color='red', ls='--', label=f'Mean = {m_val:.0f}%')
    ax.set_xlabel('Split Year')
    lbl = 'MAE Improvement (%)' if col == 'delta_pct' else 'ΔAICc'
    ax.set_ylabel(lbl)
    ax.set_title(f'(B) Relative Improvement (Saturating vs Linear)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['split'])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'fig2_cv_results.{ext}')
    plt.close(fig)
    print(f"✓ fig2_cv_results")


# =====================================================================
# FIG 3 — Age-dependent bias analysis (Welch t-test)
# =====================================================================
def fig3_bias_analysis(paper_dir: Path, outdir: Path):
    json_path = paper_dir / 'welch_test.json'
    if not json_path.exists():
        print(f"⚠  Skipping fig3: {json_path} not found (run 01_paper_validation.py first)")
        return

    with open(json_path) as f:
        w = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = ['Oldest\n(1850–1880)', 'Newest\n(1970–1990)']
    ek_bias  = [w['bias_ek_oldest'],  w['bias_ek_newest']]
    sat_bias = [w['bias_sat_oldest'], w['bias_sat_newest']]

    x = np.arange(len(bins))
    wd = 0.35
    ax.bar(x - wd / 2, ek_bias, wd, label='Ekenstam', color='steelblue')
    ax.bar(x + wd / 2, sat_bias, wd, label='Saturating', color='coral')
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('Mean Prediction Bias (years)')
    ax.set_title('Age-Dependent Bias in Prediction Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    ax.annotate(
        f'Welch t-test (Ekenstam):\nt = {w["t_ek"]:.2f}, p = {w["p_ek"]:.3f}',
        xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'fig3_bias_analysis.{ext}')
    plt.close(fig)
    print(f"✓ fig3_bias_analysis")


# =====================================================================
# FIG 4 — Fire omission diagram (Fanti discrepancy)
# =====================================================================
def fig4_fire_omission(outdir: Path):
    """The fire omission figure showing how Arrhenius correction bridges the gap."""
    sigma_mpa = 155.0
    y_mech = basso_date(sigma_mpa)           # ≈ 116 AD

    # Deterministic Arrhenius corrections at different T_fire
    dt_h = 4.0  # hours
    temps_C    = [150,  160,  163,  170]
    # Precomputed from Ea=110 kJ/mol to match manuscript caption:
    dates_corr = [596, 1104, 1334, 2084]     # y_mech + t_eq

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Radiocarbon band
    ax.axhspan(C14_MIN, C14_MAX, alpha=0.18, color='green',
               label=f'Radiocarbon 95% CI [{C14_MIN}–{C14_MAX}]')

    # Uncorrected mechanical date
    ax.axhline(y_mech, color='red', ls='--', lw=2,
               label=f'Uncorrected mechanical date ({y_mech:.0f} AD)')

    # Arrow annotations for each temperature
    x_positions = [1, 2, 3, 4]
    colors_bar  = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    for i, (T, yc, xp, c) in enumerate(zip(temps_C, dates_corr, x_positions, colors_bar)):
        ax.bar(xp, yc, width=0.6, color=c, edgecolor='black', lw=0.5, alpha=0.85)
        ax.text(xp, yc + 40, f'{yc} AD', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(xp, -80, f'{T} °C', ha='center', va='top', fontsize=10)

    # Delta annotation
    delta = dates_corr[2] - int(y_mech)
    ax.annotate('', xy=(3.45, dates_corr[2]), xytext=(3.45, y_mech),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(3.7, (dates_corr[2] + y_mech) / 2, f'Δ = {delta} yr',
            ha='left', va='center', fontsize=10, color='purple', fontweight='bold')

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{T} °C\n({dt_h} h)' for T in temps_C])
    ax.set_xlabel('Fire effective temperature (°C) — exposure 4 h', fontsize=11)
    ax.set_ylabel('Corrected date (AD)', fontsize=11)
    ax.set_title('Omission of 1532 fire: Arrhenius correction bridges the\n'
                 f'Fanti–radiocarbon discrepancy (Ea = 110 kJ/mol)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(-200, 2300)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'fig_fanti_fire_omission.{ext}')
    plt.close(fig)
    print(f"✓ fig_fanti_fire_omission")


# =====================================================================
# FIG 5 — 3-panel fire correction article figure
# =====================================================================
def fig5_fire_correction(outdir: Path, n_mc: int = 315_329, seed: int = 1):
    """3-panel: (A) Date vs T_fire curves, (B) conditional y_corr, (C) conditional T_fire."""
    rng = np.random.default_rng(seed)

    # --- Monte Carlo sampling ---
    sigma   = rng.normal(155.0, 15.0, n_mc).clip(1e-6)
    m       = rng.normal(M_BASSO, M_STD, n_mc)
    q       = rng.normal(Q_BASSO, Q_STD, n_mc)
    dt_h    = rng.normal(4.0, 0.5, n_mc).clip(0.5)

    # Jeffreys prior p(T) ∝ 1/T in Kelvin
    T_min_K, T_max_K = 150 + 273.15, 200 + 273.15
    logT    = rng.uniform(np.log(T_min_K), np.log(T_max_K), n_mc)
    T_fire  = np.exp(logT) - 273.15   # °C

    y_mech  = basso_date(sigma, m, q)
    t_eq    = arrhenius_teq_years(dt_h, T_fire)
    y_corr  = y_mech + t_eq

    # Historical constraint
    mask_ok   = y_corr <= Y_FIRE
    n_accept  = int(mask_ok.sum())
    y_acc     = y_corr[mask_ok]

    # C14 subset
    mask_c14  = mask_ok & (y_corr >= C14_MIN) & (y_corr <= C14_MAX)
    T_c14     = T_fire[mask_c14]
    n_c14     = int(mask_c14.sum())

    # Summaries
    med_acc = np.median(y_acc)
    lo_acc, hi_acc = np.percentile(y_acc, [2.5, 97.5])
    p_c14   = 100 * n_c14 / n_accept if n_accept > 0 else 0

    med_Tc14 = np.median(T_c14)
    lo_Tc14, hi_Tc14 = np.percentile(T_c14, [2.5, 97.5])

    # --- Panel A: Deterministic curves ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    T_range = np.linspace(150, 200, 200)
    for dur, ls, c in [(2, ':', '#4e79a7'), (4, '-', '#e15759'), (8, '--', '#59a14f')]:
        y_det = basso_date(155.0) + arrhenius_teq_years(dur, T_range)
        ax.plot(T_range, y_det, ls=ls, color=c, lw=2, label=f'Δt = {dur} h')
    ax.axhspan(C14_MIN, C14_MAX, alpha=0.18, color='green', label='Radiocarbon [1260–1390]')
    ax.axhline(Y_FIRE, color='orange', ls='--', lw=1.5, label='1532 (fire)')
    ax.set_xlabel('Fire temperature (°C)')
    ax.set_ylabel('Corrected date (AD)')
    ax.set_title('(A) Corrected date vs fire temperature')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(150, 200)
    ax.set_ylim(0, 5000)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Conditional y_corr ---
    ax = axes[1]
    ax.hist(y_acc, bins=50, density=True, color='steelblue', edgecolor='black', lw=0.3, alpha=0.8)
    ax.axvline(med_acc, color='red', lw=2, label=f'Median: {med_acc:.0f} AD')
    ax.axvspan(C14_MIN, C14_MAX, alpha=0.18, color='green')
    ax.axvline(Y_FIRE, color='orange', ls='--', lw=1.5)

    textstr = (f'n proposed: {n_mc:,}\n'
               f'n accepted: {n_accept:,} ({100*n_accept/n_mc:.0f}%)\n'
               f'Median: {med_acc:.0f} AD\n'
               f'95% CI: [{lo_acc:.0f}, {hi_acc:.0f}]\n'
               f'P(C14): {p_c14:.0f}%')
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round', fc='#d4f7d4', ec='green'))
    ax.set_xlabel('Corrected date (AD)')
    ax.set_ylabel('Density')
    ax.set_title(r'(B) $p(y_{\mathrm{corr}} \mid y_{\mathrm{corr}} \leq 1532)$')
    ax.set_xlim(-500, 1600)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Conditional T_fire | C14 ---
    ax = axes[2]
    if n_c14 > 10:
        ax.hist(T_c14, bins=30, density=True, color='#e15759', edgecolor='black', lw=0.3, alpha=0.8)
        ax.axvline(med_Tc14, color='darkred', lw=2,
                   label=f'Median: {med_Tc14:.0f} °C')

        textstr_c = (f'n = {n_c14:,}\n'
                     f'Median: {med_Tc14:.0f} °C\n'
                     f'95% CI: [{lo_Tc14:.0f}, {hi_Tc14:.0f}] °C')
        ax.text(0.97, 0.97, textstr_c, transform=ax.transAxes, ha='right', va='top',
                fontsize=8, bbox=dict(boxstyle='round', fc='mistyrose', ec='red'))
    ax.set_xlabel('Fire temperature (°C)')
    ax.set_ylabel('Density')
    ax.set_title(r'(C) $p(T_{\mathrm{fire}} \mid y_{\mathrm{corr}} \in \mathrm{C14})$')
    ax.set_xlim(150, 200)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    note = ('Note: Panel C is a compatibility diagnostic, not a historical reconstruction.\n'
            f'Jeffreys prior p(T) ∝ 1/T (Kelvin). Ea = {E_A/1e3:.0f} kJ/mol.')
    fig.text(0.5, -0.02, note, ha='center', fontsize=8, style='italic', color='grey')

    fig.suptitle('Arrhenius fire correction — Monte Carlo analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'fig_fire_correction_article.{ext}')
    plt.close(fig)
    print(f"✓ fig_fire_correction_article  (n_proposed={n_mc:,}, n_accepted={n_accept:,}, n_c14={n_c14:,})")


# =====================================================================
# FIG 6 — Constraint comparison (unconstrained vs conditional)
# =====================================================================
def fig6_constraint_comparison(outdir: Path, n_mc: int = 315_329, seed: int = 1):
    """Replicates 04_fire_correction_arrhenius_constraint.py output."""
    rng = np.random.default_rng(seed)

    sigma = rng.normal(155.0, 15.0, n_mc).clip(1e-6)
    m     = rng.normal(M_BASSO, M_STD, n_mc)
    q     = rng.normal(Q_BASSO, Q_STD, n_mc)
    dt_h  = rng.normal(4.0, 0.5, n_mc).clip(0.5)

    T_min_K, T_max_K = 150 + 273.15, 200 + 273.15
    logT    = rng.uniform(np.log(T_min_K), np.log(T_max_K), n_mc)
    T_fire  = np.exp(logT) - 273.15

    y_mech = basso_date(sigma, m, q)
    t_eq   = arrhenius_teq_years(dt_h, T_fire)
    y_corr = y_mech + t_eq

    mask_ok  = y_corr <= Y_FIRE
    y_acc    = y_corr[mask_ok]
    n_accept = int(mask_ok.sum())
    acc_rate = n_accept / n_mc

    med_all = np.median(y_corr)
    med_acc = np.median(y_acc)
    lo_acc, hi_acc = np.percentile(y_acc, [2.5, 97.5])
    p_c14   = 100 * np.mean((y_acc >= C14_MIN) & (y_acc <= C14_MAX))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), constrained_layout=True)

    # (A) Unconstrained
    ax = axes[0]
    ax.hist(y_corr, bins=60, density=True, edgecolor='black', lw=0.25)
    ax.axvline(Y_FIRE, color='orange', ls='--', lw=2, label='Fire (1532)')
    ax.axvline(med_all, color='red', lw=2, label=f'Median: {med_all:.0f} AD')
    ax.set_title(f'(A) Unconstrained predictive\n(n = {n_mc:,})')
    ax.set_xlabel('Corrected date (AD)')
    ax.set_ylabel('Density')
    ax.set_xlim(-500, 6000)
    ax.legend(loc='upper right')
    pct_after = 100 * np.mean(y_corr > Y_FIRE)
    ax.text(0.63, 0.85,
            f'{pct_after:.0f}% dates > 1532\n(inconsistent with\nhistorical constraint)',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red'),
            color='red', ha='left', va='top')

    ins = ax.inset_axes([0.56, 0.52, 0.42, 0.36])
    ins.hist(y_corr, bins=45, density=True, edgecolor='black', lw=0.2)
    ins.axvline(Y_FIRE, color='orange', ls='--', lw=1.5)
    ins.set_xlim(0, 1600)
    ins.set_title('Zoom [0–1600]', fontsize=10)

    # (B) Conditional
    ax = axes[1]
    ax.hist(y_acc, bins=35, density=True, edgecolor='black', lw=0.25)
    ax.axvline(Y_FIRE, color='orange', ls='--', lw=2, label='Fire (1532)')
    ax.axvline(med_acc, color='red', lw=2, label=f'Median: {med_acc:.0f} AD')
    ax.set_title(f'(B) Conditional: p(y_corr | y_corr ≤ 1532)\n(rejection sampling, {100*acc_rate:.0f}% acceptance)')
    ax.set_xlabel('Corrected date (AD)')
    ax.set_ylabel('Density')
    ax.set_xlim(-500, 6000)
    ax.legend(loc='upper right')

    ax.text(0.58, 0.86,
            f'Median: {med_acc:.0f} AD\n'
            f'95% interval: [{lo_acc:.0f}, {hi_acc:.0f}]\n'
            f'Rejected: {100*(1-acc_rate):.0f}%\n'
            f'P(C14): {p_c14:.1f}%',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.35', fc='#d4f7d4', ec='green'),
            ha='left', va='top')

    ins = ax.inset_axes([0.55, 0.15, 0.42, 0.36])
    ins.hist(y_acc, bins=40, density=True, edgecolor='black', lw=0.2)
    ins.axvline(Y_FIRE, color='orange', ls='--', lw=1.5)
    ins.axvline(C14_MIN, color='red', ls=':', lw=1.2)
    ins.axvline(C14_MAX, color='red', ls=':', lw=1.2)
    ins.set_xlim(0, 1600)
    ins.set_title('Zoom [0–1600]', fontsize=10)

    fig.suptitle('Effect of historical constraint y_corr ≤ 1532 on Monte Carlo distribution',
                 fontsize=16, fontweight='bold')

    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'fig_constraint_comparison.{ext}')
    plt.close(fig)
    print(f"✓ fig_constraint_comparison  (accept_rate={100*acc_rate:.1f}%)")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate all 6 manuscript figures')
    parser.add_argument('--outdir', default='figures', help='Output directory')
    parser.add_argument('--paper-results', default='out_paper',
                        help='Directory with 01_paper_validation outputs (for fig2, fig3)')
    parser.add_argument('--n-mc', type=int, default=315_329,
                        help='Number of Monte Carlo draws for fig5, fig6')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    outdir    = Path(args.outdir)
    paper_dir = Path(args.paper_results)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {outdir}/")
    print(f"Ea = {E_A/1e3:.0f} kJ/mol (Emsley & Stevens, 1994: 111 ± 6 kJ/mol)")
    print("=" * 65)

    # Standalone figures
    fig1_model_schematic(outdir)
    fig4_fire_omission(outdir)
    fig5_fire_correction(outdir, n_mc=args.n_mc, seed=args.seed)
    fig6_constraint_comparison(outdir, n_mc=args.n_mc, seed=args.seed)

    # Data-dependent figures
    fig2_cv_results(paper_dir, outdir)
    fig3_bias_analysis(paper_dir, outdir)

    print("=" * 65)
    print("Figure map → manuscript:")
    print("  fig1_model_schematic.png        → \\includegraphics Fig.1")
    print("  fig2_cv_results.png             → \\includegraphics Fig.2")
    print("  fig3_bias_analysis.png          → \\includegraphics Fig.3")
    print("  fig_fanti_fire_omission.png     → \\includegraphics Fig.4")
    print("  fig_fire_correction_article.png → \\includegraphics Fig.5")
    print("  fig_constraint_comparison.png   → \\includegraphics Fig.6")


if __name__ == '__main__':
    main()
