#!/usr/bin/env python3
"""
Paper temporal validation - Strlič et al. (2020) database.

Compares Ekenstam (linear) vs Saturating degradation models using
forward temporal cross-validation (train on older samples, test on newer).

Models (in 1/DP space):
- Ekenstam: 1/DP(t) = 1/DP₀ + k·t                              [2 params]
- Saturating: 1/DP(t) = 1/DP∞ + (1/DP₀ - 1/DP∞)·exp(-t/τ)      [3 params]

Convention: t = age = T_REF - calendar_year

Usage:
    python 01_paper_validation.py --xlsx data/10570_2020_3344_MOESM2_ESM.xlsx
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

T_REF = 2000  # Reference year for age calculation

# =============================================================================
# Models
# =============================================================================

def ekenstam(t, inv_dp0, k):
    """Ekenstam: 1/DP(t) = 1/DP₀ + k·t"""
    return inv_dp0 + k * t

def saturating(t, inv_dp0, inv_dp_inf, tau):
    """Saturating: 1/DP(t) = 1/DP∞ + (1/DP₀ - 1/DP∞)·exp(-t/τ)"""
    return inv_dp_inf + (inv_dp0 - inv_dp_inf) * np.exp(-t / tau)

def fit_ekenstam(t, y):
    """Fit Ekenstam model via OLS."""
    A = np.vstack([t, np.ones_like(t)]).T
    k, inv_dp0 = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.array([max(inv_dp0, 1e-8), max(k, 1e-10)])

def fit_saturating(t, y):
    """Fit saturating model via NLS."""
    y_min, y_max = y.min(), y.max()
    try:
        popt, _ = curve_fit(
            saturating, t, y,
            p0=[y_min * 0.5, y_max * 1.3, 80],
            bounds=([0, y_max * 0.5, 10], [y_min * 3, y_max * 5, 1000]),
            maxfev=50000
        )
        return popt
    except:
        return np.array([y_min * 0.5, y_max * 1.3, 80])

def invert_ekenstam(inv_dp_obs, inv_dp0, k):
    """Invert Ekenstam to get age from 1/DP."""
    if k <= 0:
        return np.nan
    return max(0, (inv_dp_obs - inv_dp0) / k)

def invert_saturating(inv_dp_obs, inv_dp0, inv_dp_inf, tau, t_max=500):
    """Invert saturating model to get age from 1/DP."""
    if inv_dp_obs <= inv_dp0:
        return 0.0
    if inv_dp_obs >= inv_dp_inf:
        return t_max
    ratio = (inv_dp_obs - inv_dp_inf) / (inv_dp0 - inv_dp_inf)
    if ratio <= 0 or ratio >= 1:
        return t_max if ratio <= 0 else 0.0
    return -tau * np.log(ratio)

# =============================================================================
# Metrics
# =============================================================================

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def aicc(n, rss, k):
    """Corrected AIC."""
    if n <= k + 1 or rss <= 0:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)

def bootstrap_ci(arr, n_boot=1000, ci=95):
    """Bootstrap confidence interval for mean."""
    np.random.seed(42)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) 
             for _ in range(n_boot)]
    lo = (100 - ci) / 2
    hi = 100 - lo
    return np.percentile(means, [lo, hi])

# =============================================================================
# Data loading
# =============================================================================

def load_strlic_data(xlsx_path, year_min=1850, year_max=1990):
    """Load and filter Strlič et al. (2020) ESM data."""
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    df.columns = df.columns.str.strip()
    
    mask = (
        df['class'].str.lower().str.contains('bleach', na=False) &
        (df['dat'] >= year_min) & (df['dat'] <= year_max) &
        (df['dp'] > 0)
    )
    subset = df[mask].copy()
    subset = subset.rename(columns={'dat': 'year', 'dp': 'DP'})
    subset['inv_DP'] = 1.0 / subset['DP']
    subset['age'] = T_REF - subset['year']
    
    return subset

# =============================================================================
# Validation protocols
# =============================================================================

def forward_temporal_cv(data, split_years):
    """
    Forward temporal CV: train on year < split, test on year >= split.
    This tests extrapolation to NEWER (less degraded) samples.
    """
    results = []
    
    for split_year in split_years:
        train = data[data['year'] < split_year]
        test = data[data['year'] >= split_year]
        
        if len(train) < 20 or len(test) < 20:
            continue
        
        # Fit on training data
        t_train = train['age'].values
        y_train = train['inv_DP'].values
        popt_ek = fit_ekenstam(t_train, y_train)
        popt_sat = fit_saturating(t_train, y_train)
        
        # Test data
        t_test = test['age'].values
        y_test = test['inv_DP'].values
        years_test = test['year'].values
        
        # Forward prediction (1/DP)
        y_pred_ek = ekenstam(t_test, *popt_ek)
        y_pred_sat = saturating(t_test, *popt_sat)
        
        # Inverse prediction (calendar year)
        age_pred_ek = np.array([invert_ekenstam(y, *popt_ek) for y in y_test])
        age_pred_sat = np.array([invert_saturating(y, *popt_sat) for y in y_test])
        year_pred_ek = T_REF - age_pred_ek
        year_pred_sat = T_REF - age_pred_sat
        
        # Errors (in years)
        errors_ek = year_pred_ek - years_test
        errors_sat = year_pred_sat - years_test
        
        # Metrics
        mae_ek = mae(years_test, year_pred_ek)
        mae_sat = mae(years_test, year_pred_sat)
        rmse_ek = rmse(years_test, year_pred_ek)
        rmse_sat = rmse(years_test, year_pred_sat)
        
        # AICc (on 1/DP)
        rss_ek = np.sum((y_pred_ek - y_test) ** 2)
        rss_sat = np.sum((y_pred_sat - y_test) ** 2)
        aicc_ek = aicc(len(test), rss_ek, k=2)
        aicc_sat = aicc(len(test), rss_sat, k=3)
        
        delta_pct = 100 * (mae_ek - mae_sat) / mae_ek if mae_ek > 0 else 0
        
        results.append({
            'split': split_year,
            'n_train': len(train),
            'n_test': len(test),
            'MAE_ek': mae_ek,
            'MAE_sat': mae_sat,
            'delta_pct': delta_pct,
            'RMSE_ek': rmse_ek,
            'RMSE_sat': rmse_sat,
            'delta_AICc': aicc_ek - aicc_sat,
            'errors_ek': errors_ek,
            'errors_sat': errors_sat,
            'bias_ek': np.mean(errors_ek),
            'bias_sat': np.mean(errors_sat)
        })
    
    return results

def welch_test_bias(data):
    """
    Welch test for age-dependent bias in Ekenstam model.
    Train on middle bins, test on oldest vs newest bins.
    """
    oldest = data[(data['year'] >= 1850) & (data['year'] < 1880)]
    newest = data[(data['year'] >= 1970) & (data['year'] <= 1990)]
    middle = data[(data['year'] >= 1880) & (data['year'] < 1970)]
    
    if len(oldest) < 10 or len(newest) < 10 or len(middle) < 50:
        return None
    
    # Fit on middle bins
    t_train = middle['age'].values
    y_train = middle['inv_DP'].values
    popt_ek = fit_ekenstam(t_train, y_train)
    popt_sat = fit_saturating(t_train, y_train)
    
    def get_errors(subset, popt_ek, popt_sat):
        y = subset['inv_DP'].values
        years = subset['year'].values
        
        age_pred_ek = np.array([invert_ekenstam(yi, *popt_ek) for yi in y])
        age_pred_sat = np.array([invert_saturating(yi, *popt_sat) for yi in y])
        
        year_pred_ek = T_REF - age_pred_ek
        year_pred_sat = T_REF - age_pred_sat
        
        return year_pred_ek - years, year_pred_sat - years
    
    err_ek_old, err_sat_old = get_errors(oldest, popt_ek, popt_sat)
    err_ek_new, err_sat_new = get_errors(newest, popt_ek, popt_sat)
    
    t_ek, p_ek = stats.ttest_ind(err_ek_old, err_ek_new, equal_var=False)
    t_sat, p_sat = stats.ttest_ind(err_sat_old, err_sat_new, equal_var=False)
    
    return {
        'n_oldest': len(oldest),
        'n_newest': len(newest),
        'n_middle': len(middle),
        'bias_ek_oldest': float(np.mean(err_ek_old)),
        'bias_ek_newest': float(np.mean(err_ek_new)),
        'bias_sat_oldest': float(np.mean(err_sat_old)),
        'bias_sat_newest': float(np.mean(err_sat_new)),
        't_ek': float(t_ek),
        'p_ek': float(p_ek),
        't_sat': float(t_sat),
        'p_sat': float(p_sat),
        'err_ek_old': err_ek_old,
        'err_ek_new': err_ek_new
    }

# =============================================================================
# Output generation
# =============================================================================

def generate_latex_table(results, welch):
    """Generate LaTeX Table 1."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Forward temporal cross-validation results on the Strlič et al. (2020) paper database. Lin = Ekenstam linear model; Sat = Saturating model. MAE and RMSE in years. $\Delta$\% = relative MAE improvement [(Lin$-$Sat)/Lin]. $\Delta$AICc $>0$ favors the saturating model.}")
    lines.append(r"\label{tab:cv-results}")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Split} & \textbf{$n$} & \textbf{MAE Lin} & \textbf{MAE Sat} & \textbf{$\Delta$\%} & \textbf{RMSE Lin} & \textbf{RMSE Sat} & \textbf{$\Delta$AICc} \\")
    lines.append(r"\midrule")
    
    for r in results:
        lines.append(f"{r['split']} & {r['n_test']} & {r['MAE_ek']:.1f} & {r['MAE_sat']:.1f} & {r['delta_pct']:.0f}\\% & {r['RMSE_ek']:.1f} & {r['RMSE_sat']:.1f} & {r['delta_AICc']:.1f} \\\\")
    
    # Mean row
    mean_mae_ek = np.mean([r['MAE_ek'] for r in results])
    mean_mae_sat = np.mean([r['MAE_sat'] for r in results])
    mean_rmse_ek = np.mean([r['RMSE_ek'] for r in results])
    mean_rmse_sat = np.mean([r['RMSE_sat'] for r in results])
    mean_aicc = np.mean([r['delta_AICc'] for r in results])
    mean_pct = 100 * (mean_mae_ek - mean_mae_sat) / mean_mae_ek
    total_n = sum(r['n_test'] for r in results)
    
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Mean}} & {total_n} & \\textbf{{{mean_mae_ek:.1f}}} & \\textbf{{{mean_mae_sat:.1f}}} & \\textbf{{{mean_pct:.0f}\\%}} & \\textbf{{{mean_rmse_ek:.1f}}} & \\textbf{{{mean_rmse_sat:.1f}}} & \\textbf{{{mean_aicc:.1f}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper temporal validation")
    parser.add_argument("--xlsx", required=True, help="Strlič ESM xlsx file")
    parser.add_argument("--outdir", default="out_paper", help="Output directory")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_strlic_data(args.xlsx)
    print(f"Loaded {len(data)} samples (bleached, 1850-1990)")
    print(f"Age range: {data['age'].min():.0f} – {data['age'].max():.0f} years")
    print(f"1/DP range: {data['inv_DP'].min():.5f} – {data['inv_DP'].max():.5f}")
    
    # Forward CV
    split_years = [1910, 1920, 1930, 1940, 1950]
    results = forward_temporal_cv(data, split_years)
    
    # Print results
    print("\n" + "=" * 80)
    print("TABLE 1: Forward temporal cross-validation results")
    print("=" * 80)
    print(f"{'Split':>6} {'n_tr':>5} {'n_te':>5} {'MAE Lin':>8} {'MAE Sat':>8} {'Δ%':>6} {'RMSE Lin':>9} {'RMSE Sat':>9} {'ΔAICc':>7}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['split']:>6} {r['n_train']:>5} {r['n_test']:>5} "
              f"{r['MAE_ek']:>8.1f} {r['MAE_sat']:>8.1f} {r['delta_pct']:>5.0f}% "
              f"{r['RMSE_ek']:>9.1f} {r['RMSE_sat']:>9.1f} {r['delta_AICc']:>7.1f}")
    
    # Summary
    mean_mae_ek = np.mean([r['MAE_ek'] for r in results])
    mean_mae_sat = np.mean([r['MAE_sat'] for r in results])
    mean_rmse_ek = np.mean([r['RMSE_ek'] for r in results])
    mean_rmse_sat = np.mean([r['RMSE_sat'] for r in results])
    mean_aicc = np.mean([r['delta_AICc'] for r in results])
    mean_pct = 100 * (mean_mae_ek - mean_mae_sat) / mean_mae_ek
    
    print("-" * 80)
    print(f"{'Mean':>6} {'':>5} {'':>5} {mean_mae_ek:>8.1f} {mean_mae_sat:>8.1f} {mean_pct:>5.0f}% "
          f"{mean_rmse_ek:>9.1f} {mean_rmse_sat:>9.1f} {mean_aicc:>7.1f}")
    
    # Welch test
    welch = welch_test_bias(data)
    if welch:
        print("\n" + "-" * 60)
        print("WELCH'S T-TEST: Age-dependent bias in Ekenstam model")
        print("-" * 60)
        print(f"Training: middle bins 1880-1970 (n={welch['n_middle']})")
        print(f"Test oldest (1850-1880, n={welch['n_oldest']}): Ek bias = {welch['bias_ek_oldest']:+.1f} years")
        print(f"Test newest (1970-1990, n={welch['n_newest']}): Ek bias = {welch['bias_ek_newest']:+.1f} years")
        print(f"Ekenstam: t = {welch['t_ek']:.3f}, p = {welch['p_ek']:.4f}")
        print(f"Saturating: t = {welch['t_sat']:.3f}, p = {welch['p_sat']:.4f}")
        
        # Bootstrap CI
        ci_old = bootstrap_ci(welch['err_ek_old'])
        ci_new = bootstrap_ci(welch['err_ek_new'])
        print(f"\nEkenstam bootstrap 95% CI:")
        print(f"  Oldest bin: ({ci_old[0]:+.1f}, {ci_old[1]:+.1f}) years")
        print(f"  Newest bin: ({ci_new[0]:+.1f}, {ci_new[1]:+.1f}) years")
        
        # CI excludes zero?
        ci_excludes_zero = ci_old[0] > 0 or ci_old[1] < 0
        print(f"  Oldest CI excludes zero: {ci_excludes_zero}")
    
    # Save CSV
    df_out = pd.DataFrame([{
        'split': r['split'], 'n_train': r['n_train'], 'n_test': r['n_test'],
        'MAE_Lin': r['MAE_ek'], 'MAE_Sat': r['MAE_sat'], 'delta_pct': r['delta_pct'],
        'RMSE_Lin': r['RMSE_ek'], 'RMSE_Sat': r['RMSE_sat'], 'delta_AICc': r['delta_AICc'],
        'Bias_Lin': r['bias_ek'], 'Bias_Sat': r['bias_sat']
    } for r in results])
    df_out.to_csv(outdir / "table1_cv_results.csv", index=False)
    
    # Save LaTeX table
    latex_table = generate_latex_table(results, welch)
    (outdir / "table1.tex").write_text(latex_table, encoding="utf-8")
    
    # Save Welch results
    if welch:
        welch_export = {k: v for k, v in welch.items() if not isinstance(v, np.ndarray)}
        with open(outdir / "welch_test.json", "w") as f:
            json.dump(welch_export, f, indent=2)
    
    # Save summary
    summary = {
        'n_samples': len(data),
        'n_splits': len(results),
        'mean_MAE_Lin': mean_mae_ek,
        'mean_MAE_Sat': mean_mae_sat,
        'mean_improvement_pct': mean_pct,
        'mean_delta_AICc': mean_aicc,
        'welch_p_value': welch['p_ek'] if welch else None
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved: {outdir}/table1_cv_results.csv")
    print(f"✓ Saved: {outdir}/table1.tex")
    print(f"✓ Saved: {outdir}/welch_test.json")
    print(f"✓ Saved: {outdir}/summary.json")

if __name__ == "__main__":
    main()
