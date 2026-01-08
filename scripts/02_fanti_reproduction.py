#!/usr/bin/env python3
"""
Fanti et al. (2015) mechanical dating reproduction and reanalysis.

Reproduces the published regression coefficients from Basso et al. (2015)
and compares linear vs saturating model predictions.

Data source: Basso R, Fanti G, Malfi P (2015) MATEC Web Conf 36:01003

Usage:
    python 02_fanti_reproduction.py --calib data/fanti_basso_2015_calibration.csv \
                                     --ts data/turin_shroud_measurements.csv
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

T_REF = 2000  # Reference year

# =============================================================================
# Data loading
# =============================================================================

def load_calibration_csv(path):
    """Load Fanti calibration data (9 samples)."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    data_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    
    from io import StringIO
    df = pd.read_csv(StringIO('\n'.join(data_lines)))
    df.columns = df.columns.str.strip().str.lower()
    
    return df

def load_ts_measurements(path):
    """Load Turin Shroud measurements from long-format CSV."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    
    measurements = {}
    for line in lines:
        if line.strip().startswith('#') or not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and parts[0].lower() != 'parameter':
            param = parts[0].lower()
            try:
                value = float(parts[1])
                measurements[param] = value
            except:
                pass
    
    return measurements

# =============================================================================
# Models
# =============================================================================

def linear_regression(x, y):
    """Simple linear regression: y = m*x + q"""
    A = np.vstack([x, np.ones_like(x)]).T
    m, q = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(q)

def saturating_model(t, y0, y_inf, tau):
    """Saturating decay: y(t) = y_inf + (y0 - y_inf) * exp(-t/tau)"""
    return y_inf + (y0 - y_inf) * np.exp(-t / tau)

def fit_saturating(age, y, bounds_tight=False):
    """Fit saturating model with appropriate bounds."""
    y_min, y_max = y.min(), y.max()
    
    if bounds_tight:
        # For ln(sigma_R): y0 ~ 7 (modern), y_inf ~ 0-1 (degraded)
        bounds = ([6, -1, 500], [8, 2, 15000])
        p0 = [7.0, 0.5, 3000]
    else:
        bounds = ([y_max * 0.8, y_min * 0.5, 100], [y_max * 1.5, y_max, 20000])
        p0 = [y_max, y_min, 2000]
    
    try:
        popt, _ = curve_fit(saturating_model, age, y, p0=p0, bounds=bounds, maxfev=50000)
        return popt
    except Exception as e:
        print(f"Saturating fit failed: {e}")
        return None

def invert_saturating(y_obs, y0, y_inf, tau, t_max=10000):
    """Invert saturating model to get age from y."""
    if y_obs >= y0:
        return 0.0
    if y_obs <= y_inf:
        return t_max
    ratio = (y_obs - y_inf) / (y0 - y_inf)
    if ratio <= 0 or ratio >= 1:
        return t_max if ratio <= 0 else 0.0
    return -tau * np.log(ratio)

# =============================================================================
# Leave-one-out cross-validation
# =============================================================================

def loo_cv(age, y, ts_y):
    """
    Leave-one-out CV for both linear and saturating models.
    Returns predicted TS dates for each LOO iteration.
    """
    n = len(age)
    loo_linear = []
    loo_sat = []
    
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        age_loo = age[mask]
        y_loo = y[mask]
        
        # Linear (on calendar year)
        year_loo = T_REF - age_loo
        m, q = linear_regression(year_loo, y_loo)
        if m != 0:
            ts_year_lin = (ts_y - q) / m
            loo_linear.append(ts_year_lin)
        
        # Saturating
        popt = fit_saturating(age_loo, y_loo, bounds_tight=True)
        if popt is not None:
            ts_age = invert_saturating(ts_y, *popt)
            ts_year_sat = T_REF - ts_age
            loo_sat.append(ts_year_sat)
    
    return np.array(loo_linear), np.array(loo_sat)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fanti reproduction")
    parser.add_argument("--calib", required=True, help="Calibration CSV")
    parser.add_argument("--ts", required=True, help="TS measurements CSV")
    parser.add_argument("--outdir", default="out_flax", help="Output directory")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    calib = load_calibration_csv(args.calib)
    ts = load_ts_measurements(args.ts)
    
    print(f"Loaded {len(calib)} calibration samples")
    print(f"TS measurements: {list(ts.keys())}")
    
    # Extract variables
    year = calib['date_ad'].values
    age = T_REF - year
    
    # Get ln values (prefer pre-computed if available)
    if 'ln_sigma_r' in calib.columns:
        ln_sigma = calib['ln_sigma_r'].values
    else:
        ln_sigma = np.log(calib['sigma_r_mpa'].values)
    
    if 'ln_e_m' in calib.columns:
        ln_em = calib['ln_e_m'].values
    else:
        ln_em = np.log(calib['e_m_gpa'].values) if 'e_m_gpa' in calib.columns else None
    
    if 'eta_m' in calib.columns:
        eta_m = calib['eta_m'].values
    else:
        eta_m = None
    
    # TS values
    ts_sigma_r = ts.get('sigma_r', 243.22)
    ts_ln_sigma = np.log(ts_sigma_r)
    ts_em = ts.get('e_m', 8.533)
    ts_ln_em = np.log(ts_em)
    ts_eta_m = ts.get('eta_m', 0.0593)
    
    print(f"\nTS measurements:")
    print(f"  σ_R = {ts_sigma_r:.2f} MPa, ln(σ_R) = {ts_ln_sigma:.3f}")
    print(f"  E_m = {ts_em:.3f} GPa, ln(E_m) = {ts_ln_em:.3f}")
    print(f"  η_m = {ts_eta_m:.4f}")
    
    results = {}
    
    # =========================================================================
    # PART 1: Reproduce Basso linear regressions
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: LINEAR REGRESSIONS (Basso et al. 2015 reproduction)")
    print("=" * 70)
    
    # Regression on calendar year (Fanti's approach)
    m_year, q_year = linear_regression(year, ln_sigma)
    ts_year_lin = (ts_ln_sigma - q_year) / m_year
    print(f"\nln(σ_R) vs year: y = {m_year:.7f}×year + {q_year:.4f}")
    print(f"  TS prediction: {ts_year_lin:.0f} AD")
    
    # Regression on age (Basso Table 3 format)
    m_age, q_age = linear_regression(age, ln_sigma)
    print(f"\nln(σ_R) vs age: y = {m_age:.7f}×age + {q_age:.4f}")
    print(f"  Published (Basso Table 3): m = 0.0009256, q = 4.936")
    print(f"  Our reproduction: m = {-m_age:.7f}, q = {q_age:.4f}")
    
    results['linear_sigma_r'] = {
        'ts_year': float(ts_year_lin),
        'slope': float(m_year),
        'intercept': float(q_year)
    }
    
    # Single-variable regressions for all parameters
    print("\n" + "-" * 50)
    print("Single-variable predictions:")
    print("-" * 50)
    
    single_dates = []
    for name, vals, ts_val in [
        ('ln(σ_R)', ln_sigma, ts_ln_sigma),
        ('ln(E_m)', ln_em, ts_ln_em) if ln_em is not None else (None, None, None),
        ('η_m', eta_m, ts_eta_m) if eta_m is not None else (None, None, None)
    ]:
        if name is None:
            continue
        m, q = linear_regression(year, vals)
        ts_yr = (ts_val - q) / m
        single_dates.append(ts_yr)
        print(f"  {name}: {ts_yr:.0f} AD")
        results[f'linear_{name}'] = float(ts_yr)
    
    avg_date = np.mean(single_dates)
    print(f"\n  Average: {avg_date:.0f} AD")
    print(f"  Published (Basso Table 5): 278 AD ± 108 years")
    results['linear_average'] = float(avg_date)
    
    # =========================================================================
    # PART 2: Saturating model
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: SATURATING MODEL")
    print("=" * 70)
    
    popt_sat = fit_saturating(age, ln_sigma, bounds_tight=True)
    if popt_sat is not None:
        y0, y_inf, tau = popt_sat
        print(f"\nSaturating fit on ln(σ_R) vs age:")
        print(f"  y₀ = {y0:.3f} (modern)")
        print(f"  y∞ = {y_inf:.3f} (asymptote)")
        print(f"  τ = {tau:.0f} years")
        
        # Predict TS
        ts_age_sat = invert_saturating(ts_ln_sigma, y0, y_inf, tau)
        ts_year_sat = T_REF - ts_age_sat
        print(f"\n  TS prediction: age = {ts_age_sat:.0f} years → {ts_year_sat:.0f} AD")
        
        results['saturating'] = {
            'ts_year': float(ts_year_sat),
            'y0': float(y0),
            'y_inf': float(y_inf),
            'tau': float(tau)
        }
    else:
        ts_year_sat = np.nan
        results['saturating'] = {'ts_year': None, 'error': 'fit_failed'}
    
    # =========================================================================
    # PART 3: Leave-one-out stability
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 70)
    
    loo_lin, loo_sat = loo_cv(age, ln_sigma, ts_ln_sigma)
    
    print(f"\nLinear LOO (n={len(loo_lin)}):")
    print(f"  Mean: {np.mean(loo_lin):.0f} AD")
    print(f"  Std: {np.std(loo_lin):.0f} years")
    print(f"  Range: {np.min(loo_lin):.0f} – {np.max(loo_lin):.0f} AD")
    
    if len(loo_sat) > 0:
        loo_sat_valid = loo_sat[np.isfinite(loo_sat) & (loo_sat > -5000) & (loo_sat < 5000)]
        print(f"\nSaturating LOO (n={len(loo_sat_valid)}):")
        print(f"  Mean: {np.mean(loo_sat_valid):.0f} AD")
        print(f"  Std: {np.std(loo_sat_valid):.0f} years")
        print(f"  Range: {np.min(loo_sat_valid):.0f} – {np.max(loo_sat_valid):.0f} AD")
    
    results['loo'] = {
        'linear_mean': float(np.mean(loo_lin)),
        'linear_std': float(np.std(loo_lin)),
        'saturating_mean': float(np.mean(loo_sat_valid)) if len(loo_sat) > 0 else None,
        'saturating_std': float(np.std(loo_sat_valid)) if len(loo_sat) > 0 else None
    }
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'TS Date (AD)':<15} {'Source'}")
    print("-" * 70)
    print(f"{'Fanti MLR (published)':<30} {'260':<15} {'Fanti et al. 2015'}")
    print(f"{'Basso σ_R alone (published)':<30} {'603':<15} {'Basso et al. 2015'}")
    print(f"{'Basso 3-param avg (published)':<30} {'278':<15} {'Basso et al. 2015'}")
    print(f"{'Our linear (σ_R)':<30} {f'{ts_year_lin:.0f}':<15} {'This analysis'}")
    print(f"{'Our linear (3-param avg)':<30} {f'{avg_date:.0f}':<15} {'This analysis'}")
    print(f"{'Our saturating':<30} {f'{ts_year_sat:.0f}':<15} {'This analysis'}")
    print(f"{'1988 Radiocarbon':<30} {'1260–1390':<15} {'Damon et al. 1989'}")
    
    # Save results
    with open(outdir / "fanti_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary table
    summary_df = pd.DataFrame([
        {'method': 'Fanti MLR (published)', 'year_AD': 260, 'source': 'Fanti et al. 2015'},
        {'method': 'Basso sigma_R alone', 'year_AD': 603, 'source': 'Basso et al. 2015'},
        {'method': 'Basso 3-param average', 'year_AD': 278, 'source': 'Basso et al. 2015'},
        {'method': 'Linear sigma_R (reproduced)', 'year_AD': ts_year_lin, 'source': 'This analysis'},
        {'method': 'Linear 3-param average', 'year_AD': avg_date, 'source': 'This analysis'},
        {'method': 'Saturating model', 'year_AD': ts_year_sat, 'source': 'This analysis'},
    ])
    summary_df.to_csv(outdir / 'fanti_summary.csv', index=False)
    
    print(f"\n✓ Saved: {outdir}/fanti_results.json")
    print(f"✓ Saved: {outdir}/fanti_summary.csv")

if __name__ == "__main__":
    main()
