#!/usr/bin/env python3
"""
03_fanti_saturating_reanalysis.py

Complete reanalysis of Fanti et al. mechanical dating method using:
1. Original MLR approach (showing multicollinearity issues)
2. Univariate linear regressions
3. Saturating (first-order kinetics with asymptote) model

Data sources:
- Fanti & Basso 2017, Int. J. Rel. Qual. Saf. Eng. 24(2), Table 2
- Basso et al. 2015, MATEC Web Conf. 36, 01003

Usage:
    python 03_fanti_saturating_reanalysis.py --outdir out_fanti_reanalysis
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# CALIBRATION DATA (Fanti & Basso 2017, Table 2)
# =============================================================================
CALIBRATION_DATA = {
    'sample': ['B', 'DII', 'D', 'FII', 'NII', 'E', 'HII', 'K', 'LII'],
    'date_ad': np.array([2000, 1072, 575, 65, -290, -375, -860, -2652, -3250], dtype=float),
    'sigma_r': np.array([1076, 678, 63.2, 150, 119, 140, 44.1, 58.9, 2.11]),  # MPa
    'Ef': np.array([24.8, 19.0, 4.20, 7.38, 4.55, 4.34, 3.96, 5.37, 0.184]),  # GPa
    'Ei': np.array([32.2, 23.3, 5.36, 9.67, 6.88, 2.98, 7.51, 6.39, 0.50]),   # GPa
    'eta_d': np.array([4.8, 5.3, 7.4, 7.9, 8.0, 8.5, 8.4, 9.6, 12.8]),        # %
    'eta_i': np.array([1.6, 3.3, 5.2, 3.7, 4.6, 3.3, 5.5, 7.0, 8.2])          # %
}

# TS measurements (Basso et al. 2015 - average of 8 fibers)
TS_MEASUREMENTS = {
    'sigma_r': 243.22,  # MPa
    'Ef': 11.58,        # GPa
    'Ei': 15.68,        # GPa
    'eta_d': 5.49,      # %
    'eta_i': 3.12       # %
}

# Measurement uncertainties (estimated ~15-20%)
TS_UNCERTAINTIES = {
    'sigma_r': 50,
    'Ef': 2.3,
    'Ei': 3.1,
    'eta_d': 0.8,
    'eta_i': 0.5
}

RADIOCARBON = {'mean': 1325, 'ci_lower': 1260, 'ci_upper': 1390}


def saturating_model(t, y0, y_inf, tau):
    """y(t) = y_inf + (y0 - y_inf) * exp(-t/tau)"""
    return y_inf + (y0 - y_inf) * np.exp(-t / tau)


def saturating_inverse(y_obs, y0, y_inf, tau):
    """Solve for age given observed value"""
    if y0 == y_inf or tau <= 0:
        return np.nan
    ratio = (y_obs - y_inf) / (y0 - y_inf)
    if ratio <= 0 or ratio >= 1:
        return np.nan
    return -tau * np.log(ratio)


def compute_vif(X):
    """Compute Variance Inflation Factors"""
    from numpy.linalg import inv
    try:
        corr_matrix = np.corrcoef(X.T)
        vif = np.diag(inv(corr_matrix))
        return vif
    except:
        return np.full(X.shape[1], np.nan)


def fit_mlr(data, ts):
    """Fit multivariate linear regression (Fanti's original approach)"""
    X = np.column_stack([
        np.log(data['sigma_r']),
        np.log(data['Ef']),
        np.log(data['Ei']),
        data['eta_d'],
        data['eta_i']
    ])
    y = data['date_ad']
    
    # Fit using normal equations
    X_with_const = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    
    # VIF
    vif = compute_vif(X)
    
    # Predict TS
    ts_features = np.array([1, np.log(ts['sigma_r']), np.log(ts['Ef']), 
                            np.log(ts['Ei']), ts['eta_d'], ts['eta_i']])
    ts_date = np.dot(ts_features, beta)
    
    # R²
    y_pred = np.dot(X_with_const, beta)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    
    return {
        'coefficients': beta,
        'vif': dict(zip(['ln_sigma', 'ln_Ef', 'ln_Ei', 'eta_d', 'eta_i'], vif)),
        'r2': r2,
        'ts_date': ts_date
    }


def fit_univariate_linear(data, ts):
    """Fit individual univariate regressions"""
    results = {}
    ages = 2000 - data['date_ad']
    
    for param in ['sigma_r', 'Ef', 'Ei', 'eta_d', 'eta_i']:
        is_log = param not in ['eta_d', 'eta_i']
        
        if is_log:
            x = np.log(data[param])
            ts_x = np.log(ts[param])
        else:
            x = data[param]
            ts_x = ts[param]
        
        slope, intercept, r, _, stderr = stats.linregress(x, data['date_ad'])
        ts_date = slope * ts_x + intercept
        
        results[param] = {
            'slope': slope,
            'intercept': intercept,
            'r': r,
            'r2': r**2,
            'ts_date': ts_date
        }
    
    # Mean estimate
    dates = [r['ts_date'] for r in results.values()]
    results['mean'] = np.mean(dates)
    results['std'] = np.std(dates)
    
    return results


def fit_saturating(data, ts):
    """Fit saturating models to each parameter"""
    results = {}
    ages = 2000 - data['date_ad']
    
    for param in ['sigma_r', 'Ef', 'Ei', 'eta_d', 'eta_i']:
        is_log = param not in ['eta_d', 'eta_i']
        
        if is_log:
            y = np.log(data[param])
            ts_y = np.log(ts[param])
        else:
            y = data[param]
            ts_y = ts[param]
        
        # Sort by age
        idx = np.argsort(ages)
        ages_sorted = ages[idx]
        y_sorted = y[idx]
        
        try:
            # Initial guesses
            y0_guess = y_sorted[0]
            y_inf_guess = y_sorted[-1]
            
            popt, pcov = curve_fit(
                saturating_model, ages_sorted, y_sorted,
                p0=[y0_guess, y_inf_guess, 3000],
                bounds=([-100, -100, 100], [100, 100, 200000]),
                maxfev=10000
            )
            y0, y_inf, tau = popt
            
            # R²
            y_pred = saturating_model(ages, y0, y_inf, tau)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res / ss_tot
            
            # TS date
            ts_age = saturating_inverse(ts_y, y0, y_inf, tau)
            ts_date = 2000 - ts_age if not np.isnan(ts_age) else np.nan
            
            results[param] = {
                'y0': y0,
                'y_inf': y_inf,
                'tau': tau,
                'r2': r2,
                'ts_date': ts_date
            }
            
        except Exception as e:
            results[param] = {'error': str(e), 'ts_date': np.nan}
    
    # Mean estimate
    valid_dates = [r['ts_date'] for r in results.values() 
                   if isinstance(r.get('ts_date'), (int, float)) and not np.isnan(r.get('ts_date', np.nan))]
    results['mean'] = np.mean(valid_dates) if valid_dates else np.nan
    results['std'] = np.std(valid_dates) if valid_dates else np.nan
    
    return results


def monte_carlo_analysis(data, ts, ts_unc, n_mc=10000):
    """Monte Carlo uncertainty propagation"""
    linear_dates = []
    saturating_dates = []
    
    for _ in range(n_mc):
        # Sample TS measurements with noise
        ts_mc = {param: np.random.normal(ts[param], ts_unc[param]) 
                 for param in ts}
        
        # Linear
        lin_results = fit_univariate_linear(data, ts_mc)
        linear_dates.append(lin_results['mean'])
        
        # Saturating
        sat_results = fit_saturating(data, ts_mc)
        if not np.isnan(sat_results['mean']):
            saturating_dates.append(sat_results['mean'])
    
    return {
        'linear': {
            'mean': np.mean(linear_dates),
            'std': np.std(linear_dates),
            'ci_lower': np.percentile(linear_dates, 2.5),
            'ci_upper': np.percentile(linear_dates, 97.5)
        },
        'saturating': {
            'mean': np.mean(saturating_dates),
            'std': np.std(saturating_dates),
            'ci_lower': np.percentile(saturating_dates, 2.5),
            'ci_upper': np.percentile(saturating_dates, 97.5)
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default='out_fanti_reanalysis')
    parser.add_argument('--n-mc', type=int, default=10000)
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    data = CALIBRATION_DATA
    ts = TS_MEASUREMENTS
    
    print("="*70)
    print("FANTI MECHANICAL DATING - SATURATING MODEL REANALYSIS")
    print("="*70)
    
    # 1. MLR (original Fanti approach)
    print("\n1. Multivariate Linear Regression (MLR)")
    print("-"*50)
    mlr_results = fit_mlr(data, ts)
    print(f"   VIF values (>10 = severe multicollinearity):")
    for param, vif in mlr_results['vif'].items():
        status = "⚠️ SEVERE" if vif > 10 else "OK"
        print(f"      {param}: {vif:.1f} ({status})")
    print(f"   R² = {mlr_results['r2']:.4f}")
    print(f"   TS Date: {mlr_results['ts_date']:.0f} AD")
    
    # 2. Univariate linear
    print("\n2. Univariate Linear Regressions")
    print("-"*50)
    lin_results = fit_univariate_linear(data, ts)
    for param in ['sigma_r', 'Ef', 'Ei', 'eta_d', 'eta_i']:
        r = lin_results[param]
        print(f"   {param}: R²={r['r2']:.3f}, TS={r['ts_date']:.0f} AD")
    print(f"   Mean: {lin_results['mean']:.0f} AD (std={lin_results['std']:.0f})")
    
    # 3. Saturating
    print("\n3. Saturating Model")
    print("-"*50)
    sat_results = fit_saturating(data, ts)
    for param in ['sigma_r', 'Ef', 'Ei', 'eta_d', 'eta_i']:
        r = sat_results[param]
        if 'tau' in r:
            print(f"   {param}: τ={r['tau']:.0f} yrs, R²={r['r2']:.3f}, TS={r['ts_date']:.0f} AD")
        else:
            print(f"   {param}: {r.get('error', 'N/A')}")
    print(f"   Mean: {sat_results['mean']:.0f} AD (std={sat_results['std']:.0f})")
    
    # 4. Monte Carlo
    print(f"\n4. Monte Carlo Analysis (n={args.n_mc})")
    print("-"*50)
    mc_results = monte_carlo_analysis(data, ts, TS_UNCERTAINTIES, args.n_mc)
    
    print(f"   Linear:     {mc_results['linear']['mean']:.0f} AD "
          f"[{mc_results['linear']['ci_lower']:.0f}, {mc_results['linear']['ci_upper']:.0f}]")
    print(f"   Saturating: {mc_results['saturating']['mean']:.0f} AD "
          f"[{mc_results['saturating']['ci_lower']:.0f}, {mc_results['saturating']['ci_upper']:.0f}]")
    
    # 5. Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    rc = RADIOCARBON['mean']
    
    summary = {
        'mlr': {
            'date': mlr_results['ts_date'],
            'gap_to_c14': rc - mlr_results['ts_date'],
            'note': 'Severe multicollinearity (all VIF > 10)'
        },
        'linear_univariate': {
            'date': mc_results['linear']['mean'],
            'ci_lower': mc_results['linear']['ci_lower'],
            'ci_upper': mc_results['linear']['ci_upper'],
            'gap_to_c14': rc - mc_results['linear']['mean']
        },
        'saturating': {
            'date': mc_results['saturating']['mean'],
            'ci_lower': mc_results['saturating']['ci_lower'],
            'ci_upper': mc_results['saturating']['ci_upper'],
            'gap_to_c14': rc - mc_results['saturating']['mean']
        },
        'radiocarbon': RADIOCARBON
    }
    
    print(f"\n{'Method':<25} {'Date (AD)':<12} {'95% CI':<20} {'Gap to C14'}")
    print("-"*70)
    print(f"{'MLR (Fanti)':<25} {summary['mlr']['date']:>8.0f}     {'N/A':<20} {summary['mlr']['gap_to_c14']:>+.0f} years")
    print(f"{'Linear univariate':<25} {summary['linear_univariate']['date']:>8.0f}     "
          f"[{summary['linear_univariate']['ci_lower']:.0f}, {summary['linear_univariate']['ci_upper']:.0f}]"
          f"{'':>8} {summary['linear_univariate']['gap_to_c14']:>+.0f} years")
    print(f"{'Saturating':<25} {summary['saturating']['date']:>8.0f}     "
          f"[{summary['saturating']['ci_lower']:.0f}, {summary['saturating']['ci_upper']:.0f}]"
          f"{'':>8} {summary['saturating']['gap_to_c14']:>+.0f} years")
    print(f"{'Radiocarbon (1988)':<25} {rc:>8}     [1260, 1390]{'':>8} ---")
    
    # Save results
    with open(outdir / 'fanti_reanalysis.json', 'w') as f:
        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        json.dump(convert(summary), f, indent=2)
    
    print(f"\n✓ Results saved to {outdir}/fanti_reanalysis.json")


if __name__ == '__main__':
    main()
