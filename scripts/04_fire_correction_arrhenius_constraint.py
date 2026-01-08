"""Monte Carlo fire-correction with explicit pre-1532 (historical) constraint.

This script complements `04_fire_correction_arrhenius.py` by making the
conditioning step explicit:

  - Unconditional predictive distribution: p(y_corr)
  - Conditional distribution (historical constraint): p(y_corr | y_corr <= 1532)

The plots are meant as *diagnostic checks*, not as a reconstruction of the
1532 fire temperature/time profile.

Outputs
-------
- fig_constraint_comparison.png
- fire_correction_results_with_constraint.json

Usage
-----
python 04_fire_correction_arrhenius_constraint.py --n 100000 --seed 1

Notes
-----
Fire-temperature prior:
- `jeffreys_K` implements a log-uniform prior on absolute temperature (Kelvin):
  p(T) ∝ 1/T on [Tmin,Tmax], sampled by uniform log(T).
- `uniform_invT_K` samples uniformly in 1/T (Kelvin), which yields p(T) ∝ 1/T^2.
  Use only if you *intend* that prior (it is NOT Jeffreys 1/T).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Model constants (matching your original script / paper defaults)
# -----------------------------------------------------------------------------

# Basso calibration: ln(sigma_R [MPa]) = m * y + q  =>  y = (ln(sigma_R)-q)/m
M_BASSO = 9.256e-4
Q_BASSO = 4.936
M_STD = 0.5e-4
Q_STD = 0.05

# Arrhenius parameters
R_GAS = 8.314  # J/mol/K
E_A = 110e3    # J/mol
T_ROOM_C = 20.0

# Historical constraint
Y_FIRE = 1532

# C14 interval used in your figures
C14_MIN = 1260
C14_MAX = 1390


def mechanical_date_basso(sigma_R_mpa: np.ndarray, m: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Invert Basso calibration (vectorized)."""
    return (np.log(sigma_R_mpa) - q) / m


def arrhenius_equivalent_years(dt_hours: np.ndarray, T_fire_C: np.ndarray, Ea_J_mol: float = E_A,
                              T_room_C: float = T_ROOM_C) -> np.ndarray:
    """Equivalent aging time at room temperature, in years.

    k(T) ∝ exp(-Ea/RT)
    dt_fire * k(T_fire) = t_eq * k(T_room)
    => t_eq = dt_fire * exp(Ea/R * (1/T_room - 1/T_fire))
    """
    T_fire_K = T_fire_C + 273.15
    T_room_K = T_room_C + 273.15
    dt_seconds = dt_hours * 3600.0
    factor = np.exp(Ea_J_mol / R_GAS * (1.0 / T_room_K - 1.0 / T_fire_K))
    t_eq_seconds = dt_seconds * factor
    return t_eq_seconds / (3600.0 * 24.0 * 365.25)


FirePrior = Literal["jeffreys_K", "uniform_T_C", "uniform_invT_K"]


@dataclass(frozen=True)
class MCConfig:
    n: int = 100_000
    seed: int = 1

    # Mechanical input
    sigma_mean_mpa: float = 155.0
    sigma_sd_mpa: float = 15.0
    m_mean: float = M_BASSO
    m_sd: float = M_STD
    q_mean: float = Q_BASSO
    q_sd: float = Q_STD

    # Fire model
    T_min_C: float = 150.0
    T_max_C: float = 200.0
    duration_mean_h: float = 4.0
    duration_sd_h: float = 0.5
    fire_prior: FirePrior = "jeffreys_K"

    # Historical constraint
    y_fire: int = Y_FIRE


def sample_T_fire_C(cfg: MCConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample fire temperature in °C according to cfg.fire_prior."""
    TminC, TmaxC = cfg.T_min_C, cfg.T_max_C

    if cfg.fire_prior == "uniform_T_C":
        return rng.uniform(TminC, TmaxC, size=cfg.n)

    TminK, TmaxK = TminC + 273.15, TmaxC + 273.15

    if cfg.fire_prior == "jeffreys_K":
        # Jeffreys prior p(T) ∝ 1/T: sample log(T) uniformly
        logT = rng.uniform(np.log(TminK), np.log(TmaxK), size=cfg.n)
        T_K = np.exp(logT)
        return T_K - 273.15

    if cfg.fire_prior == "uniform_invT_K":
        # Uniform in 1/T_K -> p(T) ∝ 1/T^2
        invT = rng.uniform(1.0 / TmaxK, 1.0 / TminK, size=cfg.n)
        T_K = 1.0 / invT
        return T_K - 273.15

    raise ValueError(f"Unknown fire_prior: {cfg.fire_prior}")


def run_monte_carlo(cfg: MCConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)

    # Sample uncertain inputs
    sigma = rng.normal(cfg.sigma_mean_mpa, cfg.sigma_sd_mpa, size=cfg.n)
    sigma = np.clip(sigma, 1e-6, None)  # avoid non-physical negatives

    m = rng.normal(cfg.m_mean, cfg.m_sd, size=cfg.n)
    q = rng.normal(cfg.q_mean, cfg.q_sd, size=cfg.n)

    dt_h = rng.normal(cfg.duration_mean_h, cfg.duration_sd_h, size=cfg.n)
    dt_h = np.maximum(0.5, dt_h)  # enforce positive exposure duration

    T_fire = sample_T_fire_C(cfg, rng)

    # Compute corrected date
    y_mech = mechanical_date_basso(sigma, m=m, q=q)
    t_eq = arrhenius_equivalent_years(dt_h, T_fire)
    y_corr = y_mech + t_eq

    # Historical constraint (post-filtering == rejection sampling)
    accept_mask = y_corr <= cfg.y_fire
    y_acc = y_corr[accept_mask]
    accept_rate = float(np.mean(accept_mask))

    # C14 probabilities (unconditional and conditional)
    c14_mask = (y_corr >= C14_MIN) & (y_corr <= C14_MAX)
    p_c14 = float(np.mean(c14_mask))
    p_c14_cond = float(np.mean((y_acc >= C14_MIN) & (y_acc <= C14_MAX))) if y_acc.size else float("nan")

    def qntl(x: np.ndarray, ps=(0.5, 0.025, 0.975)):
        if x.size == 0:
            return [float("nan")] * len(ps)
        return [float(np.quantile(x, p)) for p in ps]

    median_all, lo_all, hi_all = qntl(y_corr)
    median_acc, lo_acc, hi_acc = qntl(y_acc)

    return {
        "config": cfg.__dict__,
        "summary": {
            "accept_rate": accept_rate,
            "rejected_rate": 1.0 - accept_rate,
            "median_unconditional": median_all,
            "ci95_unconditional": [lo_all, hi_all],
            "median_conditional": median_acc,
            "ci95_conditional": [lo_acc, hi_acc],
            "p_c14_unconditional": p_c14,
            "p_c14_conditional": p_c14_cond,
        },
        "samples": {
            "y_corr": y_corr,
            "y_acc": y_acc,
        },
    }


def plot_constraint_comparison(results: dict, outpath: Path) -> None:
    y = results["samples"]["y_corr"]
    y_acc = results["samples"]["y_acc"]
    summ = results["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), constrained_layout=True)

    # Panel A: unconditional
    ax = axes[0]
    ax.hist(y, bins=60, density=True, edgecolor="black", linewidth=0.25)
    ax.axvline(Y_FIRE, color="orange", linestyle="--", linewidth=2, label="Fire (1532)")
    ax.axvline(summ["median_unconditional"], color="red", linewidth=2, label=f"Median: {summ['median_unconditional']:.0f} AD")
    ax.set_title("(A) Unconstrained predictive\n(n = {:,})".format(y.size))
    ax.set_xlabel("Corrected date (AD)")
    ax.set_ylabel("Density")
    ax.set_xlim(-500, 6000)
    ax.legend(loc="upper right")

    pct_after = 100.0 * float(np.mean(y > Y_FIRE))
    ax.text(0.63, 0.85,
            f"{pct_after:.1f}% dates > 1532\n(inconsistent with\nhistorical constraint)",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red"),
            color="red", ha="left", va="top")

    ax.text(0.5, 0.05,
            "Predictive check\n(not an estimate of true date)",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5"),
            ha="center", va="bottom")

    ins = ax.inset_axes([0.56, 0.52, 0.42, 0.36])
    ins.hist(y, bins=45, density=True, edgecolor="black", linewidth=0.2)
    ins.axvline(Y_FIRE, color="orange", linestyle="--", linewidth=1.5)
    ins.set_xlim(0, 1600)
    ins.set_title("Zoom [0–1600]", fontsize=10)

    # Panel B: conditional
    ax = axes[1]
    ax.hist(y_acc, bins=35, density=True, edgecolor="black", linewidth=0.25)
    ax.axvline(Y_FIRE, color="orange", linestyle="--", linewidth=2, label="Fire (1532)")
    ax.axvline(summ["median_conditional"], color="red", linewidth=2, label=f"Median: {summ['median_conditional']:.0f} AD")
    ax.set_title("(B) Conditional: p(y_corr | y_corr ≤ 1532)\n(rejection sampling, {:.0f}% acceptance)".format(100 * summ["accept_rate"]))
    ax.set_xlabel("Corrected date (AD)")
    ax.set_ylabel("Density")
    ax.set_xlim(-500, 6000)
    ax.legend(loc="upper right")

    ax.text(0.58, 0.86,
            "Median: {:.0f} AD\n95% interval: [{:.0f}, {:.0f}]\nRejected: {:.0f}%\nP(C14): {:.1f}%".format(
                summ["median_conditional"],
                summ["ci95_conditional"][0],
                summ["ci95_conditional"][1],
                100 * summ["rejected_rate"],
                100 * summ["p_c14_conditional"],
            ),
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#d4f7d4", edgecolor="green"),
            ha="left", va="top")

    ins = ax.inset_axes([0.55, 0.15, 0.42, 0.36])
    ins.hist(y_acc, bins=40, density=True, edgecolor="black", linewidth=0.2)
    ins.axvline(Y_FIRE, color="orange", linestyle="--", linewidth=1.5)
    ins.axvline(C14_MIN, color="red", linestyle=":", linewidth=1.2)
    ins.axvline(C14_MAX, color="red", linestyle=":", linewidth=1.2)
    ins.set_xlim(0, 1600)
    ins.set_title("Zoom [0–1600]", fontsize=10)

    fig.suptitle("Effect of historical constraint y_corr ≤ 1532 on Monte Carlo distribution", fontsize=16, fontweight="bold")

    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--fire_prior", type=str, default="jeffreys_K", choices=["jeffreys_K", "uniform_T_C", "uniform_invT_K"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = MCConfig(n=args.n, seed=args.seed, fire_prior=args.fire_prior)
    results = run_monte_carlo(cfg)

    # Save figure + JSON summary
    fig_path = outdir / "fig_constraint_comparison.png"
    plot_constraint_comparison(results, fig_path)

    json_path = outdir / "fire_correction_results_with_constraint.json"
    payload = {
        "config": results["config"],
        "summary": results["summary"],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    # Console summary
    s = results["summary"]
    print("Acceptance rate: {:.1f}%".format(100 * s["accept_rate"]))
    print("Unconditional median (AD): {:.0f} ; 95% [{:.0f}, {:.0f}]".format(
        s["median_unconditional"], s["ci95_unconditional"][0], s["ci95_unconditional"][1]))
    print("Conditional median (AD): {:.0f} ; 95% [{:.0f}, {:.0f}]".format(
        s["median_conditional"], s["ci95_conditional"][0], s["ci95_conditional"][1]))
    print("P(C14) unconditional: {:.2f}%".format(100 * s["p_c14_unconditional"]))
    print("P(C14) conditional: {:.2f}%".format(100 * s["p_c14_conditional"]))
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
