# Cellulose model selection validation (temporal CV) + fire-correction diagnostic

<!-- Zenodo DOI badge will be added after first Zenodo archive (GitHub Release -> Zenodo record). -->

This repository accompanies a manuscript focused on **model selection and temporal validation** for cellulosic ageing models, with a **diagnostic fire-correction** case study applied to published flax mechanical-dating data.

It is intentionally split into two ideas:

1. **Methodology (core result):** *temporal validation beats “good fit on the calibration set”*.
2. **Case study (illustration):** a documented thermal event (e.g., the **1532 fire**) can dominate chronological discrepancies if omitted in environmental histories.

> **Scope / tone control**  
> This code is for **methodological evaluation** and **compatibility diagnostics**.  
> It does **not** claim to reconstruct a fire’s exact conditions, nor to adjudicate the authenticity of any artifact.

---

## Repository layout

```
.
├── scripts/                 # analysis scripts (reproducible pipeline)
├── data/                    # input data (fanti CSV included; external XLSX not redistributed)
│   ├── fanti/               # published flax mechanical-dating datasets (CSV)
│   └── external/            # third-party datasets (ignored by git; user-provided)
├── figures/                 # generated figures (PNG + PDF) + selected JSON outputs
├── results/                 # optional: your local outputs (empty by default)
├── CITATION.cff             # citation metadata (GitHub / Zenodo)
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT
└── README.md
```

---

## Quickstart

### 0) Create a Python environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data

### A) External paper database (XLSX, not redistributed)

This repo uses an external Excel file (Strlič et al., 2020; supplementary material) for the **temporal validation** experiment.

Because it is third-party content, `.xlsx` files are ignored by default (`.gitignore` contains `data/**/*.xlsx`).

1) Download the supplementary file named:

`10570_2020_3344_MOESM2_ESM.xlsx`

2) Place it here:

`data/external/10570_2020_3344_MOESM2_ESM.xlsx`

---

### B) Published flax mechanical datasets (included)

CSV files under `data/fanti/` are derived from **published** sources used for reproduction / reanalysis / diagnostics.

Key references:
- Fanti & Malfi (2014), *Textile Research Journal*, DOI: 10.1177/0040517513507366  
- Fanti, Malfi & Crosilla (2015), *MATEC Web of Conferences*, DOI: 10.1051/matecconf/20153601001  
- Basso, Fanti & Malfi (2015), *MATEC Web of Conferences*, DOI: 10.1051/matecconf/20153601003  
- Fanti & Basso (2017), *Int. J. Reliability, Quality and Safety Engineering*, DOI: 10.1142/S0218539317500061  

---

## Reproduce the main results

All commands below are run from the repository root.

### 1) Temporal cross-validation on paper dataset (core methodology)

```bash
python scripts/01_paper_validation.py   --xlsx data/external/10570_2020_3344_MOESM2_ESM.xlsx   --outdir out_paper
```

This produces model comparison outputs used by the manuscript (forward/temporal CV; bias under extrapolation).

---

### 2) Reproduce and compare published flax regressions (baseline reproduction)

```bash
python scripts/02_fanti_reproduction.py   --calib data/fanti/fanti_basso_2015_calibration.csv   --ts data/fanti/turin_shroud_measurements.csv   --outdir out_flax
```

This script reproduces published regression behaviour and provides a baseline for comparison.

---

### 3) Saturating reanalysis (mechanistic alternative in 1/DP space)

```bash
python scripts/03_fanti_saturating_reanalysis.py   --outdir out_fanti_reanalysis   --n-mc 10000
```

---

### 4) Fire-correction Monte Carlo (explicit historical constraint)

This is a **diagnostic** step showing how conditioning on a minimal historical constraint (e.g. “pre-1532”) changes the predictive distribution.

```bash
python scripts/04_fire_correction_arrhenius_constraint.py   --n 100000   --seed 1   --outdir figures   --fire_prior jeffreys_K
```

Outputs include:
- `figures/fig_constraint_comparison.(png|pdf)`
- `figures/fire_correction_results_with_constraint.json`

---

### 5) Generate publication-quality figures

```bash
python scripts/03_generate_figures.py   --paper-results out_paper   --flax-results out_flax   --outdir figures
```

The `figures/` directory contains both `.png` and `.pdf` for each figure.

---

## Notes on interpretation

- The **paper dataset** experiment is the core methodological result: it tests whether model selection is stable under **forward/temporal** validation (train on older samples, test on newer).
- The **flax case study** is intentionally framed as a **stress-test** for sensitivity to environmental history and thermal shocks.
- “Fire temperature” results are **not** historical reconstructions: they are conditional diagnostics showing how an Arrhenius-type scaling can amplify time scales.

---

## How to cite

Use `CITATION.cff` (GitHub understands it and Zenodo will import metadata on release archiving).

After the first Zenodo archive is created (via a GitHub Release), add the **Concept DOI** to:
- `README.md` (badge at the top)
- `CITATION.cff` (field `doi:`)

---

## License

Code: **MIT** (see `LICENSE`).

---

## Contact / issues

Please use GitHub Issues for reproducibility questions, bug reports, or clarification requests.
