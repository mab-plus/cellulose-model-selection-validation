# Cellulose model selection validation (temporal CV) + fire-correction diagnostic

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18203402.svg)](https://doi.org/10.5281/zenodo.18203402)

This repository accompanies a manuscript on **model selection in cellulosic ageing** using **forward temporal cross-validation** on the European paper database (Strlič *et al.* 2020), and a **diagnostic Arrhenius fire-correction** case study for published flax mechanical-dating data.

It is intentionally split into two ideas:

1. **Methodology (core result):** *Temporal validation beats “good fit on the calibration set”*.
2. **Case study (illustration):** A documented thermal event (the **1532 Chambéry fire**) can dominate dating discrepancies if omitted.

> Scope / tone control: this code is for *methodological evaluation* and *compatibility diagnostics*.
> It does **not** claim to reconstruct a fire’s exact conditions, nor to adjudicate authenticity of any artifact.

---

## Repository layout

- `data/`
  - `10570_2020_3344_MOESM2_ESM.xlsx` *(Strlič et al. supplementary spreadsheet — see “Data” below)*
  - `fanti_basso_2015_calibration.csv`, `fanti_2017_calibration.csv`, `fanti_2017_ts_sample.csv`, etc.
- `scripts/`
  - `01_paper_validation.py` — temporal CV on paper database (linear vs saturating in 1/DP)
  - `02_fanti_reproduction.py` — reproductions / sanity checks on published flax calibration
  - `03_fanti_saturating_reanalysis.py` — small-n cautions + model-form comparisons
  - `03_generate_figures.py` — figure assembly helpers
  - `04_fire_correction_arrhenius_constraint.py` — Arrhenius MC with explicit **y_corr ≤ 1532** constraint
- `figures/` — generated figures and JSON summaries (can be committed for convenience)

---

## Quickstart

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Put the data file in place

Download the Strlič *et al.* (2020) supplementary spreadsheet and save it as:

```
data/10570_2020_3344_MOESM2_ESM.xlsx
```

(If you already have it, just drop it there.)

### 3) Run the pipeline

```bash
python scripts/01_paper_validation.py
python scripts/02_fanti_reproduction.py
python scripts/03_fanti_saturating_reanalysis.py
python scripts/04_fire_correction_arrhenius_constraint.py --n 100000 --seed 1
```

Outputs are written to `figures/` by default (see each script’s `--help`).

---

## Notes on interpretation

- **Temporal CV result (paper database):** the “saturating” model reduces MAE consistently, while RMSE can worsen due to outlier sensitivity (bias–variance tradeoff).
- **Small calibration sample (flax, n=9):** do *not* overinterpret “which model wins”; the dataset is too small and heterogeneous to discriminate robustly.
- **Fire correction:** the key methodological point is that any “corrected date” must respect the external constraint **y_corr ≤ 1532** (the date of the fire), and that this conditioning should be reported transparently (acceptance rate, prior choices).

---

## Data provenance

- Paper database source: Strlič *et al.* (2020) supplementary material (not redistributed here by default).
- Flax calibration / measurements: digitized / transcribed from the cited publications (see manuscript and the CSV headers).

If you redistribute any third-party data, please verify the original license/terms.

---

## How to cite

For reproducibility, please cite the **Zenodo release (version DOI)** you used:

- **Version DOI (v1.0.3)**: https://doi.org/10.5281/zenodo.18203719

The stable **project concept DOI** (all versions) is:

- **Concept DOI**: https://doi.org/10.5281/zenodo.18203402

## License

Code: MIT (see `LICENSE`).
