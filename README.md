# Cellulose model selection validation (temporal CV) + fire-correction diagnostic

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18203402.svg)](https://doi.org/10.5281/zenodo.18203402)

This repository accompanies a manuscript on **model selection in cellulosic ageing** using **forward temporal cross-validation** on the European paper database (Strlič *et al.* 2020), and a **diagnostic Arrhenius fire-correction** case study for published flax mechanical-dating data.

It is intentionally split into two ideas:

1. **Methodology (core result):** *Temporal validation beats "good fit on the calibration set"*.
2. **Case study (illustration):** A documented thermal event (the **1532 Chambéry fire**) can dominate dating discrepancies if omitted.

> Scope / tone control: this code is for *methodological evaluation* and *compatibility diagnostics*.
> It does **not** claim to reconstruct a fire's exact conditions, nor to adjudicate authenticity of any artifact.

---

## Repository layout

```
├── data/
│   ├── 10570_2020_3344_MOESM2_ESM.xlsx   (Strlič et al. supplementary — see "Data" below)
│   ├── fanti_basso_2015_calibration.csv
│   ├── fanti_2017_calibration.csv
│   └── fanti_2017_ts_sample.csv
├── scripts/
│   ├── 01_paper_validation.py             — temporal CV on paper database (linear vs saturating)
│   ├── 02_fanti_reproduction.py           — reproductions of published flax calibration
│   ├── 03_fanti_saturating_reanalysis.py  — small-n cautions + VIF multicollinearity analysis
│   ├── 04_fire_correction_arrhenius_constraint.py — Arrhenius MC with y_corr ≤ 1532 constraint
│   └── 05_generate_all_figures.py         — generates ALL 6 manuscript figures (unified script)
├── figures/                               — generated figures (.png, .pdf) and JSON summaries
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
└── .zenodo.json
```

---

## Key parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Activation energy Ea | 110 kJ/mol | Emsley & Stevens (1994): 111 ± 6 kJ/mol |
| Basso calibration slope m | 9.256 × 10⁻⁴ | Basso et al. (2015) |
| Basso calibration intercept q | 4.936 | Basso et al. (2015) |
| σ_R (Shroud sample) | 155 MPa | Fanti & Basso (2017), not the 243 MPa of Basso et al. (2015) |
| Ambient temperature T₀ | 293.15 K (20 °C) | Standard reference |
| Radiocarbon range | 1260–1390 AD | Damon et al. (1989) |
| Historical constraint | y_corr ≤ 1532 | Chambéry fire date |

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

### 3) Run the pipeline

```bash
# Step 1: Paper database validation (temporal CV, Welch test, bias analysis)
python scripts/01_paper_validation.py --outdir out_paper

# Step 2: Reproduce published Fanti/Basso calibration
python scripts/02_fanti_reproduction.py

# Step 3: Saturating reanalysis + VIF multicollinearity
python scripts/03_fanti_saturating_reanalysis.py

# Step 4: Monte Carlo fire correction with historical constraint
python scripts/04_fire_correction_arrhenius_constraint.py --n 315329 --seed 1

# Step 5: Generate ALL 6 manuscript figures
python scripts/05_generate_all_figures.py --outdir figures --paper-results out_paper --n-mc 315329
```

### Figure → manuscript map

| Output file | Manuscript figure | Content |
|---|---|---|
| `fig1_model_schematic.png` | Fig. 1 | Ekenstam vs saturating model schematic |
| `fig2_cv_results.png` | Fig. 2 | Forward temporal CV results (MAE, ΔAICc) |
| `fig3_bias_analysis.png` | Fig. 3 | Age-dependent bias (Welch t-test) |
| `fig_fanti_fire_omission.png` | Fig. 4 | Fire omission diagram (116→1334 AD) |
| `fig_fire_correction_article.png` | Fig. 5 | 3-panel MC analysis (curves, y_corr, T_fire) |
| `fig_constraint_comparison.png` | Fig. 6 | Unconstrained vs conditional distribution |

---

## Notes on interpretation

- **Temporal CV result (paper database):** the saturating model reduces MAE consistently (~32%), while RMSE can worsen due to outlier sensitivity (bias–variance tradeoff).
- **Small calibration sample (flax, n = 9):** do *not* overinterpret "which model wins"; the dataset is too small and heterogeneous to discriminate robustly.
- **Fire correction:** the key methodological point is that any "corrected date" must respect the external constraint **y_corr ≤ 1532** (the date of the fire), and that this conditioning should be reported transparently (acceptance rate, prior choices).
- **Activation energy:** Ea = 110 kJ/mol is within the consensus range established by Emsley & Stevens (1994): 111 ± 6 kJ/mol, confirmed by Lundgaard et al. (2004) and Jalbert et al. (2015).

---

## Data provenance

- **Paper database:** Strlič *et al.* (2020) supplementary material (not redistributed here).
- **Flax calibration/measurements:** digitized from Basso *et al.* (2015) and Fanti & Basso (2017); see CSV headers and manuscript for full citations.

If you redistribute any third-party data, please verify the original license/terms.

---

## How to cite

For reproducibility, please cite the **Zenodo release (version DOI)** you used:

- **Version DOI (v1.1.0)**: *[to be updated after release]*

The stable **project concept DOI** (all versions) is:

- **Concept DOI**: https://doi.org/10.5281/zenodo.18203402

See also `CITATION.cff` for machine-readable metadata.

## License

Code: MIT (see `LICENSE`).
