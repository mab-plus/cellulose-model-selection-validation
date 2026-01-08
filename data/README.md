# Data directory

This folder contains input datasets used by the analysis scripts.

## Contents (expected)

### 1) Historical paper dataset (Strlič et al., 2020)

File name expected by the scripts:

- `10570_2020_3344_MOESM2_ESM.xlsx`

**Provenance:** supplementary material of Strlič *et al.* (2020), *Cellulose* (European paper database).

**Note on redistribution:** please verify the publisher/supplement terms before re-uploading this spreadsheet in a public repository.
By default, the repository `.gitignore` excludes `data/*.xlsx` to avoid accidental redistribution.

**How to add it:**
1. Download the supplementary spreadsheet from the journal’s supplementary materials.
2. Save it in this folder under the exact name above.

### 2) Flax / mechanical-dating calibration CSVs (from cited literature)

These are small, curated CSVs extracted/transcribed from the cited publications and/or their tables:

- `fanti_basso_2015_calibration.csv`
- `fanti_2017_calibration.csv`
- `fanti_2017_ts_sample.csv`
- `turin_shroud_measurements.csv`

Each CSV should include a short header comment or column naming that makes the source table/figure traceable.
If you add new extracted datasets, please also add a one-line provenance note below.

## Provenance notes (fill as you finalize)

- `fanti_basso_2015_calibration.csv`: extracted from Basso, Fanti & Malfi (2015), *Monte Carlo method applied to the mechanical dating of the Turin Shroud* (MATEC Web of Conferences).
- `fanti_2017_calibration.csv` and `fanti_2017_ts_sample.csv`: extracted from Fanti & Basso (2017), *Mechanical characterization of linen fibers: the Turin Shroud dating* (Int. J. Reliability, Quality and Safety Engineering).
- `turin_shroud_measurements.csv`: curated summary of Turin Shroud tensile-strength values used in the fire-correction section, consistent with the cited sources.

## Reproducibility tip

To keep runs deterministic, you may want to record:
- the exact spreadsheet version / download date for the Strlič dataset,
- any preprocessing steps (filters, column selections),
- script version (git commit hash) used to generate the figures/tables.

