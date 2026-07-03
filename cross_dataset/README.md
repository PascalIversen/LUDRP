# Cross-dataset transfer / OOD experiments

Code that generates the cross-dataset out-of-distribution predictions and the CTRPv2 benchmark
reproduction. The resulting raw per-pair predictions are large and live on Zenodo; the figures and
tables are produced by `figure_code/cross_dataset_ood_figure.py` and
`figure_code/ctrpv2_benchmark_tables.py` from the aggregated summary / benchmark results.

- `run_transfer_native.py` / `run_transfer_harmonized.py` — train a model on one screen and score it
  on the others, in native (per-dataset) or harmonized (shared Broad RNA-seq) gene-expression space.
- `run_transfer_beataml.py` — CTRPv2 → BeatAML transfer (same platform, different laboratory).
- `run_ctrpv2_cv.py` — reproduce the 5-fold benchmark on CTRPv2 (`ctrpv2_features.py` is the loader).
- `transfer_data.py`, `native_transfer_data.py` — feature/response builders for the transfers.

Requires the `uadr` package (repo root) and the drevalpy screening data.
