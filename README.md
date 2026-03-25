# Uncertainty-Aware Drug Response Prediction

This repository contains the code for benchmarking uncertainty quantification methods in drug response prediction using the GDSC database, and for leveraging the estiamtes using various applications.



```

## Installation

Requires Python >= 3.9.

```bash
pip install -e .
```

This installs `uadr` as an editable package along with its dependencies (PyTorch, PyTorch Lightning, scikit-learn, etc.). For the SHAP analysis scripts, also install:

```bash
pip install -e ".[shap]"
```

## Data

The experiments use the [GDSC](https://www.cancerrxgene.org/) drug response database. Processed data with the right structure is available at: [Zenodo](https://zenodo.org/records/19219091)

## Running experiments

### Cross-validation

```bash
python examples/cross_validation_models.py --model_type pnne --cv_type cell_line_cold_start --scaling_mode z_norm
```

Supported model types: `pnn`(Guassian NN), `pnne` (Gaussian NN Ensemble), `mcd` (MCDropout NN), `qfn` (Quantile NN), `br` (Bayesian Ridge), `rf` (Random Forest).

### Case-specific fine-tuning

```bash
python examples/case_specific_finetune.py --run_id my_run --scaling_mode z_norm
```

### SHAP driver analysis

```bash
cd examples/XAI_drivers
python XAI_train_model.py
python XAI_shap_analysis.py
```

## Generating figures

All figure scripts read results from `examples/data/experiments/` and write to `figure_code/figures/`. Run any script from the repository root:

```bash
python figure_code/prediction_performance_figures.py
python figure_code/uncertainty_performance_figures.py
python figure_code/OOD_figures.py
python figure_code/tissue_analysis_figures.py
python figure_code/case_specific_fine_tuning_figures.py
python figure_code/XAI_drivers_figures.py
python figure_code/uncertainty_prediction_illustration_figure.py
```
