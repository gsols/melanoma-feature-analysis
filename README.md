# melanoma-feature-analysis

## Setup

Create and use a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python isic_data_prep.py
python isic_model.py
```

If you do not activate the virtual environment, run the model with:

```bash
.venv/bin/python isic_model.py
```

## Tune LightGBM

Run controlled LightGBM experiments and compare PR-AUC, recall, precision, and
benign flag rates:

```bash
.venv/bin/python isic_lgbm_experiments.py
```

Results are saved to:

```bash
reports/lgbm_experiment_results.csv
```
