# Fake News ML/AI Project

Baseline fake news and claim truthfulness classification with TF-IDF features and linear models.

## Project layout

- `src/`: core pipeline (preprocessing, features, models, evaluation)
- `experiments/`: runnable experiments (baseline, robustness, calibration, ablations, cross-dataset)
- `data/raw/`: datasets (Fake/True CSVs and LIAR TSVs)
- `artifacts/`: experiment outputs (JSON/CSV/plots)
- `notebooks/`: exploratory analysis

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datasets

- Fake/True dataset: `data/raw/Fake.csv`, `data/raw/True.csv`
- LIAR dataset splits: `data/raw/train.tsv`, `data/raw/valid.tsv`, `data/raw/test.tsv`

LIAR is used strictly as a claim-level truthfulness benchmark:
- Binary mapping: REAL = {true, mostly-true, half-true}, FAKE = {barely-true, false, pants-fire}
- Input text: statement only (no speaker, party, or historical counts)

## Experiments

### Baseline (tuned linear models)

```powershell
python experiments\baseline.py --dataset fake_true
python experiments\baseline.py --dataset liar
python experiments\baseline.py --dataset combined
```

Outputs:
- `artifacts/baseline/baseline_results.json`
- `artifacts/baseline/baseline.json`

### Robustness

```powershell
python experiments\robustness.py --dataset fake_true
python experiments\robustness.py --dataset liar
```

Outputs:
- `artifacts/robustness/robustness_results.csv`
- `artifacts/robustness/robustness_summary.json`

### Calibration

```powershell
python experiments\calibration.py --dataset fake_true
python experiments\calibration.py --dataset liar
```

Outputs:
- `artifacts/calibration/calibration_results.json`
- `artifacts/calibration/roc_*.png`, `artifacts/calibration/pr_*.png`

### TF-IDF ablations

```powershell
python experiments\tfidf_ablation.py --dataset fake_true
python experiments\tfidf_ablation.py --dataset liar
```

Outputs:
- `artifacts/tfidf_ablation/tfidf_ablation_results.csv`
- `artifacts/tfidf_ablation/tfidf_ablation_summary.json`

### Cross-dataset generalization

```powershell
python experiments\cross_dataset.py --direction both --models logreg
```

Outputs:
- `artifacts/cross_dataset/cross_dataset_results.json`
- `artifacts/cross_dataset/errors_*.csv`
- `artifacts/cross_dataset/analysis_notes.md`

## Configuration

Key defaults live in `src/config.py`:
- text preprocessing controls
- TF-IDF feature views (word 1-2 grams, optional char 3-5 grams)
- model calibration and tuning ranges

## Notes

- Large TF-IDF vocabularies can be memory-intensive. Reduce `max_features` in `src/config.py` or lower CV folds via `--cv`.
- If you enable lemmatization, install spaCy and `en_core_web_sm`.
