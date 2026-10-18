# StudySmart AI

Reproducible machine learning research demo predicting final grades using historical student-performance records from two Portuguese secondary schools. **Not a diagnostic or individualized educational decision tool.**

## Setup

1. Install Python 3.10+.
2. `pip install -r requirements.txt`
3. Download UCI [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), extract `student-por.csv` into `data/`.
4. `python train.py --data data/student-por.csv`
5. `streamlit run app.py`

Training saves `results/metrics.json`, `results/summary.csv`, fitted `.joblib` pipelines and prediction/importance PNG plots. **Real-data results have been generated and included in `results/`.**

## Method

- Predict `G3` (0–20 final grade) from either habits/context or habits/context plus `G1` and `G2`.
- 80% training / 20% held-out test (seed 42).
- Select from mean baseline, Ridge, Random Forest and Histogram Gradient Boosting by five-fold cross-validated MAE **within the training set only**.
- Report MAE, RMSE, R² on the held-out test set; generate permutation importance plots.
- All preprocessing occurs within the cross-validation pipeline.

## Limitations

Observational, nonrepresentative, old Portuguese school data; correlations do not establish causation. Previous grades may not be available early in a school year. Model choice is based on cross-validation; comparing test-set scenarios is descriptive, not an independent confirmation of generalization. The data do not contain sleep duration or recreational screen time. No fairness audit or external validation has been performed. Do not use predictions to rank students or make high-stakes decisions.

## Reproducibility / authorship

Source code and experiment design can be inspected. Record your own experiments and changes before describing this as independent research. Credit AI assistance when appropriate and comply with the university's application policies.

## Reference

Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. FUBUTEC. Dataset: UCI Machine Learning Repository, Student Performance (dataset 320).

## Real-data findings (Portuguese course, n=649)

| Experiment | CV MAE | Test MAE | Test RMSE | Test R² | Selected algorithm |
|---|---:|---:|---:|---:|---|
| Habits/context only | 2.154 | 2.157 | 2.873 | 0.153 | ridge |
| Habits/context + previous grades | 0.831 | 0.754 | 1.182 | 0.857 | ridge |

These metrics come from a single held-out split of 130 records (519 training). **Previous grades strongly improve accuracy**, but this is not a causal result or evidence of generalization outside the source schools. Models are selected by 5-fold CV within training data; no independent external validation was performed.

The original dataset is included at `data/student-por.csv` for reproducibility; see its source and citation in `data/README.md`. Never submit student-level predictions for high-stakes decisions.

## GitHub publication

Create a new empty GitHub repository named `StudySmart-AI`, then run from the extracted project folder:

```bash
git init
git add .
git commit -m "Publish reproducible student performance ML experiments"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/StudySmart-AI.git
git push -u origin main
```

## Streamlit deployment

Connect the GitHub repository to Streamlit Community Cloud and set the entry point to `app.py`. The trained `.joblib` files are excluded from Git for portability/security; before deployment, either run `python train.py` and commit only **trusted** model artifacts with a deliberate policy change, or configure deployment to retrain on startup. The current app requires the model artifacts, so it is **not yet one-click deployable** from a fresh GitHub clone.
