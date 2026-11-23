# Credit Card Fraud Detection

A machine-learning pipeline that detects fraudulent credit card transactions in a
highly imbalanced dataset (about 0.17% fraud). It combines feature scaling,
SMOTE oversampling and tree-based classifiers, tuned with stratified
cross-validated grid search.

## Dataset

[Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 transactions made by European cardholders over two days in September 2013.

| Column      | Description                                                    |
|-------------|----------------------------------------------------------------|
| `Time`      | Seconds elapsed since the first transaction in the dataset     |
| `V1`-`V28`  | Anonymised features (PCA components of the original data)      |
| `Amount`    | Transaction amount                                             |
| `Class`     | Target: `1` = fraud, `0` = legitimate                          |

1,081 duplicate rows are dropped, which leaves 283,726 transactions.

> The CSV is ~150 MB and is **not** stored in the repository. Download
> `creditcard.csv` from Kaggle and place it at `data/creditcard.csv`.

## Project structure

```
Credit_Card_Detection/
├── data/
│   └── creditcard.csv        # dataset (download separately, git-ignored)
├── models/
│   └── <model>.joblib        # trained pipelines saved by train.py (git-ignored)
├── notebooks/
│   └── playground.ipynb      # EDA and baseline experiments (Logistic Regression)
├── src/
│   ├── config.py             # paths, split/CV settings, SMOTE grid, model selection
│   ├── data.py               # load, de-duplicate, stratified train/test split
│   ├── models.py             # candidate estimators and their hyperparameter grids
│   └── train.py              # build pipeline, run grid search, save best model
├── requirements.txt
└── README.md
```

## Approach

1. **Load and clean.** Read the CSV and drop duplicate rows.
2. **Split.** An 80/20 stratified train/test split keeps the fraud ratio the same in both sets.
3. **Pipeline** (`imblearn.pipeline.Pipeline`):
   `StandardScaler` → `SMOTE` → classifier.
   SMOTE runs inside the pipeline, so synthetic samples are created only from
   the training folds and never leak into validation data.
4. **Tuning.** `GridSearchCV` with 3-fold `StratifiedKFold`. It searches the
   model's hyperparameters together with SMOTE's `sampling_strategy` (0.2, 0.5).
5. **Metric.** **Average precision** (area under the precision-recall curve).
   Accuracy is misleading here: a model that predicts "not fraud" for every
   transaction is 99.8% accurate.
6. **Save.** The best pipeline is written to `models/<model_name>.joblib`.

### Models

| Key             | Estimator                | Tuned hyperparameters                                         |
|-----------------|--------------------------|---------------------------------------------------------------|
| `decision_tree` | `DecisionTreeClassifier` | `max_depth`, `min_samples_leaf`                               |
| `random_forest` | `RandomForestClassifier` | `n_estimators`, `max_depth`, `min_samples_leaf`               |
| `xgboost`       | `XGBClassifier`          | `n_estimators`, `max_depth`, `learning_rate`, `subsample`     |

The notebook also explores a Logistic Regression baseline, with and without
SMOTE, scored with F2 and average precision.

## Getting started

```bash
# 1. Clone
git clone https://github.com/a7madmostafa/credit-card-detection.git
cd credit-card-detection

# 2. Create an environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 3. Add the dataset at data/creditcard.csv

# 4. Train
python src/train.py
```

Paths are resolved from `src/config.py`, so the scripts work from any working directory.

### Choosing a model

Set `MODEL_SELECT` in [`src/config.py`](src/config.py) to `decision_tree`,
`random_forest` or `xgboost`. The same file holds the other settings: test size,
CV folds, scoring metric, SMOTE grid and random seed.

### Using a trained model

```python
import joblib
import pandas as pd

model = joblib.load("models/random_forest.joblib")
X_new = pd.read_csv("data/creditcard.csv").drop(columns="Class").head()
model.predict(X_new)           # 0 / 1 labels
model.predict_proba(X_new)[:, 1]  # fraud probability
```

At prediction time the saved pipeline applies the scaler and the classifier.
SMOTE only runs during `fit`.

## Results

| Model         | CV average precision | Test average precision | Test recall | Test precision |
|---------------|---------------------:|-----------------------:|------------:|---------------:|
| Decision Tree | _TBD_                | _TBD_                  | _TBD_       | _TBD_          |
| Random Forest | _TBD_                | _TBD_                  | _TBD_       | _TBD_          |
| XGBoost       | _TBD_                | _TBD_                  | _TBD_       | _TBD_          |

## Tech stack

Python 3.13 · pandas · scikit-learn · imbalanced-learn · XGBoost · joblib · Jupyter
