import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count()))

import joblib
from data import load_data, split_data
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from models import models_dict
from config import (RANDOM_STATE, CV_SPLITS, SCORING, N_JOBS, SMOTE_SAMPLING_STRATEGY,
                    SMOTE_GRID, MODEL_SELECT, MODELS_DIR, model_path)


def build_model(model):
    smote_pipeline = Pipeline([
        ("prep", StandardScaler()),
        ("smote", SMOTE(sampling_strategy=SMOTE_SAMPLING_STRATEGY, random_state=RANDOM_STATE)),
        ("model", model)
    ])
    return smote_pipeline


def perform_grid_search(model_pipe_line, param_grid, scoring_metric, X_train, y_train, cv):
    """Run a grid search over `param_grid` for `model_pipe_line` and return the fitted GridSearchCV."""
    grid_search = GridSearchCV(
        estimator=model_pipe_line,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv,
        n_jobs=N_JOBS,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print("Best params:", grid_search.best_params_)
    print(f"Best {scoring_metric}: {grid_search.best_score_:.4f}")
    return grid_search


if __name__ == '__main__':
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    kf = StratifiedKFold(shuffle=True, random_state=RANDOM_STATE, n_splits=CV_SPLITS)

    model_select = MODEL_SELECT
    model = models_dict[model_select]["model"]
    param_grid = {**models_dict[model_select]["param_grid"], **SMOTE_GRID}
    smote_pipeline = build_model(model)
    grid_search = perform_grid_search(smote_pipeline, param_grid, SCORING,
                                      X_train, y_train, kf)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = model_path(model_select)
    joblib.dump(grid_search.best_estimator_, out_path)
    print(f"Saved best model to {out_path}")
