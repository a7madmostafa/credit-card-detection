from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from config import RANDOM_STATE

# Each entry: an estimator plus its param grid.
# Grid keys use the "model__" prefix, so the estimator must be the
# pipeline step named "model", e.g. Pipeline([..., ("model", estimator)]).
models_dict = {
    "decision_tree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [1, 5, 20],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, None],
            "model__min_samples_leaf": [1, 5],
        },
    },
    "xgboost": {
        "model": XGBClassifier(random_state=RANDOM_STATE, eval_metric="aucpr", n_jobs=1),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
        },
    },
}
