from pathlib import Path

# ---------- Paths ----------
# Resolved relative to this file, so scripts work from any working directory.
# config.py lives in src/, so the project root is one level up.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODELS_DIR = BASE_DIR / "models"  # where trained .joblib models are saved

# ---------- Data ----------
TARGET_COL = "Class"
TEST_SIZE = 0.2

# ---------- Reproducibility ----------
RANDOM_STATE = 42

# ---------- Cross-validation / search ----------
CV_SPLITS = 3
SCORING = "average_precision"
N_JOBS = -1  # parallelism for GridSearchCV

# ---------- SMOTE ----------
SMOTE_SAMPLING_STRATEGY = 0.2
SMOTE_GRID = {"smote__sampling_strategy": [0.2, 0.5]}

# ---------- Training ----------
MODEL_SELECT = "random_forest"  # one of: decision_tree, random_forest, xgboost


def model_path(model_name):
    """Path where the trained model `model_name` is saved."""
    return MODELS_DIR / f"{model_name}.joblib"
