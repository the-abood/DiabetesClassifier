# ============================================================
# FILE: src/models/random_forest.py
# PURPOSE: Random Forest training wrapper with GridSearchCV.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# ── Default grid ──────────────────────────────────────────────────────────────

DEFAULT_PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}


# ── Training ──────────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    class_weights: dict,
    param_grid: dict    = None,
    cv:         int     = 5,
    scoring:    str     = "roc_auc",
    random_state: int   = 42,
    n_jobs:     int     = -1,
    verbose:    int     = 1,
) -> tuple[RandomForestClassifier, GridSearchCV]:
    """
    Train a Random Forest classifier with 5-fold cross-validated
    hyperparameter grid search, optimising for ROC-AUC.

    Algorithm rationale
    -------------------
    Random Forest is an ensemble of independently grown decision trees
    (bagging + feature subsampling). Each tree is trained on a bootstrap
    sample; predictions are aggregated by majority vote (classification).

    Strengths for this dataset:
    - Handles mixed feature types (numerical + encoded categorical) natively
    - Robust to outliers — splits are threshold comparisons, not distances
    - Non-linear decision boundaries — captures interactions like
      high HbA1c AND high BMI jointly predicting diabetes
    - Built-in Gini importance provides free feature attribution
    - Scale-invariant — StandardScaler is not required but does not harm it

    Key hyperparameters
    -------------------
    n_estimators      : more trees → lower variance, higher compute cost
    max_depth         : None = fully grown; smaller = more regularisation
    min_samples_split : minimum samples to attempt a split; higher = smoother
    class_weight      : re-weights the loss to compensate for the 60/40 imbalance

    Parameters
    ----------
    X_train       : scaled training feature matrix
    y_train       : training target series
    class_weights : {0: w0, 1: w1} from compute_class_weight
    param_grid    : hyperparameter grid (default: DEFAULT_PARAM_GRID)

    Returns
    -------
    (best_estimator, fitted GridSearchCV object)
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    rf = RandomForestClassifier(
        class_weight  = class_weights,
        random_state  = random_state,
        n_jobs        = n_jobs,
    )

    gs = GridSearchCV(
        estimator  = rf,
        param_grid = param_grid,
        cv         = cv,
        scoring    = scoring,
        n_jobs     = n_jobs,
        verbose    = verbose,
    )

    total_fits = (
        len(param_grid.get("n_estimators", [1])) *
        len(param_grid.get("max_depth", [1])) *
        len(param_grid.get("min_samples_split", [1])) *
        cv
    )
    print(f"Random Forest grid search: {total_fits} fits ({cv}-fold CV)")
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    print(f"\nBest params   : {gs.best_params_}")
    print(f"CV {scoring}  : {gs.best_score_:.4f}")

    return best, gs
