# ============================================================
# FILE: src/models/logistic_regression.py
# PURPOSE: Logistic Regression training wrapper with GridSearchCV.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# ── Default grid ──────────────────────────────────────────────────────────────

DEFAULT_PARAM_GRID = {
    "C":      [0.001, 0.01, 0.1, 1.0, 10.0],
    "solver": ["lbfgs", "liblinear"],
}


# ── Training ──────────────────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    class_weights: dict,
    param_grid:   dict  = None,
    cv:           int   = 5,
    scoring:      str   = "roc_auc",
    max_iter:     int   = 1000,
    random_state: int   = 42,
    n_jobs:       int   = -1,
    verbose:      int   = 1,
) -> tuple[LogisticRegression, GridSearchCV]:
    """
    Train a Logistic Regression classifier with 5-fold cross-validated
    hyperparameter grid search, optimising for ROC-AUC.

    Algorithm rationale
    -------------------
    Logistic Regression models the log-odds of the positive class as a
    linear combination of input features. The sigmoid function maps this
    to a probability in [0, 1]. Classification is by a 0.5 threshold
    (adjustable for precision/recall trade-off).

    Strengths for this dataset:
    - Highly interpretable: each coefficient is the change in log-odds
      for a one-unit increase in that feature, all else equal
    - Probabilistic output with well-calibrated probabilities
    - Fast to train even on 100,000 rows
    - Provides a strong, transparent linear baseline to compare against RF
    - Regulatory contexts (clinical risk scoring) often prefer linear models
      for explainability and auditability

    Limitations:
    - Assumes a linear decision boundary — cannot capture interaction effects
      (e.g. "high HbA1c AND sedentary lifestyle together") without explicit
      feature engineering
    - Sensitive to feature scale — StandardScaler is mandatory

    Key hyperparameters
    -------------------
    C       : inverse regularisation strength. Small C = strong L2 penalty
              (shrinks coefficients toward zero, prevents overfitting).
              Large C = weak regularisation (fits training data more closely).
    solver  : optimisation algorithm.
              'lbfgs'    — L-BFGS quasi-Newton; good for L2, multi-class
              'liblinear' — coordinate descent; good for small datasets, L1/L2
    max_iter: convergence budget; 1000 is safe for this feature count.
    class_weight: re-weights the log-loss to compensate for the 60/40 imbalance.

    Parameters
    ----------
    X_train       : SCALED training feature matrix (StandardScaler required)
    y_train       : training target series
    class_weights : {0: w0, 1: w1} from compute_class_weight

    Returns
    -------
    (best_estimator, fitted GridSearchCV object)
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    lr = LogisticRegression(
        class_weight  = class_weights,
        random_state  = random_state,
        max_iter      = max_iter,
    )

    gs = GridSearchCV(
        estimator  = lr,
        param_grid = param_grid,
        cv         = cv,
        scoring    = scoring,
        n_jobs     = n_jobs,
        verbose    = verbose,
    )

    total_fits = (
        len(param_grid.get("C", [1])) *
        len(param_grid.get("solver", [1])) *
        cv
    )
    print(f"Logistic Regression grid search: {total_fits} fits ({cv}-fold CV)")
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    print(f"\nBest params   : {gs.best_params_}")
    print(f"CV {scoring}  : {gs.best_score_:.4f}")

    return best, gs
