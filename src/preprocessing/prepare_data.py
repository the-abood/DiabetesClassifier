# ============================================================
# FILE: src/preprocessing/prepare_data.py
# PURPOSE: Data cleaning, encoding, scaling, and splitting
#          helpers for the diabetes classification pipeline.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "employment_status",
    "smoking_status",
]

# diabetes_stage directly re-encodes the target variable — including it
# would constitute target leakage and produce artificially perfect scores.
LEAKAGE_COLS = ["diabetes_stage"]

TARGET_COL = "diagnosed_diabetes"


# ── Encode ────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode all categorical string columns.

    LabelEncoder maps each unique string to an integer (0-based, ordered
    alphabetically). This is appropriate for:
      - Tree-based models (Random Forest): splits are threshold comparisons
        on integer values; no ordinal meaning is assumed.
      - Logistic Regression: ordinal meaning IS assumed for LabelEncoder,
        but with StandardScaler applied afterwards and relatively low
        cardinality in these columns, the impact is minimal.
    For higher-cardinality or truly nominal features, one-hot encoding
    would be preferable.

    Parameters
    ----------
    df : raw DataFrame (modifies a copy, does not mutate the original)

    Returns
    -------
    DataFrame with categorical columns replaced by integer codes
    """
    df_enc = df.copy()
    le     = LabelEncoder()

    for col in CATEGORICAL_COLS:
        if col in df_enc.columns:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            print(f"  Encoded '{col}'  →  {df_enc[col].nunique()} unique classes")

    return df_enc


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that would cause target leakage.

    'diabetes_stage' contains values like 'Type 2', 'No Diabetes',
    'Pre-diabetic' — a direct re-encoding of the target. A model trained
    with this column would learn to predict the target from itself,
    producing unrealistically high scores that would not generalise
    to real clinical data.
    """
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped leakage columns: {cols_to_drop}")
    return df


# ── Split ─────────────────────────────────────────────────────────────────────

def get_features_and_target(
    df: pd.DataFrame,
    target: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X from target series y.
    Requires encode_categoricals() and drop_leakage_columns() to have
    been applied first.
    """
    X = df.drop(columns=[target])
    y = df[target]
    print(f"  Feature matrix : {X.shape}")
    print(f"  Target series  : {y.shape}  |  classes: {sorted(y.unique())}")
    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size:   float = 0.2,
    random_state: int  = 42,
) -> tuple:
    """
    Stratified 80/20 train/test split followed by StandardScaler.

    Stratification ensures the class ratio is preserved in both sets,
    which is important with mild imbalance (~60/40).

    StandardScaler
    --------------
    - Critical for Logistic Regression: regularisation (C parameter)
      is sensitive to feature scale. Unscaled features with large
      ranges dominate the gradient and bias coefficient estimates.
    - Harmless for Random Forest: tree splits are threshold comparisons
      that are scale-invariant. Scaling does not change RF predictions.
    - Fitted ONLY on training data and applied to test data to prevent
      data leakage (the scaler must not see test set statistics).

    Returns
    -------
    X_train_sc, X_test_sc, y_train, y_test, scaler, class_weights
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    # Class weights computed from training labels only
    cw_values    = compute_class_weight("balanced",
                                        classes=np.array([0, 1]),
                                        y=y_train)
    class_weights = {0: cw_values[0], 1: cw_values[1]}

    print(f"\n  Training rows  : {X_train_sc.shape[0]:,}")
    print(f"  Test rows      : {X_test_sc.shape[0]:,}")
    print(f"  Class weights  : {class_weights}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, class_weights


# ── Full pipeline ─────────────────────────────────────────────────────────────

def full_preprocess(
    df: pd.DataFrame,
    test_size:    float = 0.2,
    random_state: int   = 42,
) -> tuple:
    """
    Convenience function: run the entire preprocessing pipeline in one call.

    Steps
    -----
    1. Encode categorical features
    2. Drop leakage columns
    3. Separate X and y
    4. Split and scale

    Returns
    -------
    X_train_sc, X_test_sc, y_train, y_test, scaler, class_weights, X
    (X is returned for feature name recovery when building importance plots)
    """
    print("=== Preprocessing Pipeline ===")
    df_enc = encode_categoricals(df)
    df_enc = drop_leakage_columns(df_enc)
    X, y   = get_features_and_target(df_enc)
    X_train_sc, X_test_sc, y_train, y_test, scaler, cw = split_and_scale(
        X, y, test_size=test_size, random_state=random_state
    )
    print("=== Preprocessing Complete ===\n")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, cw, X
