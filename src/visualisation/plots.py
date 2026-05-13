# ============================================================
# FILE: src/visualisation/plots.py
# PURPOSE: Feature importance and coefficient visualisation
#          helpers for the two trained classifiers.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGURES_DIR = "outputs/figures"


def plot_rf_feature_importance(
    model,
    feature_names: pd.Index,
    top_n:    int = 20,
    filename: str = "rf_feature_importance.png",
) -> pd.Series:
    """
    Horizontal bar chart of the top-N Random Forest feature importances
    (mean decrease in Gini impurity).

    Interpretation
    --------------
    Gini importance measures how much each feature reduces impurity
    (class mixing) when used as a split point, averaged across all trees.
    Higher value → the feature is used more and reduces impurity more.

    Important caveats:
    - Gini importance can over-rate high-cardinality numerical features
      because they offer more split thresholds.
    - Features correlated with the true signal (e.g. `glucose_fasting`
      and `glucose_postprandial`) will share importance; neither will
      individually appear as dominant as the true driver.
    - For regulatory or clinical use, SHAP values provide more reliable
      attribution and should complement Gini importance.

    Parameters
    ----------
    model         : fitted RandomForestClassifier
    feature_names : column index from X (feature names for labelling)
    top_n         : number of top features to display

    Returns
    -------
    Sorted pd.Series of importance values (top_n entries)
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    top.plot(kind="barh", color="steelblue", edgecolor="white", ax=ax)
    ax.set_title(
        f"Top {top_n} Features — Random Forest Gini Importance",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Mean Decrease in Gini Impurity")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    plt.show()
    print(f"Saved → {FIGURES_DIR}/{filename}")

    print(f"\nTop {top_n} features (RF Gini importance):")
    print(top[::-1].round(4).to_string())
    return top


def plot_lr_coefficients(
    model,
    feature_names: pd.Index,
    filename: str = "lr_coefficients.png",
) -> pd.DataFrame:
    """
    Polarity-coloured horizontal bar chart of Logistic Regression
    coefficients, sorted from most positive to most negative.

    Interpretation
    --------------
    Each coefficient represents the change in log-odds of a positive
    (diabetes) diagnosis for a one-standard-deviation increase in that
    feature (since features are StandardScaler-transformed).

    - Blue bar  → positive coefficient → increases diabetes probability
    - Red bar   → negative coefficient → decreases diabetes probability
    - Bar length → effect magnitude

    Clinical sanity check: features like `hba1c`, `glucose_fasting`,
    `diabetes_risk_score` should appear with strong positive coefficients
    because elevated values are established diagnostic indicators.
    Features like `physical_activity_minutes_per_week` and `hdl_cholesterol`
    (protective) should appear negative.

    Parameters
    ----------
    model         : fitted LogisticRegression
    feature_names : column index from X

    Returns
    -------
    DataFrame of feature names and coefficients
    """
    coef_df = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient", ascending=False)

    coef_df["Color"] = coef_df["Coefficient"].apply(
        lambda x: "steelblue" if x > 0 else "indianred"
    )

    fig, ax = plt.subplots(figsize=(10, max(7, len(coef_df) * 0.35)))
    ax.barh(
        coef_df["Feature"],
        coef_df["Coefficient"],
        color=coef_df["Color"],
        edgecolor="white",
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(
        "Logistic Regression Feature Coefficients\n"
        "Blue = increases risk  |  Red = decreases risk",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Coefficient Value  (scaled units)")
    ax.invert_yaxis()  # strongest positive at top
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    plt.show()
    print(f"Saved → {FIGURES_DIR}/{filename}")

    return coef_df.drop(columns=["Color"])


def plot_metric_comparison_bar(
    comparison_df: pd.DataFrame,
    metrics: list  = None,
    filename: str  = "metric_comparison.png",
) -> None:
    """
    Grouped bar chart comparing key metrics across models.

    Parameters
    ----------
    comparison_df : output of evaluation.metrics.comparison_table()
    metrics       : list of column names to include
                    (default: Accuracy, F1 (wtd), ROC-AUC)
    """
    if metrics is None:
        metrics = ["Accuracy", "F1 (wtd)", "ROC-AUC"]

    plot_df = comparison_df[metrics]
    ax = plot_df.plot(
        kind="bar", figsize=(9, 5), width=0.6,
        color=["steelblue", "tomato", "mediumpurple"][:len(metrics)],
        edgecolor="white"
    )
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150)
    plt.show()
    print(f"Saved → {FIGURES_DIR}/{filename}")
