# ============================================================
# FILE: src/eda/exploratory_analysis.py
# PURPOSE: Reusable EDA helpers for the diabetes dataset.
#          All functions print summaries and optionally save
#          charts to outputs/figures/.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FIGURES_DIR = "outputs/figures"


# ── Summary ───────────────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a consolidated EDA summary: shape, dtypes, missing values,
    duplicates, and basic descriptive statistics.
    """
    print("=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Rows     : {df.shape[0]:,}")
    print(f"  Columns  : {df.shape[1]}")
    print(f"  Missing  : {df.isnull().sum().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")
    print()

    numerical   = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"  Numerical features   ({len(numerical)}) : {numerical}")
    print(f"  Categorical features ({len(categorical)}): {categorical}")
    print()
    print("--- Descriptive Statistics (numerical) ---")
    print(df[numerical].describe().round(3).to_string())


def class_balance(df: pd.DataFrame, target: str = "diagnosed_diabetes") -> None:
    """
    Print and plot the class balance of the target column.
    Flags if the minority class is below 35% (mild imbalance threshold).
    """
    counts = df[target].value_counts()
    pct    = df[target].value_counts(normalize=True) * 100

    print(f"\nClass distribution — '{target}':")
    for cls, cnt in counts.items():
        print(f"  {cls} : {cnt:>7,}  ({pct[cls]:.1f}%)")

    minority_pct = pct.min()
    if minority_pct < 35:
        print(f"\n  ⚠  Mild class imbalance detected ({minority_pct:.1f}% minority).")
        print("     Class weights will be applied during model training.")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    colours = ["#5cb85c", "#d9534f"]
    bars = ax.bar(
        ["No Diabetes (0)", "Diabetes (1)"],
        counts.values,
        color=colours, edgecolor="white", width=0.5
    )
    for bar, cnt, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 300,
                f"{cnt:,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=11)
    ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/class_distribution.png", dpi=150)
    plt.show()


# ── Correlation ───────────────────────────────────────────────────────────────

def correlation_heatmap(
    df: pd.DataFrame,
    target: str = "diagnosed_diabetes",
    figsize: tuple = (16, 13),
) -> None:
    """
    Lower-triangle correlation heatmap for all numerical features.
    Prints the top 10 features most correlated with the target.
    """
    num_df = df.select_dtypes(include=[np.number])
    corr   = num_df.corr()

    # Top correlations with target
    top_corr = (
        corr[target].drop(target).abs().sort_values(ascending=False).head(10)
    )
    print(f"\nTop 10 features correlated with '{target}':")
    print(top_corr.round(4).to_string())

    # Heatmap
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", linewidths=0.3,
        vmin=-1, vmax=1, annot=False, ax=ax
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", dpi=150)
    plt.show()


# ── Distributions ─────────────────────────────────────────────────────────────

def feature_distributions_by_class(
    df: pd.DataFrame,
    features: list,
    target: str = "diagnosed_diabetes",
) -> None:
    """
    Overlaid histograms for a list of features, split by target class.
    Useful for visually verifying that selected features separate classes.
    """
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        for label, colour, lname in [
            (0, "#5cb85c", "No Diabetes"),
            (1, "#d9534f", "Diabetes"),
        ]:
            ax.hist(
                df[df[target] == label][feat],
                bins=40, alpha=0.6, color=colour, label=lname
            )
        ax.set_title(feat.replace("_", "\n"), fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle(
        "Key Feature Distributions by Diagnosis",
        y=1.02, fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        f"{FIGURES_DIR}/feature_distributions.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()


def boxplots_by_class(
    df: pd.DataFrame,
    features: list,
    target: str = "diagnosed_diabetes",
) -> None:
    """
    Side-by-side boxplots for selected features grouped by target class.
    Highlights median differences and outlier spread.
    """
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 5))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        groups = [
            df[df[target] == 0][feat].dropna(),
            df[df[target] == 1][feat].dropna(),
        ]
        bp = ax.boxplot(groups, patch_artist=True, notch=False)
        colours = ["#5cb85c", "#d9534f"]
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)
        ax.set_xticklabels(["No Diabetes", "Diabetes"], fontsize=8)
        ax.set_title(feat.replace("_", "\n"), fontsize=9)

    plt.suptitle(
        "Feature Boxplots by Diagnosis",
        y=1.02, fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        f"{FIGURES_DIR}/feature_boxplots.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()


# ── Categorical ───────────────────────────────────────────────────────────────

def categorical_diabetes_rate(
    df: pd.DataFrame,
    cat_col: str,
    target: str = "diagnosed_diabetes",
) -> None:
    """
    Bar chart showing the diabetes diagnosis rate (%) per category of a
    categorical feature. Useful for identifying demographic risk patterns.
    """
    rates = (
        df.groupby(cat_col)[target].mean().sort_values(ascending=False) * 100
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    rates.plot(kind="bar", color="steelblue", edgecolor="white", ax=ax)
    ax.set_title(
        f"Diabetes Diagnosis Rate (%) by {cat_col}",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Diabetes Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.axhline(df[target].mean() * 100, color="tomato",
               linestyle="--", linewidth=1.5, label="Overall mean")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        f"{FIGURES_DIR}/diabetes_rate_{cat_col}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
