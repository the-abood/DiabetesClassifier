# ============================================================
# FILE: src/evaluation/metrics.py
# PURPOSE: Evaluation helpers — per-model reporting, ROC curves,
#          confusion matrices, and head-to-head comparison table.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
)

FIGURES_DIR = "outputs/figures"


# ── Single Model ──────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test:    np.ndarray,
    y_test:    pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Compute and print a full evaluation report for a single classifier.

    Metrics reported
    ----------------
    Accuracy
        Overall fraction of correct predictions. Intuitive but misleading
        under class imbalance — a model predicting "always diabetes" on a
        60% positive dataset scores 60% accuracy without learning anything.

    Precision (per class)
        Of all patients predicted positive, what fraction truly are?
        Clinical relevance: low precision → unnecessary follow-up burden.

    Recall / Sensitivity (per class)
        Of all true positive patients, what fraction were caught?
        Clinical relevance: low recall → missed diagnoses. In healthcare,
        recall for the positive class is usually the priority metric.

    F1 Score
        Harmonic mean of precision and recall. Balances both; useful when
        classes are imbalanced and a single number is needed.

    ROC-AUC
        Area under the ROC curve. Measures the model's ability to rank
        positive cases above negative ones across all thresholds.
        0.5 = random baseline; 1.0 = perfect. Threshold-independent —
        more robust than accuracy for imbalanced problems.

    Parameters
    ----------
    model      : fitted sklearn classifier
    X_test     : scaled test feature matrix
    y_test     : ground-truth test labels
    model_name : display name for printout

    Returns
    -------
    dict of metric values (for comparison table)
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec  = recall_score(y_test, y_pred, pos_label=1)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(classification_report(
        y_test, y_pred, target_names=["No Diabetes", "Diabetes"]
    ))
    print(f"  ROC-AUC               : {auc:.4f}")
    print(f"  F1 (positive class)   : {f1_1:.4f}")
    print(f"  Precision (positive)  : {prec:.4f}")
    print(f"  Recall (positive)     : {rec:.4f}")

    return dict(
        model_name = model_name,
        accuracy   = acc,
        f1_weighted= f1_w,
        f1_positive= f1_1,
        precision  = prec,
        recall     = rec,
        roc_auc    = auc,
        y_pred     = y_pred,
        y_prob     = y_prob,
    )


# ── Comparison ────────────────────────────────────────────────────────────────

def comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Build and print a side-by-side metric comparison table.

    Parameters
    ----------
    results : list of dicts returned by evaluate_model()

    Returns
    -------
    DataFrame with one row per model, one column per metric
    """
    rows = []
    for r in results:
        rows.append({
            "Model":       r["model_name"],
            "Accuracy":    round(r["accuracy"],    4),
            "F1 (wtd)":    round(r["f1_weighted"],  4),
            "F1 (pos)":    round(r["f1_positive"],  4),
            "Precision":   round(r["precision"],    4),
            "Recall":      round(r["recall"],       4),
            "ROC-AUC":     round(r["roc_auc"],      4),
        })
    df = pd.DataFrame(rows).set_index("Model")
    print("\n=== Head-to-Head Comparison ===")
    print(df.to_string())
    df.to_csv("outputs/results_summary.csv")
    print("Saved → outputs/results_summary.csv")
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    results:  list[dict],
    y_test:   pd.Series,
) -> None:
    """
    Side-by-side confusion matrices for all models in results list.
    Annotated with raw counts. Blue colour scale.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"],
        )
        ax.set_title(f"{r['model_name']}\nConfusion Matrix",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrices.png", dpi=150)
    plt.show()
    print(f"Saved → {FIGURES_DIR}/confusion_matrices.png")


def plot_roc_curves(
    results: list[dict],
    y_test:  pd.Series,
) -> None:
    """
    Overlaid ROC curves for all models, with random baseline reference.
    Each curve labelled with its AUC score.
    """
    palette = ["steelblue", "tomato", "mediumpurple", "darkorange"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for r, colour in zip(results, palette):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=colour,
                label=f"{r['model_name']}  (AUC = {r['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/roc_comparison.png", dpi=150)
    plt.show()
    print(f"Saved → {FIGURES_DIR}/roc_comparison.png")
