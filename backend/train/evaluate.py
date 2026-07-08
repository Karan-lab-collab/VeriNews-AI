"""
evaluate.py – Metrics computation, reporting, and confusion matrix plotting.
"""
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train.config import (
    CONF_MATRIX_PATH,
    LABEL_NAMES,
    METRICS_PATH,
    REPORT_PATH,
)


def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Compute a full suite of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, f1.
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def print_report(y_true, y_pred) -> str:
    """
    Generate and print the full sklearn classification report.

    Returns
    -------
    str
        The report string (also printed to stdout).
    """
    target_names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)
    return report


def save_metrics(metrics: Dict[str, Any], path: Path = METRICS_PATH) -> None:
    """Save metrics dictionary as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved → {path}")


def save_report(report: str, path: Path = REPORT_PATH) -> None:
    """Save the classification report text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[evaluate] Report saved  → {path}")


def plot_confusion_matrix(
    y_true,
    y_pred,
    path: Path = CONF_MATRIX_PATH,
) -> None:
    """
    Plot and save a styled confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    path : Path
        Where to save the PNG image.
    """
    labels = [LABEL_NAMES[0], LABEL_NAMES[1]]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("VeriNews AI — Confusion Matrix", fontsize=14, fontweight="bold", pad=14)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {path}")


def run_full_evaluation(y_true, y_pred) -> Dict[str, Any]:
    """
    Convenience wrapper: compute metrics, print report, save everything.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Metrics dictionary.
    """
    metrics = compute_metrics(y_true, y_pred)

    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12} {v:.4f}")

    report = print_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    save_metrics(metrics)
    save_report(report)

    return metrics
