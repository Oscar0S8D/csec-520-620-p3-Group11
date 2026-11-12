"""Evaluation metrics including Recall @ fixed analyst workload and FPR."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, workload_rate: float = 0.05) -> dict:
    # 1. Fixed analyst review workload (flag top k = 5% highest-risk transactions)
    k = max(1, int(np.ceil(len(y_true) * workload_rate)))
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    y_pred_workload = np.zeros_like(y_true)
    y_pred_workload[top_k_idx] = 1

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_workload, labels=[0, 1]).ravel()
    recall_at_workload = tp / max(1, (tp + fn))
    precision_at_workload = tp / max(1, (tp + fp))
    fpr_at_workload = fp / max(1, (fp + tn))
    f1_at_workload = (
        2 * precision_at_workload * recall_at_workload
        / max(1e-12, (precision_at_workload + recall_at_workload))
    )

    # 2. Standard 0.5 threshold metrics for reference
    y_pred_default = (y_prob >= 0.5).astype(int)

    return {
        "workload_rate": float(workload_rate),
        "recall_at_workload": float(recall_at_workload),
        "precision_at_workload": float(precision_at_workload),
        "fpr_at_workload": float(fpr_at_workload),
        "f1_at_workload": float(f1_at_workload),
        "meets_success_criteria": bool(recall_at_workload >= 0.80 and fpr_at_workload <= 0.05),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy_05": float(accuracy_score(y_true, y_pred_default)),
        "precision_05": float(precision_score(y_true, y_pred_default, zero_division=0)),
        "recall_05": float(recall_score(y_true, y_pred_default, zero_division=0)),
        "f1_05": float(f1_score(y_true, y_pred_default, zero_division=0)),
    }


def save_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: str, workload_rate: float = 0.05
) -> None:
    k = max(1, int(np.ceil(len(y_true) * workload_rate)))
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    y_pred = np.zeros_like(y_true)
    y_pred[top_k_idx] = 1

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (@ {int(workload_rate * 100)}% Workload)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: str, model_name: str = "Model") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title(f"ROC Curve ({model_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_combined_roc_curve(roc_entries: list[dict], out_path: str) -> None:
    """Plots multiple models' ROC curves onto a single comparison figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    linestyles = ["-", "-", "-.", "--", ":"]

    for i, entry in enumerate(roc_entries):
        disp = RocCurveDisplay.from_predictions(
            entry["y_true"],
            entry["y_prob"],
            ax=ax,
            name=entry["label"],
        )
        disp.line_.set(
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2,
            alpha=0.85,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance (AUC = 0.50)")
    ax.set_title("ROC Curve Comparison Across Baselines & Primary Models")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)