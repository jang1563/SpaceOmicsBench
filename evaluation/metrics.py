#!/usr/bin/env python3
"""
Evaluation metrics for SpaceOmicsBench v2.

Supports:
  - Classification: accuracy, macro/weighted F1, per-class F1, confusion matrix
  - Binary (imbalanced): AUROC, AUPRC, precision@k, NDCG@k
  - Multi-label: micro F1, Hamming loss, per-label AUROC
  - Multi-class: macro F1, per-class F1
  - Direction concordance (for cross-biofluid tasks)
  - Bootstrap confidence intervals
"""

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    roc_auc_score,
)


# ─── Classification ───────────────────────────────────────────────────────────


def classification_metrics(y_true, y_pred, labels=None):
    """Standard classification metrics for sample-level tasks."""
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }

    # Per-class F1
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    if labels is not None:
        result["per_class_f1"] = {str(l): float(f) for l, f in zip(labels, per_class)}
    else:
        unique = sorted(set(y_true) | set(y_pred))
        result["per_class_f1"] = {str(l): float(f) for l, f in zip(unique, per_class)}

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result["confusion_matrix"] = cm.tolist()

    return result


# ─── Binary (imbalanced) ──────────────────────────────────────────────────────


def binary_metrics(y_true, y_score, threshold=0.5):
    """Metrics for imbalanced binary tasks (B1, C1, C2, E1-E4)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    result = {}

    # AUROC
    try:
        result["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        result["auroc"] = float("nan")

    # AUPRC
    try:
        result["auprc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        result["auprc"] = float("nan")

    # Hard predictions at threshold
    y_pred = (y_score >= threshold).astype(int)
    result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Positive rate
    result["positive_rate"] = float(y_true.mean())
    result["n_positive"] = int(y_true.sum())
    result["n_total"] = len(y_true)

    return result


def ranking_metrics(y_true, y_score, k_values=(50, 100)):
    """Ranking metrics: precision@k, NDCG@k."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Sort by descending score
    ranked_idx = np.argsort(-y_score)
    y_ranked = y_true[ranked_idx]

    result = {}

    for k in k_values:
        k_actual = min(k, len(y_true))

        # Precision@k
        result[f"precision_at_{k}"] = float(y_ranked[:k_actual].sum() / k_actual)

        # NDCG@k
        dcg = np.sum(y_ranked[:k_actual] / np.log2(np.arange(2, k_actual + 2)))
        # Ideal DCG: all positives first
        n_pos = int(y_true.sum())
        ideal_top = min(n_pos, k_actual)
        idcg = np.sum(1.0 / np.log2(np.arange(2, ideal_top + 2)))
        result[f"ndcg_at_{k}"] = float(dcg / idcg) if idcg > 0 else 0.0

    return result


# ─── Multi-label ───────────────────────────────────────────────────────────────


def multilabel_metrics(y_true, y_pred, y_score=None, label_names=None):
    """Multi-label metrics for B2 (gene cluster prediction).

    Args:
        y_true: (N, L) binary matrix
        y_pred: (N, L) binary predictions
        y_score: (N, L) probability scores (optional, for per-label AUROC)
        label_names: list of label names
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    result = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    # Per-label AUROC (if scores provided)
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        n_labels = y_true.shape[1]
        per_label_auroc = {}
        valid_aurocs = []

        for i in range(n_labels):
            name = label_names[i] if label_names else f"label_{i}"
            if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true[:, i]):
                try:
                    auc = float(roc_auc_score(y_true[:, i], y_score[:, i]))
                    per_label_auroc[name] = auc
                    valid_aurocs.append(auc)
                except ValueError:
                    per_label_auroc[name] = float("nan")
            else:
                per_label_auroc[name] = float("nan")

        result["per_label_auroc"] = per_label_auroc
        if valid_aurocs:
            result["mean_auroc"] = float(np.mean(valid_aurocs))

    return result


# ─── Multi-class (feature-level) ──────────────────────────────────────────────


def multiclass_metrics(y_true, y_pred, labels=None):
    """Multi-class metrics for D1 (metabolite pathway classification)."""
    return classification_metrics(y_true, y_pred, labels=labels)


# ─── Direction concordance ─────────────────────────────────────────────────────


def direction_concordance(logfc_a, logfc_b):
    """Compute direction concordance between two sets of logFC values.

    Used for C2: plasma vs EVP DE direction agreement.
    """
    logfc_a = np.asarray(logfc_a, dtype=float)
    logfc_b = np.asarray(logfc_b, dtype=float)

    # Remove NaN pairs
    valid = ~(np.isnan(logfc_a) | np.isnan(logfc_b))
    a, b = logfc_a[valid], logfc_b[valid]

    if len(a) == 0:
        return {"direction_concordance": float("nan"), "n_valid": 0}

    same_dir = np.sum(np.sign(a) == np.sign(b))
    return {
        "direction_concordance": float(same_dir / len(a)),
        "n_valid": int(len(a)),
        "n_concordant": int(same_dir),
    }


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────


def bootstrap_ci(y_true, y_pred_or_score, metric_fn, n_bootstrap=1000,
                 ci=0.95, seed=42, **metric_kwargs):
    """Bootstrap confidence interval for any metric.

    Args:
        y_true: ground truth
        y_pred_or_score: predictions or scores
        metric_fn: function(y_true, y_pred) → dict with metric values
        n_bootstrap: number of bootstrap samples
        ci: confidence level (default 0.95)
        seed: random seed
        **metric_kwargs: additional kwargs for metric_fn

    Returns:
        dict with mean, ci_lower, ci_upper for each metric
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    y_true = np.asarray(y_true)
    y_pred_or_score = np.asarray(y_pred_or_score)

    # Collect bootstrap results
    boot_results = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            result = metric_fn(y_true[idx], y_pred_or_score[idx], **metric_kwargs)
            boot_results.append(result)
        except (ValueError, ZeroDivisionError):
            continue

    if not boot_results:
        warnings.warn("All bootstrap samples failed")
        return {}

    # Aggregate
    alpha = (1 - ci) / 2
    output = {}
    keys = [k for k in boot_results[0] if isinstance(boot_results[0][k], (int, float))]
    for key in keys:
        vals = [r[key] for r in boot_results if key in r and not np.isnan(r[key])]
        if vals:
            output[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_lower": float(np.percentile(vals, alpha * 100)),
                "ci_upper": float(np.percentile(vals, (1 - alpha) * 100)),
            }

    return output
