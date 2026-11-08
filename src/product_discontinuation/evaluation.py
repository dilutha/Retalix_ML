"""Metrics, threshold selection and evaluation plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score,
    average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob have different lengths")
    if len(np.unique(y_true)) < 2:
        raise ValueError("metrics need both classes present")
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def threshold_table(y_true, y_prob, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    rows = [classification_metrics(y_true, y_prob, t) for t in thresholds]
    return pd.DataFrame(rows)[["threshold", "precision", "recall", "f1", "balanced_accuracy",
                               "tp", "fp", "fn", "tn"]]


def segment_metrics(y_true, y_prob, segments, threshold):
    """Metrics per segment (e.g. new vs returning products)."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob),
                          "segment": np.asarray(segments)})
    rows = []
    for name, part in frame.groupby("segment"):
        if part["y"].nunique() < 2:
            continue
        metrics = classification_metrics(part["y"], part["p"], threshold)
        rows.append({"segment": name, **metrics})
    return pd.DataFrame(rows)


def grouped_permutation_importance(pipe, X, y, n_repeats=5, random_state=42, max_rows=10000):
    """Permutation importance on raw input features (before one-hot encoding)."""
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=random_state)
        y = y.loc[X.index]
    result = permutation_importance(pipe, X, y, scoring="average_precision",
                                    n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)


def _save(fig, path):
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_confusion_matrix(y_true, y_prob, threshold, title, path=None):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (np.asarray(y_prob) >= threshold).astype(int),
        display_labels=["Continued", "Discontinued"], cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"{title}\n(threshold = {threshold:.2f})")
    return _save(fig, path)


def plot_roc_pr(y_true, probabilities, path=None, baselines=None):
    """ROC and precision-recall curves for several models on one figure."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, prob in probabilities.items():
        RocCurveDisplay.from_predictions(y_true, prob, name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y_true, prob, name=name, ax=ax_pr)
    for name, prob in (baselines or {}).items():
        RocCurveDisplay.from_predictions(y_true, prob, name=name, ax=ax_roc,
                                         curve_kwargs={"linestyle": "--"})
        PrecisionRecallDisplay.from_predictions(y_true, prob, name=name, ax=ax_pr, linestyle="--")
    ax_roc.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle=":")
    ax_pr.axhline(np.mean(y_true), color="grey", lw=0.8, linestyle=":", label="No skill")
    ax_roc.set_title("ROC curve")
    ax_pr.set_title("Precision-recall curve")
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_roc.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_calibration(y_true, probabilities, path=None, n_bins=10):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, prob in probabilities.items():
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed discontinuation rate")
    ax.set_title("Calibration curve (raw model scores)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_feature_importance(importance, title, path=None, top=15):
    top_rows = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(top_rows) + 1))
    ax.barh(top_rows["feature"], top_rows["importance_mean"], xerr=top_rows["importance_std"],
            color="#4C72B0")
    ax.set_xlabel("Drop in PR-AUC when permuted")
    ax.set_title(title)
    return _save(fig, path)


def plot_threshold_tradeoff(y_true, y_prob, chosen, path=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.plot(thresholds, f1[:-1], label="F1")
    ax.axvline(chosen, color="black", linestyle="--", lw=1, label=f"Chosen = {chosen:.2f}")
    ax.set_xlabel("Probability threshold")
    ax.set_title("Threshold trade-off (raw scores, validation predictions)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_horizon_comparison(table, path=None, metric="pr_auc"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, part in table.groupby("model"):
        part = part.sort_values("horizon")
        style = "--" if "status" in name else "-"
        ax.plot(part["horizon"], part[metric], marker="o", linestyle=style, label=name)
    ax.invert_xaxis()
    ax.set_xlabel("Weeks before range refresh (prediction horizon)")
    ax.set_ylabel(metric.upper().replace("_", "-"))
    ax.set_title(f"Test {metric.upper().replace('_', '-')} by prediction horizon")
    ax.legend(fontsize=8, title="dashed = uses Status (benchmark)", title_fontsize=8)
    return _save(fig, path)
