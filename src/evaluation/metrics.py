"""
Evaluation metrics with bootstrap confidence intervals and statistical tests.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score,
)


def bootstrap_metrics(y_true, y_prob, n_boot=1000, seed=42):
    """Compute AUC-ROC, PR-AUC, Accuracy, F1 with bootstrap 95% CI.

    Returns
    -------
    dict with keys 'auc', 'prauc', 'acc', 'f1', each -> (mean, lo, hi).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_pred = (y_prob >= 0.5).astype(int)

    aucs, praucs, accs, f1s = [], [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp, ypd = y_true[idx], y_prob[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
            praucs.append(average_precision_score(yt, yp))
            accs.append(accuracy_score(yt, ypd))
            f1s.append(f1_score(yt, ypd, zero_division=0))
        except Exception:
            pass

    def ci(arr):
        if len(arr) < 10:
            return np.nan, np.nan, np.nan
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "auc": ci(aucs),
        "prauc": ci(praucs),
        "acc": ci(accs),
        "f1": ci(f1s),
    }


def delong_permutation_test(y_true, y_prob1, y_prob2, n_perm=5000, seed=42):
    """Permutation-based test for the difference of two AUCs on the same data.

    Returns
    -------
    float : two-sided p-value
    """
    try:
        auc1 = roc_auc_score(y_true, y_prob1)
        auc2 = roc_auc_score(y_true, y_prob2)
    except ValueError:
        return np.nan
    diff = auc1 - auc2
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(len(y_true)) > 0.5
        p1 = np.where(swap, y_prob2, y_prob1)
        p2 = np.where(swap, y_prob1, y_prob2)
        try:
            d = roc_auc_score(y_true, p1) - roc_auc_score(y_true, p2)
            if abs(d) >= abs(diff):
                count += 1
        except Exception:
            pass
    return count / n_perm


def aggregate_cv_results(fold_results):
    """Aggregate per-fold metric dicts into mean ± std summary.

    Parameters
    ----------
    fold_results : list of dict
        Each dict has keys like 'auc', 'prauc', 'acc', 'f1',
        where each value is a float (point estimate per fold).

    Returns
    -------
    dict : metric_name -> {'mean': float, 'std': float, 'values': list}
    """
    if not fold_results:
        return {}
    keys = fold_results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in fold_results if not np.isnan(r.get(k, np.nan))]
        agg[k] = {
            "mean": float(np.mean(vals)) if vals else np.nan,
            "std": float(np.std(vals)) if vals else np.nan,
            "values": vals,
        }
    return agg


def compute_fold_metrics(y_true, y_prob):
    """Compute point-estimate metrics for a single fold.

    Returns
    -------
    dict : metric_name -> float
    """
    if len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "prauc": np.nan, "acc": np.nan, "f1": np.nan}
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    try:
        prauc = average_precision_score(y_true, y_prob)
    except ValueError:
        prauc = np.nan
    return {
        "auc": float(auc),
        "prauc": float(prauc),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
