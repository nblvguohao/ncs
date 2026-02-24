#!/usr/bin/env python3
"""
Standalone leakage diagnostic tool.

Accepts user-supplied coupling predictions (CSV) and evaluates performance
degradation under no-leak splitting, outputting a diagnostic report.

Usage:
    python scripts/leakage_test.py                          # default dataset
    python scripts/leakage_test.py --predictions user.csv   # user predictions
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from leakageguard.data.dataset import GPCRDataset
from leakageguard.features.handcrafted import build_feature_matrix
from leakageguard.features.bw_site import load_bw_cache, build_bw_feature_matrix
from leakageguard.splits.strategies import random_split, subfamily_split, grouped_kfold_cv
from leakageguard.models.classifiers import build_models
from leakageguard.evaluation.metrics import bootstrap_metrics, compute_fold_metrics, aggregate_cv_results
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler


def run_leakage_diagnostic(dataset, X, feature_name, target="Gq", n_folds=5):
    """Run leakage diagnostic: compare random vs no-leak performance."""
    y = dataset.get_labels(target)
    models = {"Ensemble": build_models()["Ensemble"]}

    print(f"\n{'='*60}")
    print(f"Leakage Diagnostic: {target} coupling ({feature_name} features)")
    print(f"{'='*60}")
    print(f"Samples: {len(y)} (pos={int(y.sum())}, neg={len(y)-int(y.sum())})")

    results = {}

    for split_name, cv_fn in [
        ("Random (leaky)", lambda: _run_cv(X, y, dataset, models, "random", n_folds)),
        ("Subfamily (no-leak)", lambda: _run_cv(X, y, dataset, models, "subfamily", n_folds)),
    ]:
        print(f"\n  {split_name}:")
        fold_metrics = cv_fn()
        agg = aggregate_cv_results(fold_metrics)
        auc_mean = agg["auc"]["mean"]
        auc_std = agg["auc"]["std"]
        print(f"    AUC-ROC: {auc_mean:.3f} ± {auc_std:.3f}")
        print(f"    PR-AUC:  {agg['prauc']['mean']:.3f} ± {agg['prauc']['std']:.3f}")
        results[split_name] = auc_mean

    delta = results.get("Random (leaky)", 0) - results.get("Subfamily (no-leak)", 0)
    print(f"\n  ΔAUC (leaky - no-leak): {delta:+.3f}")
    if delta > 0.05:
        print(f"  ⚠️  WARNING: Significant leakage detected (ΔAUC = {delta:.3f})")
    else:
        print(f"  ✓  No significant leakage detected")

    return results


def _run_cv(X, y, dataset, models, strategy, n_folds):
    """Run k-fold CV with given strategy."""
    if strategy == "random":
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = [(tr, te) for tr, te in skf.split(X, y)]
    else:
        folds = grouped_kfold_cv(y, dataset.families, n_folds=n_folds, seed=42)

    fold_metrics = []
    for tr_idx, te_idx in folds:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if y_tr.sum() < 2 or y_te.sum() < 1:
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        for model_name, model_template in models.items():
            model = clone(model_template)
            try:
                model.fit(X_tr_s, y_tr)
                y_prob = model.predict_proba(X_te_s)[:, 1]
                fm = compute_fold_metrics(y_te, y_prob)
                fold_metrics.append(fm)
            except Exception:
                pass
    return fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Leakage diagnostic tool")
    parser.add_argument("--predictions", type=str, default=None,
                        help="CSV with user predictions (columns: entry_name, prediction)")
    parser.add_argument("--target", type=str, default="Gq")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    dataset = GPCRDataset().load()
    bw_cache = load_bw_cache()

    X_hc, _ = build_feature_matrix(dataset.sequences, dataset.families)
    X_bw, _ = build_bw_feature_matrix(dataset.entry_names, bw_cache)

    print("\n" + "="*60)
    print("GPCR Coupling Leakage Diagnostic Report")
    print("="*60)

    for feat_name, X in [("handcrafted", X_hc), ("bw_site", X_bw)]:
        run_leakage_diagnostic(dataset, X, feat_name, target=args.target,
                                n_folds=args.n_folds)

    print(f"\n{'='*60}")
    print("Diagnostic complete.")
    print("="*60)


if __name__ == "__main__":
    main()
