#!/usr/bin/env python3
"""
Multi-label G protein coupling prediction benchmark.

Trains independent binary classifiers for each of the four G protein families
(Gs, Gi/o, Gq/11, G12/13) using grouped k-fold CV, then reports per-target
and macro-averaged performance.

Usage:
    python scripts/run_multilabel.py [--n-folds 5] [--n-repeats 5]
"""
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.data.dataset import GPCRDataset, COUPLING_TARGETS
from src.features.handcrafted import build_feature_matrix
from src.features.bw_site import load_bw_cache, build_bw_feature_matrix
from src.splits.strategies import grouped_kfold_cv, repeated_grouped_kfold_cv
from src.models.classifiers import build_models
from src.evaluation.metrics import compute_fold_metrics, aggregate_cv_results

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_multilabel_cv(dataset, feature_matrices, n_folds=5, n_repeats=5):
    """Run multi-label CV across all targets and feature types."""
    models_dict = build_models(include_mlp=True, include_xgb=True)
    # Focus on key models for multi-label
    key_models = {k: v for k, v in models_dict.items()
                  if k in ("LR", "RF", "SVM", "MLP", "Ensemble")}

    all_results = []

    for target in COUPLING_TARGETS:
        y = dataset.get_labels(target)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        print(f"\n{'='*60}")
        print(f"TARGET: {target}  (pos={n_pos}, neg={n_neg})")
        print(f"{'='*60}")

        if n_pos < 5:
            print(f"  SKIP: too few positives")
            continue

        for feat_name, X in feature_matrices.items():
            print(f"\n  Feature: {feat_name} ({X.shape[1]}d)")

            # Subfamily-grouped CV
            cv_folds = repeated_grouped_kfold_cv(
                y, dataset.families, n_folds=n_folds, n_repeats=n_repeats
            )

            for model_name, model_template in key_models.items():
                fold_metrics = []
                for rep, fold_idx, tr_idx, te_idx in cv_folds:
                    X_tr, X_te = X[tr_idx], X[te_idx]
                    y_tr, y_te = y[tr_idx], y[te_idx]
                    if y_tr.sum() < 2 or y_te.sum() < 1:
                        continue
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)
                    model = clone(model_template)
                    try:
                        model.fit(X_tr_s, y_tr)
                        y_prob = model.predict_proba(X_te_s)[:, 1]
                        fm = compute_fold_metrics(y_te, y_prob)
                        fold_metrics.append(fm)
                    except Exception:
                        pass

                agg = aggregate_cv_results(fold_metrics)
                if agg:
                    result = {
                        "target": target,
                        "feature": feat_name,
                        "model": model_name,
                        "n_folds": len(fold_metrics),
                        "AUC_mean": agg["auc"]["mean"],
                        "AUC_std": agg["auc"]["std"],
                        "PRAUC_mean": agg.get("prauc", {}).get("mean", np.nan),
                        "PRAUC_std": agg.get("prauc", {}).get("std", np.nan),
                        "F1_mean": agg.get("f1", {}).get("mean", np.nan),
                        "F1_std": agg.get("f1", {}).get("std", np.nan),
                    }
                    all_results.append(result)
                    print(f"    {model_name:>10s}  AUC={result['AUC_mean']:.3f}±{result['AUC_std']:.3f}")

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-label G Protein Coupling Prediction Benchmark")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CV: {args.n_folds}-fold × {args.n_repeats} repeats (subfamily-grouped)")
    print("=" * 60)

    dataset = GPCRDataset().load()
    dataset.summary()

    bw_cache = load_bw_cache()
    X_hc, _ = build_feature_matrix(dataset.sequences, dataset.families)
    X_bw, _ = build_bw_feature_matrix(dataset.entry_names, bw_cache)
    X_comb = np.hstack([X_hc, X_bw])

    feature_matrices = {
        "handcrafted": X_hc,
        "bw_site": X_bw,
        "combined": X_comb,
    }

    results = run_multilabel_cv(dataset, feature_matrices,
                                 n_folds=args.n_folds, n_repeats=args.n_repeats)

    # Save
    df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"multilabel_cv_{ts}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    # Summary pivot
    print(f"\n{'='*60}")
    print("SUMMARY: Ensemble AUC (mean ± std) by target × feature")
    print(f"{'='*60}")
    ens = df[df["model"] == "Ensemble"]
    if not ens.empty:
        for _, row in ens.iterrows():
            print(f"  {row['target']:>4s} | {row['feature']:>12s} | "
                  f"AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}")

    # Macro average
    print(f"\nMacro-averaged AUC across all targets:")
    for feat in feature_matrices:
        sub = ens[ens["feature"] == feat]
        if not sub.empty:
            macro = sub["AUC_mean"].mean()
            print(f"  {feat:>12s}: {macro:.3f}")

    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
