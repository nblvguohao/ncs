#!/usr/bin/env python3
"""
NCS-level benchmark: Grouped k-fold CV for GPCR–G protein coupling prediction.

Key upgrades over BIB version:
  - Repeated grouped k-fold CV (5-fold × 10 repeats) instead of single split
  - Multi-target evaluation (Gs, Gi, Gq, G12/13)
  - MLP neural network baseline
  - Three feature representations: handcrafted, BW-site, combined
  - Statistical comparison between split strategies

Usage:
    python scripts/run_benchmark.py [--target Gq] [--n-folds 5] [--n-repeats 10]
"""
import os
import sys
import time
import argparse
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from leakageguard.data.dataset import GPCRDataset, COUPLING_TARGETS
from leakageguard.features.handcrafted import build_feature_matrix
from leakageguard.features.bw_site import load_bw_cache, build_bw_feature_matrix
from leakageguard.splits.strategies import (
    random_split, subfamily_split, seqcluster_split,
    grouped_kfold_cv, repeated_grouped_kfold_cv, seqcluster_kfold_cv,
)
from leakageguard.models.classifiers import build_models
from leakageguard.evaluation.metrics import (
    bootstrap_metrics, compute_fold_metrics, aggregate_cv_results,
    delong_permutation_test,
)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single_split_benchmark(X, y, families, sequences, models, split_name, split_fn,
                                feature_name="handcrafted"):
    """Run all models on a single train/test split. Returns list of result dicts."""
    train_idx, test_idx = split_fn()
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    if y_tr.sum() < 3 or (len(y_tr) - y_tr.sum()) < 3:
        print(f"  SKIP {split_name}: insufficient positive/negative in train")
        return []
    if y_te.sum() < 2 or (len(y_te) - y_te.sum()) < 2:
        print(f"  SKIP {split_name}: insufficient positive/negative in test")
        return []

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    results = []
    for model_name, model_template in models.items():
        model = clone(model_template)
        t0 = time.time()
        model.fit(X_tr_s, y_tr)
        train_time = time.time() - t0

        y_prob = model.predict_proba(X_te_s)[:, 1]
        bm = bootstrap_metrics(y_te, y_prob)

        results.append({
            "feature": feature_name,
            "split": split_name,
            "model": model_name,
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "train_time": train_time,
            "AUC": bm["auc"][0],
            "AUC_lo": bm["auc"][1],
            "AUC_hi": bm["auc"][2],
            "PR_AUC": bm["prauc"][0],
            "PRAUC_lo": bm["prauc"][1],
            "PRAUC_hi": bm["prauc"][2],
            "Accuracy": bm["acc"][0],
            "F1": bm["f1"][0],
        })
        print(f"    {model_name:>10s}  AUC={bm['auc'][0]:.3f} [{bm['auc'][1]:.3f}-{bm['auc'][2]:.3f}]  "
              f"PR-AUC={bm['prauc'][0]:.3f}  t={train_time:.1f}s")

    return results


def run_cv_benchmark(X, y, families, sequences, models, cv_strategy, cv_name,
                     feature_name="handcrafted", n_folds=5, n_repeats=10):
    """Run repeated grouped k-fold CV. Returns list of result dicts."""
    print(f"\n  CV strategy: {cv_name} ({n_folds}-fold × {n_repeats} repeats)")

    if cv_name == "subfamily_cv":
        cv_folds = repeated_grouped_kfold_cv(y, families, n_folds=n_folds,
                                              n_repeats=n_repeats)
    elif cv_name == "seqcluster_cv_0.3":
        cv_folds = []
        for rep in range(n_repeats):
            folds = seqcluster_kfold_cv(y, sequences, threshold=0.3,
                                         n_folds=n_folds, seed=42 + rep * 1000)
            for fi, (tr, te) in enumerate(folds):
                cv_folds.append((rep, fi, tr, te))
    elif cv_name == "random_cv":
        cv_folds = []
        for rep in range(n_repeats):
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                   random_state=42 + rep * 1000)
            for fi, (tr, te) in enumerate(skf.split(X, y)):
                cv_folds.append((rep, fi, tr, te))
    else:
        raise ValueError(f"Unknown CV strategy: {cv_name}")

    total_folds = len(cv_folds)
    print(f"    Total folds: {total_folds}")

    results_by_model = {name: [] for name in models}

    for idx, (rep, fold_idx, train_idx, test_idx) in enumerate(cv_folds):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if y_tr.sum() < 3 or (len(y_tr) - y_tr.sum()) < 3:
            continue
        if y_te.sum() < 2 or (len(y_te) - y_te.sum()) < 2:
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
                fm["repeat"] = rep
                fm["fold"] = fold_idx
                results_by_model[model_name].append(fm)
            except Exception as e:
                pass

        if (idx + 1) % (n_folds * 2) == 0:
            print(f"    Completed {idx+1}/{total_folds} folds...")

    # Aggregate
    all_results = []
    for model_name, fold_metrics in results_by_model.items():
        agg = aggregate_cv_results(fold_metrics)
        if not agg:
            continue
        result = {
            "feature": feature_name,
            "split": cv_name,
            "model": model_name,
            "n_folds": len(fold_metrics),
            "AUC_mean": agg["auc"]["mean"],
            "AUC_std": agg["auc"]["std"],
            "PRAUC_mean": agg.get("prauc", {}).get("mean", np.nan),
            "PRAUC_std": agg.get("prauc", {}).get("std", np.nan),
            "Acc_mean": agg.get("acc", {}).get("mean", np.nan),
            "F1_mean": agg.get("f1", {}).get("mean", np.nan),
            "F1_std": agg.get("f1", {}).get("std", np.nan),
        }
        all_results.append(result)
        print(f"    {model_name:>10s}  AUC={result['AUC_mean']:.3f}±{result['AUC_std']:.3f}  "
              f"PR-AUC={result['PRAUC_mean']:.3f}±{result['PRAUC_std']:.3f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="NCS-level GPCR coupling benchmark")
    parser.add_argument("--target", type=str, default="all",
                        help="Coupling target: Gs, Gi, Gq, G12, or 'all'")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--skip-seqcluster", action="store_true",
                        help="Skip slow sequence clustering splits")
    args = parser.parse_args()

    print("=" * 72)
    print("NCS Benchmark: Leakage-Aware GPCR–G Protein Coupling Evaluation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CV: {args.n_folds}-fold × {args.n_repeats} repeats")
    print("=" * 72)

    # ── Load data ────────────────────────────────────────────────────────
    dataset = GPCRDataset().load()
    dataset.summary()

    # ── Load BW cache ────────────────────────────────────────────────────
    bw_cache = load_bw_cache()
    print(f"BW cache: {len(bw_cache)} entries")

    # ── Build features ───────────────────────────────────────────────────
    print("\nExtracting features...")
    X_hc, hc_names = build_feature_matrix(dataset.sequences, dataset.families)
    X_bw, bw_names = build_bw_feature_matrix(dataset.entry_names, bw_cache)
    X_comb = np.hstack([X_hc, X_bw])
    comb_names = hc_names + bw_names
    print(f"  Handcrafted: {X_hc.shape}")
    print(f"  BW-site:     {X_bw.shape}")
    print(f"  Combined:    {X_comb.shape}")

    # ── Determine targets ────────────────────────────────────────────────
    if args.target == "all":
        targets = COUPLING_TARGETS
    else:
        targets = [args.target]

    # ── Build models ─────────────────────────────────────────────────────
    models = build_models(include_mlp=True, include_xgb=True)
    print(f"\nModels: {list(models.keys())}")

    all_single_results = []
    all_cv_results = []

    for target in targets:
        y = dataset.get_labels(target)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        print(f"\n{'='*72}")
        print(f"TARGET: {target}  (pos={n_pos}, neg={n_neg})")
        print(f"{'='*72}")

        if n_pos < 5:
            print(f"  SKIP {target}: too few positives ({n_pos})")
            continue

        feature_sets = {
            "handcrafted": X_hc,
            "bw_site": X_bw,
            "combined": X_comb,
        }

        for feat_name, X in feature_sets.items():
            print(f"\n  Feature: {feat_name} ({X.shape[1]}d)")

            # ── Single-split benchmarks ──────────────────────────────
            print(f"\n  --- Single-split benchmarks ---")

            # Random split
            print(f"  [Random split]")
            res = run_single_split_benchmark(
                X, y, dataset.families, dataset.sequences, models,
                "Random", lambda: random_split(y), feat_name,
            )
            for r in res:
                r["target"] = target
            all_single_results.extend(res)

            # Subfamily split
            print(f"  [Subfamily split]")
            res = run_single_split_benchmark(
                X, y, dataset.families, dataset.sequences, models,
                "Subfamily", lambda: subfamily_split(y, dataset.families), feat_name,
            )
            for r in res:
                r["target"] = target
            all_single_results.extend(res)

            # Sequence cluster splits (optional)
            if not args.skip_seqcluster:
                for thresh in [0.4, 0.3]:
                    print(f"  [SeqCluster t={thresh}]")
                    res = run_single_split_benchmark(
                        X, y, dataset.families, dataset.sequences, models,
                        f"SeqCluster_{thresh}",
                        lambda t=thresh: seqcluster_split(y, dataset.sequences, threshold=t),
                        feat_name,
                    )
                    for r in res:
                        r["target"] = target
                    all_single_results.extend(res)

            # ── Cross-validation benchmarks ──────────────────────────
            print(f"\n  --- Cross-validation benchmarks ---")

            # Random CV (baseline)
            cv_res = run_cv_benchmark(
                X, y, dataset.families, dataset.sequences, models,
                "random_cv", "random_cv", feat_name,
                n_folds=args.n_folds, n_repeats=args.n_repeats,
            )
            for r in cv_res:
                r["target"] = target
            all_cv_results.extend(cv_res)

            # Subfamily-grouped CV (no-leak)
            cv_res = run_cv_benchmark(
                X, y, dataset.families, dataset.sequences, models,
                "subfamily_cv", "subfamily_cv", feat_name,
                n_folds=args.n_folds, n_repeats=args.n_repeats,
            )
            for r in cv_res:
                r["target"] = target
            all_cv_results.extend(cv_res)

    # ── Save results ─────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_single_results:
        df_single = pd.DataFrame(all_single_results)
        single_path = os.path.join(RESULTS_DIR, f"benchmark_single_split_{timestamp}.csv")
        df_single.to_csv(single_path, index=False)
        print(f"\nSingle-split results saved: {single_path}")

    if all_cv_results:
        df_cv = pd.DataFrame(all_cv_results)
        cv_path = os.path.join(RESULTS_DIR, f"benchmark_cv_{timestamp}.csv")
        df_cv.to_csv(cv_path, index=False)
        print(f"CV results saved: {cv_path}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY: Ensemble AUC (mean ± std) by target × feature × split")
    print(f"{'='*72}")
    if all_cv_results:
        df_cv = pd.DataFrame(all_cv_results)
        ens = df_cv[df_cv["model"] == "Ensemble"]
        if not ens.empty:
            pivot = ens.pivot_table(
                values="AUC_mean", index=["target", "feature"],
                columns="split", aggfunc="first",
            )
            print(pivot.round(3).to_string())

    print(f"\nBenchmark completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
