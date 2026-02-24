#!/usr/bin/env python3
"""
ESM-2 attention comparison and leakage gradient quantification.

1. Extracts ESM-2 attention weights at BW contact sites
2. Compares attention distributions between Gq-coupled vs non-Gq receptors
3. Quantifies leakage gradient with ESM-2 embeddings across identity thresholds
4. Generates publication-ready comparison plots

Usage:
    python scripts/run_esm2_leakage.py [--model esm2_t6_8M_UR50D] [--device cpu]
"""
import os
import sys
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from leakageguard.data.dataset import GPCRDataset
from leakageguard.features.bw_site import (
    load_bw_cache, build_bw_feature_matrix, GP_CONTACT_SITES,
)
from leakageguard.splits.strategies import (
    grouped_kfold_cv, seqcluster_kfold_cv,
)
from leakageguard.models.classifiers import build_models
from leakageguard.evaluation.metrics import compute_fold_metrics, aggregate_cv_results

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Part 1: ESM-2 Attention at BW sites ──────────────────────────────────

def run_attention_comparison(dataset, bw_cache, model_name, device, target="Gq"):
    """Extract ESM-2 attention at BW sites, compare Gq vs non-Gq."""
    try:
        from leakageguard.features.esm2 import extract_esm2_attention
    except ImportError:
        print("  [SKIP] ESM-2 not available (install fair-esm + torch)")
        return None

    print(f"\n{'='*64}")
    print(f"  ESM-2 Attention Analysis at BW Contact Sites")
    print(f"  Model: {model_name}  |  Target: {target}")
    print(f"{'='*64}")

    y = dataset.get_labels(target)
    gq_idx = np.where(y == 1)[0]
    non_gq_idx = np.where(y == 0)[0]

    print(f"  Extracting attention for {len(dataset.sequences)} receptors...")
    attn_matrix, site_names = extract_esm2_attention(
        dataset.sequences, dataset.entry_names, bw_cache,
        model_name=model_name, device=device,
    )

    # Statistical comparison per BW site
    results = []
    print(f"\n  {'BW Site':>10s}  {'Gq mean':>8s}  {'nonGq mean':>10s}  {'t-stat':>8s}  {'p-value':>8s}")
    print(f"  {'-'*52}")
    for i, site in enumerate(GP_CONTACT_SITES):
        gq_attn = attn_matrix[gq_idx, i]
        non_gq_attn = attn_matrix[non_gq_idx, i]
        t_stat, p_val = stats.ttest_ind(gq_attn, non_gq_attn, equal_var=False)
        effect_size = (gq_attn.mean() - non_gq_attn.mean()) / np.sqrt(
            (gq_attn.std()**2 + non_gq_attn.std()**2) / 2
        ) if (gq_attn.std() + non_gq_attn.std()) > 0 else 0.0
        results.append({
            "bw_site": site,
            "gq_mean": float(gq_attn.mean()),
            "gq_std": float(gq_attn.std()),
            "non_gq_mean": float(non_gq_attn.mean()),
            "non_gq_std": float(non_gq_attn.std()),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(effect_size),
        })
        sig = "*" if p_val < 0.05 else ""
        print(f"  {site:>10s}  {gq_attn.mean():8.4f}  {non_gq_attn.mean():10.4f}  "
              f"{t_stat:8.3f}  {p_val:8.4f} {sig}")

    df = pd.DataFrame(results)
    n_sig = (df["p_value"] < 0.05).sum()
    print(f"\n  Nominally significant sites (p<0.05): {n_sig}/{len(GP_CONTACT_SITES)}")

    return df, attn_matrix


# ── Part 2: Leakage Gradient with ESM-2 Features ─────────────────────────

def run_esm2_leakage_gradient(dataset, bw_cache, model_name, device, target="Gq"):
    """Quantify leakage gradient using ESM-2 BW-site embeddings."""

    # Try precomputed embeddings first
    precomputed_paths = [
        os.path.join(PROJECT_DIR, "data", "esm2_bw_embeddings.npz"),
        os.path.join(PROJECT_DIR, "data", "esm2_35m_bw_embeddings.npz"),
    ]

    X_esm = None
    for pp in precomputed_paths:
        if os.path.exists(pp):
            print(f"  Loading precomputed ESM-2 embeddings: {pp}")
            data = np.load(pp, allow_pickle=True)
            X_esm = data["embeddings"] if "embeddings" in data else data["X"]
            break

    if X_esm is None:
        try:
            from leakageguard.features.esm2 import extract_esm2_bw_embeddings
            print(f"  Computing ESM-2 BW embeddings ({model_name})...")
            X_esm, _ = extract_esm2_bw_embeddings(
                dataset.sequences, dataset.entry_names, bw_cache,
                model_name=model_name, device=device,
            )
        except ImportError:
            print("  [SKIP] ESM-2 not available")
            return None

    print(f"\n{'='*64}")
    print(f"  ESM-2 Leakage Gradient Quantification")
    print(f"  Feature dims: {X_esm.shape[1]}  |  Target: {target}")
    print(f"{'='*64}")

    y = dataset.get_labels(target)
    models = {"Ensemble": build_models()["Ensemble"]}

    # Also get BW-site physicochemical features for comparison
    X_bw, _ = build_bw_feature_matrix(dataset.entry_names, bw_cache)

    results = []

    # 1. Random CV
    for feat_name, X in [("ESM-2 BW", X_esm), ("BW-site physchem", X_bw)]:
        print(f"\n  Feature: {feat_name}")

        # Random CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = _run_cv(X, y, list(skf.split(X, y)), models)
        agg = aggregate_cv_results(fold_metrics)
        auc_r = agg["auc"]["mean"]
        print(f"    Random CV:     AUC = {auc_r:.3f} +/- {agg['auc']['std']:.3f}")
        results.append({"feature": feat_name, "split": "random_cv",
                        "threshold": None, "AUC_mean": auc_r,
                        "AUC_std": agg["auc"]["std"]})

        # Subfamily CV
        folds = grouped_kfold_cv(y, dataset.families, n_folds=5, seed=42)
        fold_metrics = _run_cv(X, y, folds, models)
        agg = aggregate_cv_results(fold_metrics)
        auc_s = agg["auc"]["mean"]
        print(f"    Subfamily CV:  AUC = {auc_s:.3f} +/- {agg['auc']['std']:.3f}")
        results.append({"feature": feat_name, "split": "subfamily_cv",
                        "threshold": None, "AUC_mean": auc_s,
                        "AUC_std": agg["auc"]["std"]})

        # Sequence cluster CV at multiple thresholds
        for thresh in [0.6, 0.5, 0.4, 0.3, 0.2]:
            try:
                folds = seqcluster_kfold_cv(y, dataset.sequences,
                                             threshold=thresh, n_folds=5, seed=42)
                fold_metrics = _run_cv(X, y, folds, models)
                agg = aggregate_cv_results(fold_metrics)
                auc_t = agg["auc"]["mean"]
                print(f"    SeqCluster {thresh:.1f}: AUC = {auc_t:.3f} +/- {agg['auc']['std']:.3f}")
                results.append({"feature": feat_name, "split": f"seqcluster_{thresh}",
                                "threshold": thresh, "AUC_mean": auc_t,
                                "AUC_std": agg["auc"]["std"]})
            except Exception as e:
                print(f"    SeqCluster {thresh:.1f}: FAILED ({e})")

    return pd.DataFrame(results)


def _run_cv(X, y, folds, models):
    """Helper: run CV and return fold metrics."""
    fold_metrics = []
    for tr_idx, te_idx in folds:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        if y_tr.sum() < 2 or y_te.sum() < 1:
            continue
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        for mname, mtpl in models.items():
            m = clone(mtpl)
            try:
                m.fit(X_tr_s, y_tr)
                yp = m.predict_proba(X_te_s)[:, 1]
                fold_metrics.append(compute_fold_metrics(y_te, yp))
            except Exception:
                pass
    return fold_metrics


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ESM-2 attention comparison and leakage gradient")
    parser.add_argument("--model", default="esm2_t6_8M_UR50D",
                        help="ESM-2 model name")
    parser.add_argument("--device", default="cpu",
                        help="Device for ESM-2 inference (cpu/cuda)")
    parser.add_argument("--target", default="Gq")
    parser.add_argument("--skip-attention", action="store_true",
                        help="Skip attention extraction (slow)")
    args = parser.parse_args()

    print("=" * 64)
    print("ESM-2 Attention Comparison & Leakage Gradient")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)

    dataset = GPCRDataset().load()
    bw_cache = load_bw_cache()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Part 1: Attention comparison
    if not args.skip_attention:
        result = run_attention_comparison(
            dataset, bw_cache, args.model, args.device, args.target)
        if result is not None:
            df_attn, attn_matrix = result
            attn_path = os.path.join(RESULTS_DIR, f"esm2_attention_{ts}.csv")
            df_attn.to_csv(attn_path, index=False)
            print(f"\n  Attention results saved: {attn_path}")

    # Part 2: Leakage gradient
    df_grad = run_esm2_leakage_gradient(
        dataset, bw_cache, args.model, args.device, args.target)
    if df_grad is not None:
        grad_path = os.path.join(RESULTS_DIR, f"esm2_leakage_gradient_{ts}.csv")
        df_grad.to_csv(grad_path, index=False)
        print(f"\n  Leakage gradient results saved: {grad_path}")

    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
