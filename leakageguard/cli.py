#!/usr/bin/env python3
"""
LeakageGuard command-line interface.

Commands
--------
leakageguard diagnose   – Run leakage diagnostic (random vs no-leak)
leakageguard benchmark  – Full repeated grouped k-fold CV benchmark
leakageguard multilabel – Multi-target G protein coupling evaluation
leakageguard info       – Print dataset summary
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def _project_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── diagnose ──────────────────────────────────────────────────────────────

def cmd_diagnose(args):
    """Run leakage diagnostic comparing random vs subfamily-grouped CV."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    from .data.dataset import GPCRDataset
    from .features.handcrafted import build_feature_matrix
    from .features.bw_site import load_bw_cache, build_bw_feature_matrix
    from .splits.strategies import grouped_kfold_cv
    from .models.classifiers import build_models
    from .evaluation.metrics import compute_fold_metrics, aggregate_cv_results

    dataset = GPCRDataset(data_dir=args.data_dir).load()
    bw_cache = load_bw_cache(
        os.path.join(args.data_dir, "gpcrdb_residues_cache.json")
        if args.data_dir else None
    )

    X_hc, _ = build_feature_matrix(dataset.sequences, dataset.families)
    X_bw, _ = build_bw_feature_matrix(dataset.entry_names, bw_cache)

    y = dataset.get_labels(args.target)
    models = {"Ensemble": build_models()["Ensemble"]}

    print(f"\n{'='*64}")
    print(f"  LeakageGuard Diagnostic Report")
    print(f"  Target: {args.target}  |  Samples: {len(y)} "
          f"(pos={int(y.sum())}, neg={len(y)-int(y.sum())})")
    print(f"  CV: {args.n_folds}-fold")
    print(f"{'='*64}")

    for feat_name, X in [("handcrafted", X_hc), ("BW-site", X_bw)]:
        print(f"\n  Feature set: {feat_name} ({X.shape[1]}d)")
        for strategy, label in [("random", "Random (leaky)"),
                                 ("subfamily", "Subfamily (no-leak)")]:
            if strategy == "random":
                skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                      random_state=42)
                folds = list(skf.split(X, y))
            else:
                folds = grouped_kfold_cv(y, dataset.families,
                                          n_folds=args.n_folds, seed=42)

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

            agg = aggregate_cv_results(fold_metrics)
            auc_m = agg["auc"]["mean"]
            auc_s = agg["auc"]["std"]
            print(f"    {label:25s}  AUC = {auc_m:.3f} +/- {auc_s:.3f}")

    print(f"\n{'='*64}")
    print("  Interpretation:")
    print("    ΔAUC > 0.05  →  significant phylogenetic leakage detected")
    print("    ΔAUC ≤ 0.05  →  no significant leakage")
    print(f"{'='*64}\n")


# ── benchmark ─────────────────────────────────────────────────────────────

def cmd_benchmark(args):
    """Delegate to scripts/run_benchmark.py logic."""
    sys.path.insert(0, _project_dir())
    script = os.path.join(_project_dir(), "scripts", "run_benchmark.py")
    sys.argv = [
        script,
        "--target", args.target,
        "--n-folds", str(args.n_folds),
        "--n-repeats", str(args.n_repeats),
    ]
    if args.skip_seqcluster:
        sys.argv.append("--skip-seqcluster")
    exec(open(script, encoding="utf-8").read(), {"__name__": "__main__"})


# ── multilabel ────────────────────────────────────────────────────────────

def cmd_multilabel(args):
    """Delegate to scripts/run_multilabel.py logic."""
    sys.path.insert(0, _project_dir())
    script = os.path.join(_project_dir(), "scripts", "run_multilabel.py")
    sys.argv = [
        script,
        "--n-folds", str(args.n_folds),
        "--n-repeats", str(args.n_repeats),
    ]
    exec(open(script, encoding="utf-8").read(), {"__name__": "__main__"})


# ── info ──────────────────────────────────────────────────────────────────

def cmd_info(args):
    """Print dataset summary."""
    from .data.dataset import GPCRDataset
    dataset = GPCRDataset(data_dir=args.data_dir).load()
    dataset.summary()


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="leakageguard",
        description="LeakageGuard — leakage-aware GPCR coupling benchmarking",
    )
    parser.add_argument("--version", action="version",
                        version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # diagnose
    p_diag = sub.add_parser("diagnose", help="Run leakage diagnostic")
    p_diag.add_argument("--target", default="Gq")
    p_diag.add_argument("--n-folds", type=int, default=5)
    p_diag.add_argument("--data-dir", default=None)
    p_diag.set_defaults(func=cmd_diagnose)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Full repeated CV benchmark")
    p_bench.add_argument("--target", default="Gq")
    p_bench.add_argument("--n-folds", type=int, default=5)
    p_bench.add_argument("--n-repeats", type=int, default=10)
    p_bench.add_argument("--skip-seqcluster", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    # multilabel
    p_ml = sub.add_parser("multilabel", help="Multi-target evaluation")
    p_ml.add_argument("--n-folds", type=int, default=5)
    p_ml.add_argument("--n-repeats", type=int, default=5)
    p_ml.set_defaults(func=cmd_multilabel)

    # info
    p_info = sub.add_parser("info", help="Print dataset summary")
    p_info.add_argument("--data-dir", default=None)
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
