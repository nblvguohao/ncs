#!/usr/bin/env python3
"""
Analyze training outputs in results/ and generate robust summaries.

Usage:
  python code/analyze_results.py
  python code/analyze_results.py --results_dir results --topk 20
  python code/analyze_results.py --prediction_file results/val_predictions.json

Optional prediction file is used for threshold scan. Supported formats:
  1) [{"y_true": [0,1,0,0], "y_prob": [0.1,0.9,0.2,0.05]}, ...]
  2) [{"Gs_true":0, "Gi/o_true":1, ... , "Gs_prob":0.1, "Gi/o_prob":0.9, ...}, ...]
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = ["Gs", "Gi/o", "Gq/11", "G12/13"]


def _safe_float(value, default=0.0):
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_standard(results_dir: Path):
    test_path = results_dir / "standard_test_results.json"
    hist_path = results_dir / "se3_gnn_standard_history.json"

    test_metrics = _load_json(test_path)
    history = _load_json(hist_path)
    if not history:
        raise ValueError("standard history is empty")

    best = min(history, key=lambda x: _safe_float(x.get("val", {}).get("loss"), 1e9))
    last = history[-1]

    summary = {
        "epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": _safe_float(best["val"].get("loss")),
        "last_epoch": int(last["epoch"]),
        "last_val_loss": _safe_float(last["val"].get("loss")),
        "test_metrics": test_metrics,
        "best_val_metrics": {
            lb: {
                "f1": _safe_float(best["val"].get(f"{lb}_f1")),
                "acc": _safe_float(best["val"].get(f"{lb}_acc")),
                "auroc": _safe_float(best["val"].get(f"{lb}_auroc")),
            }
            for lb in LABELS
        },
    }
    return summary


def _bootstrap_ci(values, n_boot=2000, seed=42):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats.append(float(sample.mean()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(arr.mean()), float(lo), float(hi))


def summarize_zeroshot(results_dir: Path, topk: int):
    z_path = results_dir / "zeroshot_results.json"
    rows = _load_json(z_path)
    if not rows:
        raise ValueError("zero-shot result file is empty")

    df = pd.DataFrame(rows)
    df["n_test"] = df["n_test"].astype(int)
    for lb in LABELS:
        df[f"{lb}_f1"] = df[f"{lb}_f1"].astype(float)
        df[f"{lb}_acc"] = df[f"{lb}_acc"].astype(float)

    by_label = {}
    total_weight = float(df["n_test"].sum())
    for lb in LABELS:
        f1_col = f"{lb}_f1"
        acc_col = f"{lb}_acc"
        macro_f1 = float(df[f1_col].mean())
        weighted_f1 = float((df[f1_col] * df["n_test"]).sum() / total_weight)
        macro_acc = float(df[acc_col].mean())
        weighted_acc = float((df[acc_col] * df["n_test"]).sum() / total_weight)
        ci_mean, ci_low, ci_high = _bootstrap_ci(df[f1_col].values)
        by_label[lb] = {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_acc": macro_acc,
            "weighted_acc": weighted_acc,
            "f1_nonzero_ratio": float((df[f1_col] > 0).mean()),
            "macro_f1_bootstrap_95ci": [ci_mean, ci_low, ci_high],
        }

    df["min_acc"] = df[[f"{lb}_acc" for lb in LABELS]].min(axis=1)
    failure_cols = [
        "subfamily",
        "n_test",
        "loss",
        "min_acc",
        "Gs_acc",
        "Gi/o_acc",
        "Gq/11_acc",
        "G12/13_acc",
        "Gs_f1",
        "Gi/o_f1",
        "Gq/11_f1",
        "G12/13_f1",
    ]

    worst_loss = (
        df.sort_values("loss", ascending=False)
        .head(topk)[failure_cols]
        .to_dict(orient="records")
    )
    worst_acc = (
        df.sort_values(["min_acc", "loss"], ascending=[True, False])
        .head(topk)[failure_cols]
        .to_dict(orient="records")
    )

    summary = {
        "num_subfamilies": int(len(df)),
        "total_test_samples": int(df["n_test"].sum()),
        "n_test_distribution": {
            str(k): int(v)
            for k, v in df["n_test"].value_counts().sort_index().to_dict().items()
        },
        "loss": {
            "mean": float(df["loss"].mean()),
            "median": float(df["loss"].median()),
            "max": float(df["loss"].max()),
        },
        "by_label": by_label,
        "top_failures_by_loss": worst_loss,
        "top_failures_by_min_acc": worst_acc,
    }

    return summary, df


def threshold_scan(prediction_file: Path):
    rows = _load_json(prediction_file)
    if not rows:
        raise ValueError("prediction file is empty")

    y_true = []
    y_prob = []
    for r in rows:
        if "y_true" in r and "y_prob" in r:
            t, p = r["y_true"], r["y_prob"]
        else:
            t = [r.get(f"{lb}_true") for lb in LABELS]
            p = [r.get(f"{lb}_prob") for lb in LABELS]

        if len(t) != 4 or len(p) != 4:
            continue
        y_true.append([int(x) for x in t])
        y_prob.append([float(x) for x in p])

    if not y_true:
        raise ValueError("No valid rows found in prediction file")

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.05, 0.95, 19)
    output = {}
    for i, lb in enumerate(LABELS):
        best_thr = 0.5
        best_f1 = -1.0
        for thr in thresholds:
            pred = (y_prob[:, i] >= thr).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_thr = float(thr)
        output[lb] = {
            "best_threshold": best_thr,
            "best_f1": best_f1,
            "default_0.5_f1": float(
                f1_score(y_true[:, i], (y_prob[:, i] >= 0.5).astype(int), zero_division=0)
            ),
        }

    return {
        "n_samples": int(y_true.shape[0]),
        "threshold_scan": output,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze GPCR training/evaluation results")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--prediction_file", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = (project_root / args.results_dir).resolve()
    os.makedirs(results_dir, exist_ok=True)

    standard_summary = summarize_standard(results_dir)
    zeroshot_summary, zs_df = summarize_zeroshot(results_dir, topk=max(1, args.topk))

    report = {
        "standard": standard_summary,
        "zeroshot": zeroshot_summary,
    }

    if args.prediction_file:
        pred_file = Path(args.prediction_file)
        if not pred_file.is_absolute():
            pred_file = (project_root / pred_file).resolve()
        report["threshold_scan"] = threshold_scan(pred_file)

    out_json = results_dir / "analysis_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save tabular zeroshot snapshot for quick filtering in spreadsheet tools
    zs_df.to_csv(results_dir / "zeroshot_results_enriched.csv", index=False)

    print("=" * 70)
    print("Analysis complete")
    print(f"Saved summary: {out_json}")
    print(f"Saved table:   {results_dir / 'zeroshot_results_enriched.csv'}")
    print("=" * 70)

    print("\n[Standard]")
    print(
        f"epochs={standard_summary['epochs']} "
        f"best_epoch={standard_summary['best_epoch']} "
        f"best_val_loss={standard_summary['best_val_loss']:.4f}"
    )
    print("test metrics (F1): " + ", ".join(
        f"{lb}={_safe_float(standard_summary['test_metrics'].get(f'{lb}_f1')):.3f}"
        for lb in LABELS
    ))

    print("\n[Zero-shot]")
    print(
        f"subfamilies={zeroshot_summary['num_subfamilies']} "
        f"total_test_samples={zeroshot_summary['total_test_samples']} "
        f"loss_mean={zeroshot_summary['loss']['mean']:.4f}"
    )
    for lb in LABELS:
        s = zeroshot_summary["by_label"][lb]
        print(
            f"{lb}: macro_f1={s['macro_f1']:.3f} "
            f"weighted_f1={s['weighted_f1']:.3f} "
            f"nonzero_ratio={s['f1_nonzero_ratio']:.3f}"
        )

    if "threshold_scan" in report:
        print("\n[Threshold scan]")
        print(f"samples={report['threshold_scan']['n_samples']}")
        for lb, s in report["threshold_scan"]["threshold_scan"].items():
            print(
                f"{lb}: best_thr={s['best_threshold']:.2f} "
                f"best_f1={s['best_f1']:.3f} "
                f"f1@0.5={s['default_0.5_f1']:.3f}"
            )


if __name__ == "__main__":
    main()
