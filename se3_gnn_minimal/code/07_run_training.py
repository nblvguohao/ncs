#!/usr/bin/env python3
"""
Phase 3c: Main training script — orchestrates the full pipeline.

Usage:
  python code/07_run_training.py --mode standard     # Standard train/val/test
  python code/07_run_training.py --mode zeroshot      # Leave-one-subfamily-out
  python code/07_run_training.py --mode both          # Run both
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
from configs.config import TRAIN_CONFIG, RESULTS_DIR, GRAPH_DIR, MODEL_DIR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="GPCR SE(3)-GNN Training")
    parser.add_argument("--mode", choices=["standard", "zeroshot", "both"],
                        default="standard")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("GPCR SE(3)-GNN Coupling Prediction — Training Pipeline")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)

    # Override config if args provided
    config = {**TRAIN_CONFIG}
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["lr"] = args.lr

    # Load graphs
    from contrastive_training import (
        load_graph_dataset, split_random, train_model, zero_shot_evaluation
    )
    graphs = load_graph_dataset(GRAPH_DIR)

    if len(graphs) == 0:
        print("ERROR: No graphs found. Run Phase 1-2 scripts first:")
        print("  python code/01_fetch_multispecies_gpcr.py")
        print("  python code/02_download_af2_structures.py")
        print("  python code/03_compute_esm2_embeddings.py")
        print("  python code/04_build_3d_graphs.py")
        sys.exit(1)

    # Print dataset stats
    print(f"\nDataset: {len(graphs)} graphs")
    species = {}
    for g in graphs:
        sp = g.species if hasattr(g, 'species') else "unknown"
        species[sp] = species.get(sp, 0) + 1
    for sp, cnt in sorted(species.items(), key=lambda x: -x[1]):
        print(f"  {sp}: {cnt}")

    labels = torch.stack([g.y for g in graphs])
    gp_names = ["Gs", "Gi/o", "Gq/11", "G12/13"]
    print(f"\nLabel distribution:")
    for i, name in enumerate(gp_names):
        pos = (labels[:, i] > 0).sum().item()
        print(f"  {name}: {pos} positive / {len(graphs) - pos} negative")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ============================================================
    # Standard training
    # ============================================================
    if args.mode in ("standard", "both"):
        print(f"\n{'=' * 70}")
        print("Standard training (80/20 random split)")
        print(f"{'=' * 70}")

        train_g, test_g = split_random(graphs, test_ratio=0.2, seed=args.seed)
        # Further split train into train/val
        val_ratio = 0.15
        val_size = int(len(train_g) * val_ratio)
        val_g = train_g[:val_size]
        train_g = train_g[val_size:]

        print(f"Train: {len(train_g)}, Val: {len(val_g)}, Test: {len(test_g)}")

        model, history = train_model(
            train_g, val_g, config=config, save_name="se3_gnn_standard"
        )

        # Final test evaluation
        from contrastive_training import evaluate, CombinedLoss
        from configs.config import CONTRASTIVE_CONFIG

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        try:
            from torch_geometric.loader import DataLoader as PyGDataLoader
            test_loader = PyGDataLoader(test_g, batch_size=config["batch_size"],
                                         shuffle=False)
        except ImportError:
            print("ERROR: torch_geometric required")
            sys.exit(1)

        criterion = CombinedLoss(
            lambda_cls=CONTRASTIVE_CONFIG["lambda_cls"],
            lambda_contrast=CONTRASTIVE_CONFIG["lambda_contrast"],
            temperature=CONTRASTIVE_CONFIG["temperature"],
        )

        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"\n{'=' * 70}")
        print("Test Results (Standard)")
        print(f"{'=' * 70}")
        for k, v in sorted(test_metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        # Save test results
        test_file = os.path.join(RESULTS_DIR, "standard_test_results.json")
        with open(test_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Saved: {test_file}")

    # ============================================================
    # Zero-shot evaluation
    # ============================================================
    if args.mode in ("zeroshot", "both"):
        print(f"\n{'=' * 70}")
        print("Zero-shot cross-subfamily evaluation")
        print(f"{'=' * 70}")

        results = zero_shot_evaluation(graphs)

        # Summary table
        if results:
            print(f"\n{'Subfamily':<30} {'N_test':<8} {'Gq F1':<8} {'Gs F1':<8} {'Gi F1':<8}")
            print("-" * 62)
            for r in sorted(results, key=lambda x: x.get("Gq/11_f1", 0), reverse=True):
                print(f"{r['subfamily']:<30} {r['n_test']:<8} "
                      f"{r.get('Gq/11_f1', 0):<8.3f} "
                      f"{r.get('Gs_f1', 0):<8.3f} "
                      f"{r.get('Gi/o_f1', 0):<8.3f}")

    print(f"\n{'=' * 70}")
    print("Pipeline complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
