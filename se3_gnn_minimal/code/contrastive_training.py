#!/usr/bin/env python3
"""
Phase 3b: Contrastive learning training pipeline.

Combines:
  1. Multi-label BCE classification loss (Gs, Gi/o, Gq/11, G12/13)
  2. Supervised contrastive loss (NT-Xent) — pull same-coupling-type receptors
     together in embedding space, push different ones apart
  3. Graph augmentation: node masking + coordinate perturbation

Supports:
  - Standard training (random split)
  - Zero-shot evaluation (leave-one-subfamily-out)
"""
import os
import sys
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score
    HAS_SKLEARN_CLUSTER = True
except ImportError:
    HAS_SKLEARN_CLUSTER = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
from configs.config import (
    DATA_DIR, GRAPH_DIR, MODEL_DIR, RESULTS_DIR,
    CONTRASTIVE_CONFIG, TRAIN_CONFIG,
)

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ================================================================
# Loss functions
# ================================================================
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised NT-Xent loss.
    For each anchor, positives are samples that share at least one coupling label.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, labels):
        """
        projections: (B, proj_dim) L2-normalized
        labels: (B, 4) multi-label binary
        """
        B = projections.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=projections.device)

        # Similarity matrix
        sim = torch.mm(projections, projections.t()) / self.temperature  # (B, B)

        # Positive mask: share at least one coupling type
        label_sim = torch.mm(labels, labels.t())  # (B, B)
        pos_mask = (label_sim > 0).float()

        # Exclude self
        self_mask = 1.0 - torch.eye(B, device=projections.device)
        pos_mask = pos_mask * self_mask

        # If no positive pairs exist, return 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=projections.device)

        # Log-sum-exp over all negatives (denominator)
        exp_sim = torch.exp(sim) * self_mask
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log(pos/denom) over positive pairs
        log_prob = sim - log_denom
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined classification + contrastive loss."""

    def __init__(self, lambda_cls=1.0, lambda_contrast=0.5, temperature=0.07):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_contrast = lambda_contrast
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.contrast_loss = SupervisedContrastiveLoss(temperature)

    def forward(self, logits, projections, labels):
        """
        logits: (B, 4)
        projections: (B, proj_dim) L2-normalized
        labels: (B, 4)
        """
        l_cls = self.cls_loss(logits, labels)
        l_con = self.contrast_loss(projections, labels)
        total = self.lambda_cls * l_cls + self.lambda_contrast * l_con
        return total, l_cls, l_con


# ================================================================
# Graph augmentation for contrastive learning
# ================================================================
def augment_graph(data, mask_ratio=0.15, coord_noise_std=0.3):
    """
    Apply stochastic augmentations to a graph for contrastive learning.
    1. Node feature masking: randomly zero out mask_ratio of nodes
    2. Coordinate perturbation: add Gaussian noise to positions
    """
    data_aug = copy.copy(data)

    N = data.x.shape[0]

    # 1. Node masking
    mask = torch.rand(N) > mask_ratio
    data_aug.x = data.x.clone()
    data_aug.x[~mask] = 0.0

    # 2. Coordinate perturbation
    data_aug.pos = data.pos + torch.randn_like(data.pos) * coord_noise_std

    return data_aug


# ================================================================
# Dataset loading
# ================================================================
def load_graph_dataset(graph_dir=None):
    """Load all pre-built graphs."""
    if graph_dir is None:
        graph_dir = GRAPH_DIR

    graphs = []
    for fname in sorted(os.listdir(graph_dir)):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(graph_dir, fname)
        try:
            g = torch.load(path, map_location="cpu")
            graphs.append(g)
        except Exception as e:
            print(f"  WARNING: Could not load {fname}: {e}")

    print(f"Loaded {len(graphs)} graphs from {graph_dir}")
    return graphs


def split_by_subfamily(graphs, test_subfamily):
    """
    Leave-one-subfamily-out split for zero-shot evaluation.
    Returns (train_graphs, test_graphs).
    """
    train = [g for g in graphs if g.subfamily != test_subfamily]
    test = [g for g in graphs if g.subfamily == test_subfamily]
    return train, test


def split_random(graphs, test_ratio=0.2, seed=42):
    """Random split."""
    rng = random.Random(seed)
    indices = list(range(len(graphs)))
    rng.shuffle(indices)
    split_idx = int(len(graphs) * (1 - test_ratio))
    train = [graphs[i] for i in indices[:split_idx]]
    test = [graphs[i] for i in indices[split_idx:]]
    return train, test


# ================================================================
# Training loop
# ================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, augment=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cls = 0
    total_con = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        logits, proj, _ = model(batch.x, batch.pos, batch.edge_index, batch.batch)
        labels = batch.y.view(-1, 4)

        # Augmented view for contrastive learning
        if augment:
            batch_aug = augment_graph(batch,
                                       mask_ratio=CONTRASTIVE_CONFIG["mask_ratio"])
            batch_aug = batch_aug.to(device)
            _, proj_aug, _ = model(batch_aug.x, batch_aug.pos,
                                   batch_aug.edge_index, batch_aug.batch)
            # Concatenate projections from both views
            proj_combined = torch.cat([proj, proj_aug], dim=0)
            labels_combined = torch.cat([labels, labels], dim=0)
        else:
            proj_combined = proj
            labels_combined = labels

        # Classification loss on original only; contrastive on combined views
        l_cls = criterion.cls_loss(logits, labels)
        l_con = criterion.contrast_loss(proj_combined, labels_combined)
        loss = criterion.lambda_cls * l_cls + criterion.lambda_contrast * l_con

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls += l_cls.item()
        total_con += l_con.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_loss": total_cls / max(n_batches, 1),
        "con_loss": total_con / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    all_logits = []
    all_labels = []
    all_proj = []
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        logits, proj, _ = model(batch.x, batch.pos, batch.edge_index, batch.batch)
        labels = batch.y.view(-1, 4)

        loss, _, _ = criterion(logits, proj, labels)
        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_proj.append(proj.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_proj = torch.cat(all_proj, dim=0)

    # Compute metrics per G protein family
    preds = (torch.sigmoid(all_logits) > 0.5).float()
    gp_names = ["Gs", "Gi/o", "Gq/11", "G12/13"]
    metrics = {"loss": total_loss / max(n_batches, 1)}

    for i, name in enumerate(gp_names):
        tp = ((preds[:, i] == 1) & (all_labels[:, i] == 1)).sum().item()
        fp = ((preds[:, i] == 1) & (all_labels[:, i] == 0)).sum().item()
        fn = ((preds[:, i] == 0) & (all_labels[:, i] == 1)).sum().item()
        tn = ((preds[:, i] == 0) & (all_labels[:, i] == 0)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)

        metrics[f"{name}_f1"] = f1
        metrics[f"{name}_acc"] = acc

    # AUROC (if sklearn available)
    try:
        from sklearn.metrics import roc_auc_score
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()
        for i, name in enumerate(gp_names):
            if labels_np[:, i].sum() > 0 and labels_np[:, i].sum() < len(labels_np):
                metrics[f"{name}_auroc"] = roc_auc_score(labels_np[:, i], probs[:, i])
    except ImportError:
        pass

    # Projection-space clustering quality (NMI)
    if HAS_SKLEARN_CLUSTER and all_proj.shape[0] >= 2:
        labels_np = all_labels.numpy().astype(int)
        # Multi-label signature code in [0, 15]
        true_codes = (
            labels_np[:, 0] * 8
            + labels_np[:, 1] * 4
            + labels_np[:, 2] * 2
            + labels_np[:, 3]
        )

        # Keep only non-empty labels to avoid all-zero ambiguity
        valid = (labels_np.sum(axis=1) > 0)
        if valid.sum() >= 2:
            true_codes = true_codes[valid]
            proj_np = all_proj.numpy()[valid]
            n_clusters = min(len(np.unique(true_codes)), len(true_codes))
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                pred_clusters = kmeans.fit_predict(proj_np)
                metrics["proj_nmi"] = float(
                    normalized_mutual_info_score(true_codes, pred_clusters)
                )

    return metrics


# ================================================================
# Main training pipeline
# ================================================================
def train_model(
    train_graphs,
    val_graphs,
    config=None,
    save_name="se3_gpcr_gnn",
):
    """Full training pipeline."""
    if config is None:
        config = TRAIN_CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    from se3_gnn_model import build_model
    model = build_model().to(device)

    # DataLoaders
    if HAS_PYG:
        train_loader = PyGDataLoader(train_graphs, batch_size=config["batch_size"],
                                      shuffle=True, num_workers=0)
        val_loader = PyGDataLoader(val_graphs, batch_size=config["batch_size"],
                                    shuffle=False, num_workers=0)
    else:
        raise RuntimeError("torch_geometric required for batched graph loading")

    # Loss & optimizer
    criterion = CombinedLoss(
        lambda_cls=CONTRASTIVE_CONFIG["lambda_cls"],
        lambda_contrast=CONTRASTIVE_CONFIG["lambda_contrast"],
        temperature=CONTRASTIVE_CONFIG["temperature"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Learning rate scheduler
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config["epochs"], eta_min=1e-6
        )
    else:
        scheduler = None

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion,
                                         device, augment=True)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        # Log
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if epoch % 10 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d} | "
                  f"Train loss: {train_metrics['loss']:.4f} | "
                  f"Val loss: {val_metrics['loss']:.4f} | "
                  f"Val Gq F1: {val_metrics.get('Gq/11_f1', 0):.3f}")

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            # Save best model
            save_path = os.path.join(MODEL_DIR, f"{save_name}_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save training history
    history_file = os.path.join(RESULTS_DIR, f"{save_name}_history.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved: {os.path.join(MODEL_DIR, f'{save_name}_best.pt')}")
    print(f"History saved: {history_file}")

    return model, history


# ================================================================
# Zero-shot evaluation
# ================================================================
def zero_shot_evaluation(graphs):
    """
    Leave-one-subfamily-out zero-shot evaluation.
    Tests generalization to unseen GPCR subfamilies.
    """
    print("=" * 70)
    print("Zero-shot cross-subfamily evaluation")
    print("=" * 70)

    # Get all subfamilies
    subfamilies = sorted(set(g.subfamily for g in graphs if hasattr(g, 'subfamily')))
    print(f"Total subfamilies: {len(subfamilies)}")

    # Filter subfamilies with enough data
    sub_counts = {}
    for g in graphs:
        sf = g.subfamily if hasattr(g, 'subfamily') else "unknown"
        sub_counts[sf] = sub_counts.get(sf, 0) + 1

    valid_subs = [sf for sf, cnt in sub_counts.items() if cnt >= 3]
    print(f"Subfamilies with >=3 samples: {len(valid_subs)}")

    all_results = []

    for sf in valid_subs:
        train_g, test_g = split_by_subfamily(graphs, sf)
        if len(test_g) < 2 or len(train_g) < 10:
            continue

        print(f"\n  Testing on subfamily: {sf} ({len(test_g)} samples, "
              f"train: {len(train_g)})")

        # Quick training (fewer epochs for evaluation)
        eval_config = {**TRAIN_CONFIG, "epochs": 50, "patience": 15}
        model, history = train_model(
            train_g, test_g, config=eval_config,
            save_name=f"zeroshot_{sf}"
        )

        # Get final test metrics
        if history:
            best_epoch = min(history, key=lambda x: x["val"]["loss"])
            result = {
                "subfamily": sf,
                "n_test": len(test_g),
                "n_train": len(train_g),
                **best_epoch["val"],
            }
            all_results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("Zero-shot evaluation summary")
    print(f"{'=' * 70}")

    if all_results:
        results_file = os.path.join(RESULTS_DIR, "zeroshot_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved: {results_file}")

        # Average metrics
        for metric in ["Gq/11_f1", "Gs_f1", "Gi/o_f1"]:
            vals = [r.get(metric, 0) for r in all_results]
            if vals:
                print(f"  Mean {metric}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    return all_results


if __name__ == "__main__":
    print("This module provides training utilities.")
    print("Run 07_run_training.py for the full pipeline.")
