#!/usr/bin/env python3
"""
Phase 2: Build residue-level 3D spatial graphs from AlphaFold2 structures.

Each GPCR becomes a graph where:
  - Nodes = residues (features: ESM-2 embedding + BW position one-hot + pLDDT)
  - Edges = spatial contacts within CONTACT_RADIUS (Cα-Cα distance)
  - Edge features = distance, direction unit vector, sequential separation

Output:
  - data/graphs/{entry_name}.pt  (PyG Data objects)
  - data/graph_build_log.json
"""
import os
import sys
import json
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DATA_DIR, STRUCTURE_DIR, EMBEDDING_DIR, GRAPH_DIR,
    CONTACT_RADIUS, MAX_SEQ_LENGTH, BW_CONTACT_SITES
)

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("WARNING: torch_geometric not installed. Will save raw tensors instead.")


# ================================================================
# PDB parsing (lightweight, no BioPython dependency required)
# ================================================================
def parse_pdb_ca(pdb_path):
    """
    Parse Cα atoms from a PDB file.
    Returns:
      coords: np.array (N, 3) — Cα coordinates
      residues: list of (chain, resid, resname)
      plddt: np.array (N,) — B-factor (= pLDDT for AlphaFold)
    """
    coords = []
    residues = []
    plddt = []
    seen = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain = line[21]
                resid = int(line[22:26].strip())
                resname = line[17:20].strip()
                key = (chain, resid)

                if key not in seen:
                    seen.add(key)
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    bfactor = float(line[60:66])

                    coords.append([x, y, z])
                    residues.append((chain, resid, resname))
                    plddt.append(bfactor)

    return np.array(coords), residues, np.array(plddt)


def compute_edges(coords, radius=10.0):
    """
    Compute edges based on Cα-Cα distance within radius.
    Returns:
      edge_index: (2, E) tensor
      edge_attr: (E, 5) — [distance, dx, dy, dz, seq_sep]
    """
    N = len(coords)
    # Pairwise distance matrix
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))       # (N, N)

    # Find edges within radius (exclude self-loops)
    mask = (dist < radius) & (dist > 0.1)
    src, dst = np.where(mask)

    # Edge features
    distances = dist[src, dst]
    directions = diff[src, dst]  # (E, 3)
    # Normalize directions
    norms = np.maximum(distances[:, None], 1e-6)
    unit_dirs = directions / norms
    # Sequential separation (absolute)
    seq_sep = np.abs(src - dst).astype(np.float32)
    # Clip and log-transform seq_sep
    seq_sep = np.log1p(np.minimum(seq_sep, 100.0))

    edge_attr = np.concatenate([
        distances[:, None],
        unit_dirs,
        seq_sep[:, None],
    ], axis=1).astype(np.float32)

    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    return edge_index, edge_attr


def build_graph(entry_name, accession, dataset_record):
    """
    Build a single graph for a GPCR.
    Returns a dict with all graph tensors (or PyG Data object).
    """
    # Load structure
    pdb_path = os.path.join(STRUCTURE_DIR, f"{accession}.pdb")
    if not os.path.exists(pdb_path):
        return None, "no_structure"

    coords, residues, plddt = parse_pdb_ca(pdb_path)
    if len(coords) < 50:
        return None, f"too_few_residues ({len(coords)})"

    # Truncate if needed
    if len(coords) > MAX_SEQ_LENGTH:
        coords = coords[:MAX_SEQ_LENGTH]
        residues = residues[:MAX_SEQ_LENGTH]
        plddt = plddt[:MAX_SEQ_LENGTH]

    N = len(coords)

    # Load ESM-2 embeddings
    emb_path = os.path.join(EMBEDDING_DIR, f"{entry_name}.pt")
    if os.path.exists(emb_path):
        esm_emb = torch.load(emb_path, map_location="cpu")  # (seq_len, 1280)
        # Align lengths (structure vs sequence may differ slightly)
        if esm_emb.shape[0] >= N:
            esm_emb = esm_emb[:N]
        else:
            # Pad with zeros
            pad = torch.zeros(N - esm_emb.shape[0], esm_emb.shape[1])
            esm_emb = torch.cat([esm_emb, pad], dim=0)
    else:
        # No embedding available — use zeros (will be computed later)
        esm_emb = torch.zeros(N, 1280)

    # pLDDT as additional node feature (normalized to [0,1])
    plddt_feat = torch.tensor(plddt / 100.0, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    # Node features: ESM-2 embedding + pLDDT
    node_feat = torch.cat([esm_emb, plddt_feat], dim=1)  # (N, 1281)

    # Coordinates as tensor
    pos = torch.tensor(coords, dtype=torch.float32)  # (N, 3)

    # Compute edges
    edge_index, edge_attr = compute_edges(coords, radius=CONTACT_RADIUS)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    # Labels
    labels = torch.tensor([
        dataset_record["gs_label"],
        dataset_record["gio_label"],
        dataset_record["gq11_label"],
        dataset_record["g1213_label"],
    ], dtype=torch.float32)

    if HAS_PYG:
        graph = Data(
            x=node_feat,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            entry_name=entry_name,
            species=dataset_record["species"],
            subfamily=dataset_record.get("subfamily", ""),
            num_nodes=N,
        )
    else:
        graph = {
            "x": node_feat,
            "pos": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": labels,
            "entry_name": entry_name,
            "species": dataset_record["species"],
            "subfamily": dataset_record.get("subfamily", ""),
        }

    return graph, "ok"


def main():
    print("=" * 70)
    print("Phase 2: Build 3D residue-level graphs")
    print(f"Contact radius: {CONTACT_RADIUS} Å")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(GRAPH_DIR, exist_ok=True)

    # Load dataset
    json_file = os.path.join(DATA_DIR, "gpcr_multispecies_dataset.json")
    if not os.path.exists(json_file):
        print("ERROR: Dataset not found. Run 01_fetch_multispecies_gpcr.py first.")
        sys.exit(1)

    with open(json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Total receptors: {len(dataset)}")

    # Check existing graphs
    existing = set()
    for f_name in os.listdir(GRAPH_DIR):
        if f_name.endswith(".pt"):
            existing.add(f_name.replace(".pt", ""))

    remaining = [d for d in dataset if d["entry_name"] not in existing]
    print(f"Already built: {len(existing)}")
    print(f"Remaining: {len(remaining)}")

    # Build graphs
    log = {}
    success = 0
    fail = 0

    for i, rec in enumerate(remaining):
        entry_name = rec["entry_name"]
        accession = rec["accession"]

        graph, status = build_graph(entry_name, accession, rec)

        if graph is not None:
            out_path = os.path.join(GRAPH_DIR, f"{entry_name}.pt")
            torch.save(graph, out_path)
            success += 1
            log[entry_name] = {"status": "ok", "num_nodes": graph.num_nodes if HAS_PYG else graph["x"].shape[0]}
        else:
            fail += 1
            log[entry_name] = {"status": status}

        if (i + 1) % 100 == 0 or (i + 1) == len(remaining):
            print(f"  Progress: {i+1}/{len(remaining)} (ok={success}, fail={fail})")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Graph building complete")
    print(f"Success: {success}, Failed: {fail}")

    # Save log
    log_file = os.path.join(DATA_DIR, "graph_build_log.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"Saved log: {log_file}")

    # Statistics on successful graphs
    if success > 0:
        node_counts = [v["num_nodes"] for v in log.values() if v["status"] == "ok"]
        print(f"\nGraph statistics:")
        print(f"  Mean nodes: {np.mean(node_counts):.0f}")
        print(f"  Min/Max nodes: {min(node_counts)}/{max(node_counts)}")


if __name__ == "__main__":
    main()
