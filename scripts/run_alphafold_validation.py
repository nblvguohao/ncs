#!/usr/bin/env python3
"""
AlphaFold-Multimer in-silico validation of BW-site coupling determinants.

Strategy:
  1. Predict GPCR–Gα complex structures using ColabFold/AF-Multimer
  2. Extract predicted interface contacts (≤5 Å heavy-atom distance)
  3. Map contacts to BW generic positions
  4. Compare predicted contacts with our FDR-significant BW sites
  5. Compute precision/recall of BW-site framework vs AF-Multimer predictions

Requirements (server):
  pip install colabfold biopython pandas numpy matplotlib
  OR use pre-installed AlphaFold-Multimer with --model_preset=multimer

Usage:
  # Step 1: Generate FASTA inputs (no GPU needed)
  python scripts/run_alphafold_validation.py --step prepare

  # Step 2: Run AF-Multimer predictions (GPU required, ~2-8h per complex)
  python scripts/run_alphafold_validation.py --step predict --af2_binary /path/to/colabfold_batch

  # Step 3: Analyze predicted contacts vs BW sites (no GPU)
  python scripts/run_alphafold_validation.py --step analyze

  # All steps:
  python scripts/run_alphafold_validation.py --step all
"""
import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from leakageguard.data.dataset import GPCRDataset, G_PROTEIN_SEQS
from leakageguard.features.bw_site import (
    load_bw_cache, GP_CONTACT_SITES, get_bw_residue,
)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "alphafold_validation")
FASTA_DIR = os.path.join(RESULTS_DIR, "fasta_inputs")
AF_OUTPUT_DIR = os.path.join(RESULTS_DIR, "af_predictions")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FASTA_DIR, exist_ok=True)
os.makedirs(AF_OUTPUT_DIR, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────

# Representative GPCRs for AF-Multimer validation
# Selected to cover: Gq-coupled, Gi-coupled, Gs-coupled, orphan/ambiguous
VALIDATION_RECEPTORS = {
    # Gq-coupled (known)
    "opn4_human":  {"name": "OPN4 (Melanopsin)",  "coupling": "Gq",  "galpha": "gnaq"},
    "hrh1_human":  {"name": "H1R",                "coupling": "Gq",  "galpha": "gnaq"},
    "acm1_human":  {"name": "M1R",                "coupling": "Gq",  "galpha": "gnaq"},
    "5ht2a_human": {"name": "5-HT2A",             "coupling": "Gq",  "galpha": "gnaq"},
    # Gi-coupled
    "oprm_human":  {"name": "MOR (μ-opioid)",     "coupling": "Gi",  "galpha": "gnai"},
    "acm4_human":  {"name": "M4R",                "coupling": "Gi",  "galpha": "gnai"},
    "opsd_human":  {"name": "Rhodopsin",          "coupling": "Gi",  "galpha": "gnai"},
    # Gs-coupled
    "adrb2_human": {"name": "β2-AR",              "coupling": "Gs",  "galpha": "gnas"},
    "glr_human":   {"name": "GLP-1R",             "coupling": "Gs",  "galpha": "gnas"},
    # Dual/ambiguous
    "drd1_human":  {"name": "D1R",                "coupling": "Gs/Gq", "galpha": "gnaq"},
}

# FDR-significant BW sites from our analysis
FDR_SIGNIFICANT_SITES = ["34.53", "5.71", "3.53", "5.65", "34.50"]

# Interface contact distance threshold (Angstroms)
CONTACT_THRESHOLD = 5.0


# ── Step 1: Prepare FASTA inputs ─────────────────────────────────────────

def prepare_fasta_inputs():
    """Generate paired FASTA files for AF-Multimer (GPCR + Gα)."""
    print("=" * 60)
    print("Step 1: Preparing AF-Multimer FASTA inputs")
    print("=" * 60)

    dataset = GPCRDataset().load()

    # Build entry_name -> sequence map
    seq_map = {}
    for i, entry in enumerate(dataset.entry_names):
        seq_map[entry] = dataset.sequences[i]

    generated = []
    for entry_name, info in VALIDATION_RECEPTORS.items():
        if entry_name not in seq_map:
            print(f"  ⚠ {entry_name} not found in dataset, skipping")
            continue

        receptor_seq = seq_map[entry_name]
        galpha_seq = G_PROTEIN_SEQS[info["galpha"]]

        # ColabFold format: sequences separated by ':'
        fasta_path = os.path.join(FASTA_DIR, f"{entry_name}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{entry_name}__{info['galpha']}\n")
            f.write(f"{receptor_seq}:{galpha_seq}\n")

        # Also write separate chains for standard AF-Multimer
        fasta_sep = os.path.join(FASTA_DIR, f"{entry_name}_separate.fasta")
        with open(fasta_sep, "w") as f:
            f.write(f">{entry_name}\n{receptor_seq}\n")
            f.write(f">{info['galpha']}\n{galpha_seq}\n")

        generated.append(entry_name)
        print(f"  ✓ {entry_name} ({info['name']}) → {fasta_path}")

    # Write batch script for ColabFold
    batch_script = os.path.join(RESULTS_DIR, "run_colabfold.sh")
    with open(batch_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Run ColabFold batch predictions for GPCR-Gα complexes\n")
        f.write("# Requires: pip install colabfold[alphafold]\n")
        f.write(f"# Expected runtime: ~2-8 hours per complex on A100\n\n")
        f.write(f"INPUT_DIR={FASTA_DIR}\n")
        f.write(f"OUTPUT_DIR={AF_OUTPUT_DIR}\n\n")
        f.write("colabfold_batch \\\n")
        f.write("  --model-type alphafold2_multimer_v3 \\\n")
        f.write("  --num-recycle 3 \\\n")
        f.write("  --num-models 5 \\\n")
        f.write("  --amber \\\n")
        f.write("  --use-gpu-relax \\\n")
        f.write("  $INPUT_DIR $OUTPUT_DIR\n")

    print(f"\n  Generated {len(generated)} FASTA files")
    print(f"  Batch script: {batch_script}")
    print(f"\n  To run on server:")
    print(f"    bash {batch_script}")
    return generated


# ── Step 2: Run AF-Multimer (calls external binary) ─────────────────────

def run_af_predictions(af2_binary="colabfold_batch"):
    """Run AF-Multimer via ColabFold or local installation."""
    import subprocess

    print("=" * 60)
    print("Step 2: Running AlphaFold-Multimer predictions")
    print("=" * 60)

    cmd = [
        af2_binary,
        "--model-type", "alphafold2_multimer_v3",
        "--num-recycle", "3",
        "--num-models", "5",
        "--amber",
        FASTA_DIR,
        AF_OUTPUT_DIR,
    ]
    print(f"  Command: {' '.join(cmd)}")
    print(f"  This may take several hours per complex on GPU...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
        if result.returncode == 0:
            print("  ✓ AF-Multimer predictions completed")
        else:
            print(f"  ✗ Error: {result.stderr[:500]}")
            print("  You may need to run the predictions manually.")
    except FileNotFoundError:
        print(f"  ✗ {af2_binary} not found. Run predictions manually:")
        print(f"    bash {os.path.join(RESULTS_DIR, 'run_colabfold.sh')}")
    except subprocess.TimeoutExpired:
        print("  ✗ Timed out (>24h). Check GPU availability.")


# ── Step 3: Analyze predicted contacts ───────────────────────────────────

def _parse_pdb_contacts(pdb_path, chain_receptor="A", chain_galpha="B",
                        threshold=CONTACT_THRESHOLD):
    """Extract inter-chain contacts from a PDB file.

    Returns list of (receptor_resid, galpha_resid, distance).
    """
    try:
        from Bio.PDB import PDBParser, NeighborSearch
    except ImportError:
        print("  ✗ BioPython required: pip install biopython")
        return []

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    model = structure[0]

    # Collect atoms by chain
    receptor_atoms = []
    galpha_atoms = []
    for chain in model:
        cid = chain.id
        for residue in chain:
            if residue.id[0] != " ":  # skip heteroatoms
                continue
            for atom in residue:
                if cid == chain_receptor:
                    receptor_atoms.append(atom)
                elif cid == chain_galpha:
                    galpha_atoms.append(atom)

    # Find contacts using NeighborSearch
    try:
        # Check if atoms have valid coordinates
        valid_receptor_atoms = [a for a in receptor_atoms if not np.isnan(a.get_coord()).any()]
        valid_galpha_atoms = [a for a in galpha_atoms if not np.isnan(a.get_coord()).any()]

        if not valid_galpha_atoms or not valid_receptor_atoms:
            return []

        ns = NeighborSearch(valid_galpha_atoms)
        contacts = []
        seen = set()
        for atom in valid_receptor_atoms:
            nearby = ns.search(atom.get_coord(), threshold)
            for ga_atom in nearby:
                r_res = atom.get_parent()
                g_res = ga_atom.get_parent()
                key = (r_res.id[1], g_res.id[1])
                if key not in seen:
                    dist = atom - ga_atom
                    contacts.append({
                        "receptor_resid": r_res.id[1],
                        "receptor_resname": r_res.resname,
                        "galpha_resid": g_res.id[1],
                        "distance": dist,
                    })
                    seen.add(key)
    except Exception as e:
        print(f"  ⚠ Warning: Contact calculation failed ({e})")
        return []

    return contacts


def _map_resid_to_bw(entry_name, resid, bw_cache):
    """Map a PDB residue ID to BW generic number using GPCRdb cache."""
    if entry_name not in bw_cache:
        return None
    residues = bw_cache[entry_name]
    for res in residues:
        seq_num = res.get("sequence_number")
        if seq_num == resid:
            gn = res.get("display_generic_number") or res.get("generic_number", "")
            if isinstance(gn, dict):
                gn = gn.get("label", "")
            if gn:
                # Normalize: "3.53x53" -> "3.53"
                return gn.split("x")[0]
    return None


def analyze_predictions():
    """Analyze AF-Multimer predictions: extract contacts, compare with BW sites."""
    print("=" * 60)
    print("Step 3: Analyzing AF-Multimer predicted contacts")
    print("=" * 60)

    bw_cache = load_bw_cache()

    # Find predicted PDB files
    results = []
    for entry_name, info in VALIDATION_RECEPTORS.items():
        # ColabFold output naming convention
        # Note: Filenames might include galpha suffix (e.g. opn4_human__gnaq...)
        pdb_candidates = [
            os.path.join(AF_OUTPUT_DIR, f"{entry_name}_relaxed_rank_001*.pdb"),
            os.path.join(AF_OUTPUT_DIR, f"{entry_name}*relaxed_rank_001*.pdb"), # Added wildcard for suffix
            os.path.join(AF_OUTPUT_DIR, f"{entry_name}_unrelaxed_rank_001*.pdb"),
            os.path.join(AF_OUTPUT_DIR, f"{entry_name}*unrelaxed_rank_001*.pdb"),
            os.path.join(AF_OUTPUT_DIR, entry_name, "ranked_0.pdb"),
            os.path.join(AF_OUTPUT_DIR, f"{entry_name}.pdb"),
        ]

        import glob
        pdb_path = None
        for pattern in pdb_candidates:
            matches = glob.glob(pattern)
            if matches:
                pdb_path = matches[0]
                break

        if pdb_path is None:
            print(f"  ⚠ No PDB found for {entry_name}, skipping")
            continue

        print(f"\n  Analyzing {entry_name} ({info['name']})...")
        contacts = _parse_pdb_contacts(pdb_path)
        print(f"    Raw contacts (≤{CONTACT_THRESHOLD}Å): {len(contacts)}")

        # Map to BW positions
        bw_contacts = set()
        for c in contacts:
            bw = _map_resid_to_bw(entry_name, c["receptor_resid"], bw_cache)
            if bw and bw in GP_CONTACT_SITES:
                bw_contacts.add(bw)

        # Compare with our FDR-significant sites
        fdr_hit = bw_contacts.intersection(FDR_SIGNIFICANT_SITES)
        all_29_hit = bw_contacts.intersection(GP_CONTACT_SITES)

        result = {
            "entry_name": entry_name,
            "receptor_name": info["name"],
            "coupling": info["coupling"],
            "n_raw_contacts": len(contacts),
            "n_bw_contacts": len(bw_contacts),
            "bw_contacts": sorted(bw_contacts),
            "fdr_hits": sorted(fdr_hit),
            "fdr_precision": len(fdr_hit) / len(FDR_SIGNIFICANT_SITES) if FDR_SIGNIFICANT_SITES else 0,
            "bw29_coverage": len(all_29_hit) / 29,
        }
        results.append(result)
        print(f"    BW contacts: {len(bw_contacts)}/29 ({result['bw29_coverage']:.1%})")
        print(f"    FDR-significant hits: {len(fdr_hit)}/5 ({result['fdr_precision']:.1%})")
        print(f"    FDR sites contacted: {sorted(fdr_hit)}")

    if not results:
        print("\n  No predictions found. Run Step 2 first.")
        return

    # Summary statistics
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "af_validation_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("SUMMARY: AlphaFold-Multimer Validation")
    print("=" * 60)

    # Per-coupling-type analysis
    for coupling in ["Gq", "Gi", "Gs"]:
        sub = df[df["coupling"].str.contains(coupling)]
        if len(sub) == 0:
            continue
        mean_fdr = sub["fdr_precision"].mean()
        mean_cov = sub["bw29_coverage"].mean()
        print(f"\n  {coupling}-coupled ({len(sub)} receptors):")
        print(f"    Mean FDR-site recovery: {mean_fdr:.1%}")
        print(f"    Mean BW-29 coverage:    {mean_cov:.1%}")

    # Overall FDR validation
    all_fdr_hits = set()
    for r in results:
        all_fdr_hits.update(r["fdr_hits"])
    print(f"\n  Overall FDR sites recovered (union): {len(all_fdr_hits)}/5")
    print(f"    Sites: {sorted(all_fdr_hits)}")

    # Coupling-specific contacts (Gq vs non-Gq)
    print("\n  Coupling-specific contact analysis:")
    gq_bw = set()
    nongq_bw = set()
    for r in results:
        if "Gq" in r["coupling"]:
            gq_bw.update(r["bw_contacts"])
        else:
            nongq_bw.update(r["bw_contacts"])
    gq_specific = gq_bw - nongq_bw
    nongq_specific = nongq_bw - gq_bw
    shared = gq_bw & nongq_bw
    print(f"    Gq-specific contacts:     {sorted(gq_specific)}")
    print(f"    Non-Gq-specific contacts: {sorted(nongq_specific)}")
    print(f"    Shared contacts:          {sorted(shared)}")

    # Generate comparison figure
    _plot_contact_comparison(results)

    print(f"\n  Results saved to: {RESULTS_DIR}")


def _plot_contact_comparison(results):
    """Generate heatmap comparing AF-Multimer contacts with BW-site predictions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not available, skipping figure")
        return

    receptors = [r["entry_name"] for r in results]
    n_rec = len(receptors)

    # Build contact matrix: receptors × 29 BW sites
    matrix = np.zeros((n_rec, len(GP_CONTACT_SITES)))
    for i, r in enumerate(results):
        for bw in r["bw_contacts"]:
            if bw in GP_CONTACT_SITES:
                j = GP_CONTACT_SITES.index(bw)
                matrix[i, j] = 1

    # Mark FDR-significant columns
    fdr_cols = [GP_CONTACT_SITES.index(s) for s in FDR_SIGNIFICANT_SITES
                if s in GP_CONTACT_SITES]

    fig, ax = plt.subplots(figsize=(14, max(4, n_rec * 0.6)))
    cmap = plt.cm.colors.ListedColormap(["#f0f0f0", "#2A9D8F"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")

    # Highlight FDR columns
    for col in fdr_cols:
        ax.axvline(col - 0.5, color="#E07A5F", linewidth=2, alpha=0.7)
        ax.axvline(col + 0.5, color="#E07A5F", linewidth=2, alpha=0.7)

    ax.set_xticks(range(len(GP_CONTACT_SITES)))
    ax.set_xticklabels(GP_CONTACT_SITES, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rec))
    labels = [f"{r['receptor_name']} ({r['coupling']})" for r in results]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("BW Contact Position")
    ax.set_title("AlphaFold-Multimer Predicted Contacts at BW Positions\n"
                 "(Orange borders = FDR-significant sites from sequence analysis)")

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "fig_af_validation_heatmap.png")
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_path.replace(".png", ".pdf"))
    plt.close()
    print(f"  ✓ Figure: {fig_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AF-Multimer BW-site validation")
    parser.add_argument("--step", choices=["prepare", "predict", "analyze", "all"],
                        default="prepare",
                        help="Which step to run")
    parser.add_argument("--af2_binary", default="colabfold_batch",
                        help="Path to ColabFold or AF-Multimer binary")
    args = parser.parse_args()

    if args.step in ("prepare", "all"):
        prepare_fasta_inputs()

    if args.step in ("predict", "all"):
        run_af_predictions(args.af2_binary)

    if args.step in ("analyze", "all"):
        analyze_predictions()


if __name__ == "__main__":
    main()
