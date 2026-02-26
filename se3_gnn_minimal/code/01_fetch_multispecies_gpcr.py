#!/usr/bin/env python3
"""
Phase 1a: Fetch all non-olfactory GPCRs from GPCRdb across multiple species.
Also retrieves coupling annotations from GPCRdb's coupling endpoint.

Output:
  - data/gpcr_multispecies_dataset.csv
  - data/gpcr_multispecies_dataset.json
  - data/coupling_annotations.json (raw GPCRdb coupling data)
"""
import os
import sys
import json
import time
import csv
import requests
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    GPCRDB_BASE_URL, TARGET_SPECIES, TARGET_FAMILIES,
    DATA_DIR, G_PROTEIN_FAMILIES
)

# ================================================================
# GPCRdb API helpers
# ================================================================
def fetch_json(url, retries=3, delay=1.0):
    """API request with retries."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            else:
                print(f"  HTTP {resp.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"  Request failed (attempt {attempt+1}/{retries}): {e}")
        time.sleep(delay * (attempt + 1))
    return None


def get_leaf_slugs(parent_slug):
    """Recursively get all leaf-node subfamily slugs."""
    url = f"{GPCRDB_BASE_URL}/proteinfamily/children/{parent_slug}/"
    children = fetch_json(url)
    if not children:
        return [parent_slug]

    leaf_slugs = []
    for child in children:
        slug = child["slug"]
        sub_children = fetch_json(f"{GPCRDB_BASE_URL}/proteinfamily/children/{slug}/")
        if sub_children:
            for sc in sub_children:
                # Go one more level
                sc_children = fetch_json(f"{GPCRDB_BASE_URL}/proteinfamily/children/{sc['slug']}/")
                if sc_children:
                    leaf_slugs.extend([scc["slug"] for scc in sc_children])
                else:
                    leaf_slugs.append(sc["slug"])
        else:
            leaf_slugs.append(slug)
    return leaf_slugs


# ================================================================
# Coupling data from GPCRdb
# ================================================================
def fetch_coupling_data():
    """
    Fetch G protein coupling data from GPCRdb.
    Returns dict: entry_name -> {Gs, Gi/o, Gq/11, G12/13} with levels.
    """
    print("\n[Fetching coupling data from GPCRdb...]")

    # GPCRdb coupling endpoint
    url = f"{GPCRDB_BASE_URL}/couplings/"
    data = fetch_json(url)

    if not data:
        print("  WARNING: Could not fetch coupling data from GPCRdb API")
        print("  Falling back to local coupling map from opsin_gq_project")
        return load_fallback_coupling()

    coupling_dict = {}
    for entry in data:
        entry_name = entry.get("entry_name", "")
        if not entry_name:
            continue

        coupling_dict[entry_name] = {
            "Gs": entry.get("gs", "none"),
            "Gi/o": entry.get("gio", "none"),
            "Gq/11": entry.get("gq11", "none"),
            "G12/13": entry.get("g1213", "none"),
            "source": entry.get("source", "gpcrdb"),
        }

    print(f"  Retrieved coupling data for {len(coupling_dict)} receptors")
    return coupling_dict


def load_fallback_coupling():
    """Load coupling data from the opsin_gq_project as fallback."""
    # Import from existing project
    old_project = os.path.join(os.path.dirname(DATA_DIR), "..", "opsin_gq_project")
    old_data = os.path.join(old_project, "data", "gpcrdb_coupling_dataset.json")

    if os.path.exists(old_data):
        with open(old_data, 'r', encoding='utf-8') as f:
            records = json.load(f)
        coupling_dict = {}
        for rec in records:
            desc = rec.get("coupling_description", "")
            entry_name = rec["entry_name"]
            coupling_dict[entry_name] = {
                "Gs": "primary" if "Gs primary" in desc else ("secondary" if "Gs secondary" in desc else "none"),
                "Gi/o": "primary" if "Gi/o primary" in desc else ("secondary" if "Gi/o secondary" in desc else "none"),
                "Gq/11": "primary" if "Gq/11 primary" in desc else ("secondary" if "Gq secondary" in desc else "none"),
                "G12/13": "primary" if "G12/13 primary" in desc else ("secondary" if "G12/13 secondary" in desc else "none"),
                "source": "fallback_opsin_project",
            }
        print(f"  Loaded fallback coupling for {len(coupling_dict)} receptors")
        return coupling_dict

    print("  ERROR: No coupling data available")
    return {}


# ================================================================
# Main pipeline
# ================================================================
def main():
    print("=" * 70)
    print("Phase 1a: Multi-species GPCR data collection")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target species: {len(TARGET_SPECIES)}")
    print(f"Target families: {len(TARGET_FAMILIES)}")
    print("=" * 70)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Fetch coupling annotations
    coupling_data = fetch_coupling_data()

    # Save raw coupling data
    coupling_file = os.path.join(DATA_DIR, "coupling_annotations.json")
    with open(coupling_file, 'w', encoding='utf-8') as f:
        json.dump(coupling_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {coupling_file}")

    # Step 2: Fetch proteins across all families and species
    target_species_set = set(TARGET_SPECIES)
    all_proteins = []
    species_counts = defaultdict(int)
    family_counts = defaultdict(int)
    skipped_no_coupling = 0

    for family_slug in TARGET_FAMILIES:
        print(f"\n[Family] {family_slug}")
        leaf_slugs = get_leaf_slugs(family_slug)
        print(f"  Leaf subfamilies: {len(leaf_slugs)}")

        for slug in leaf_slugs:
            url = f"{GPCRDB_BASE_URL}/proteinfamily/proteins/{slug}/"
            proteins = fetch_json(url)
            if not proteins:
                continue

            # Filter for target species
            filtered = [p for p in proteins if p.get("species") in target_species_set]

            for prot in filtered:
                entry_name = prot["entry_name"]
                species = prot["species"]
                accession = prot.get("accession", "")
                sequence = prot.get("sequence", "")

                if not sequence or len(sequence) < 100:
                    continue

                # Match coupling data (try exact match, then human ortholog)
                coupling = coupling_data.get(entry_name)
                if coupling is None:
                    # Try human ortholog name
                    human_name = entry_name.rsplit("_", 1)[0] + "_human"
                    coupling = coupling_data.get(human_name)

                if coupling is None:
                    skipped_no_coupling += 1
                    continue

                # Extract subfamily from family slug
                family_full = prot.get("family", slug)

                record = {
                    "entry_name": entry_name,
                    "name": prot.get("name", "").replace("<sub>", "").replace("</sub>", "")
                                                .replace("<i>", "").replace("</i>", ""),
                    "accession": accession,
                    "family": family_full,
                    "subfamily": slug,
                    "species": species,
                    "sequence": sequence,
                    "seq_length": len(sequence),
                    # 4-way G protein coupling labels
                    "gs_coupling": coupling.get("Gs", "none"),
                    "gio_coupling": coupling.get("Gi/o", "none"),
                    "gq11_coupling": coupling.get("Gq/11", "none"),
                    "g1213_coupling": coupling.get("G12/13", "none"),
                    # Binary labels for each G protein
                    "gs_label": 1 if coupling.get("Gs", "none") != "none" else 0,
                    "gio_label": 1 if coupling.get("Gi/o", "none") != "none" else 0,
                    "gq11_label": 1 if coupling.get("Gq/11", "none") != "none" else 0,
                    "g1213_label": 1 if coupling.get("G12/13", "none") != "none" else 0,
                    "coupling_source": coupling.get("source", "unknown"),
                }
                all_proteins.append(record)
                species_counts[species] += 1
                family_counts[family_slug] += 1

            time.sleep(0.15)

    # Step 3: Report statistics
    print(f"\n{'=' * 70}")
    print("Data collection complete")
    print(f"{'=' * 70}")
    print(f"Total receptors: {len(all_proteins)}")
    print(f"Skipped (no coupling data): {skipped_no_coupling}")

    print(f"\nBy species:")
    for sp, cnt in sorted(species_counts.items(), key=lambda x: -x[1]):
        print(f"  {sp}: {cnt}")

    print(f"\nBy family:")
    for fam, cnt in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {fam}: {cnt}")

    # Coupling statistics
    print(f"\nCoupling statistics (binary, all species):")
    for gp in ["gs", "gio", "gq11", "g1213"]:
        pos = sum(1 for p in all_proteins if p[f"{gp}_label"] == 1)
        print(f"  {gp}: {pos} positive / {len(all_proteins) - pos} negative")

    # Step 4: Save
    if all_proteins:
        # CSV
        csv_file = os.path.join(DATA_DIR, "gpcr_multispecies_dataset.csv")
        fieldnames = list(all_proteins[0].keys())
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_proteins)
        print(f"\nSaved CSV: {csv_file} ({os.path.getsize(csv_file)/1024:.1f} KB)")

        # JSON
        json_file = os.path.join(DATA_DIR, "gpcr_multispecies_dataset.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_proteins, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {json_file} ({os.path.getsize(json_file)/1024:.1f} KB)")

    # Step 5: Export UniProt accessions for AF2 download
    accessions = [(p["accession"], p["entry_name"], p["species"])
                  for p in all_proteins if p["accession"]]
    acc_file = os.path.join(DATA_DIR, "uniprot_accessions.tsv")
    with open(acc_file, 'w', encoding='utf-8') as f:
        f.write("accession\tentry_name\tspecies\n")
        for acc, name, sp in accessions:
            f.write(f"{acc}\t{name}\t{sp}\n")
    print(f"Saved accession list: {acc_file} ({len(accessions)} entries)")

    return all_proteins


if __name__ == "__main__":
    data = main()
