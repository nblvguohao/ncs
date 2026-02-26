#!/usr/bin/env python3
"""
Phase 1b: Download AlphaFold2 predicted structures for all GPCRs.

Downloads PDB files from AlphaFold DB using UniProt accessions.
Handles rate limiting and resumable downloads.

Output:
  - data/structures/{accession}.pdb  (one per receptor)
  - data/af2_download_log.json       (download status log)
"""
import os
import sys
import json
import time
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_DIR, STRUCTURE_DIR, AF2_FILE_URL, AF2_API_URL

# ================================================================
# Download helpers
# ================================================================
def download_af2_pdb(accession, output_dir):
    """
    Download a single AlphaFold2 predicted structure via the API.
    The API returns the correct versioned URL (currently v6).
    Returns (accession, success, message).
    """
    out_path = os.path.join(output_dir, f"{accession}.pdb")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return (accession, True, "already_exists")

    try:
        # Step 1: Query API for the correct PDB URL
        api_url = f"{AF2_API_URL}/{accession}"
        api_resp = requests.get(api_url, timeout=30)
        if api_resp.status_code == 404:
            return (accession, False, "not_in_alphafold_db")
        elif api_resp.status_code != 200:
            return (accession, False, f"API HTTP {api_resp.status_code}")

        data = api_resp.json()
        if isinstance(data, list) and len(data) > 0:
            pdb_url = data[0].get("pdbUrl", "")
        else:
            return (accession, False, "no_pdbUrl_in_api_response")

        if not pdb_url:
            return (accession, False, "empty_pdbUrl")

        # Step 2: Download the PDB file
        resp = requests.get(pdb_url, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            return (accession, True, f"downloaded ({len(resp.content)//1024} KB)")
        else:
            return (accession, False, f"PDB HTTP {resp.status_code}")

    except requests.exceptions.RequestException as e:
        return (accession, False, str(e))


def main():
    print("=" * 70)
    print("Phase 1b: Download AlphaFold2 predicted structures")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(STRUCTURE_DIR, exist_ok=True)

    # Load accession list
    acc_file = os.path.join(DATA_DIR, "uniprot_accessions.tsv")
    if not os.path.exists(acc_file):
        print(f"ERROR: {acc_file} not found. Run 01_fetch_multispecies_gpcr.py first.")
        sys.exit(1)

    accessions = []
    with open(acc_file, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                accessions.append({
                    "accession": parts[0],
                    "entry_name": parts[1],
                    "species": parts[2],
                })

    print(f"Total accessions to download: {len(accessions)}")

    # Check how many already exist
    existing = sum(1 for a in accessions
                   if os.path.exists(os.path.join(STRUCTURE_DIR, f"{a['accession']}.pdb")))
    print(f"Already downloaded: {existing}")
    print(f"Remaining: {len(accessions) - existing}")

    # Download with thread pool (limited concurrency)
    log = {}
    success_count = 0
    fail_count = 0

    # Sequential download with rate limiting (be nice to AF2 servers)
    for i, entry in enumerate(accessions):
        acc = entry["accession"]
        result = download_af2_pdb(acc, STRUCTURE_DIR)
        acc_id, ok, msg = result
        log[acc_id] = {"success": ok, "message": msg, "entry_name": entry["entry_name"]}

        if ok:
            success_count += 1
        else:
            fail_count += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(accessions)} "
                  f"(success={success_count}, fail={fail_count})")

        # Rate limiting: 0.1s between requests
        if msg != "already_exists":
            time.sleep(0.1)

    # Summary
    print(f"\n{'=' * 70}")
    print("Download complete")
    print(f"{'=' * 70}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

    # Save download log
    log_file = os.path.join(DATA_DIR, "af2_download_log.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"Saved download log: {log_file}")

    # Report failures
    failures = {k: v for k, v in log.items() if not v["success"]}
    if failures:
        print(f"\nFailed downloads ({len(failures)}):")
        for acc, info in list(failures.items())[:20]:
            print(f"  {acc} ({info['entry_name']}): {info['message']}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
