#!/usr/bin/env python3
"""
Fetch GPCRdb residue annotations to populate gpcrdb_residues_cache.json.
This is needed for BW-site analysis in ESM-2 and AlphaFold validation scripts.
"""
import os
import json
import time
import subprocess
import pandas as pd
import os

# Specific receptors needed for AlphaFold validation
AF_VALIDATION_RECEPTORS = [
    "opn4_human", "hrh1_human", "acm1_human", "5ht2a_human",
    "oprm_human", "acm4_human", "opsd_human", "adrb2_human",
    "glr_human", "drd1_human"
]

def fetch_gpcrdb_cache():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    csv_path = os.path.join(data_dir, "gpcrdb_coupling_dataset.csv")
    cache_path = os.path.join(data_dir, "gpcrdb_residues_cache.json")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    all_entries = df["entry_name"].tolist()
    
    # Prioritize AF validation receptors, then add others up to limit
    entries_to_fetch = []
    for entry in AF_VALIDATION_RECEPTORS:
        if entry in all_entries:
            entries_to_fetch.append(entry)
        else:
            print(f"Warning: {entry} not found in dataset CSV.")
            
    # Add other entries up to a reasonable limit (e.g. 50 total) to save time
    limit = 50
    for entry in all_entries:
        if len(entries_to_fetch) >= limit:
            break
        if entry not in entries_to_fetch:
            entries_to_fetch.append(entry)
            
    print(f"Fetching residue data for {len(entries_to_fetch)} receptors...")
    
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} existing entries from cache.")
        except:
            pass
            
    # Load manually downloaded oprm_human.json if exists
    oprm_path = os.path.join(data_dir, "oprm_human.json")
    if os.path.exists(oprm_path) and "oprm_human" not in cache:
        try:
            with open(oprm_path, "r") as f:
                cache["oprm_human"] = json.load(f)
            print("Loaded oprm_human from manual file.")
        except:
            pass

    success_count = 0
    for i, entry in enumerate(entries_to_fetch):
        if entry in cache:
            print(f"[{i+1}/{len(entries_to_fetch)}] {entry}: already cached")
            success_count += 1
            continue
            
        url = f"https://gpcrdb.org/services/residues/extended/{entry}/"
        try:
            print(f"[{i+1}/{len(entries_to_fetch)}] Fetching {entry}...", end="", flush=True)
            # Use curl instead of requests
            result = subprocess.run(
                ["curl", "-s", "-f", "--max-time", "60", url],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                cache[entry] = data
                print(" OK")
                success_count += 1
            else:
                print(f" Failed (curl error: {result.returncode})")
        except Exception as e:
            print(f" Error: {e}")
            
        # Be nice to the server
        time.sleep(0.5)
        
    print(f"Saving cache with {len(cache)} entries to {cache_path}")
    with open(cache_path, "w") as f:
        json.dump(cache, f)

if __name__ == "__main__":
    fetch_gpcrdb_cache()
