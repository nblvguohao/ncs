#!/usr/bin/env python3
"""
Phase 1c: Compute ESM-2 per-residue embeddings for all GPCRs.

Uses esm2_t33_650M_UR50D (650M params) for richer representations.
Saves per-residue embeddings as .pt files for graph node features.

Output:
  - data/embeddings/{entry_name}.pt  (per-residue embedding tensor)
  - data/embedding_log.json
"""
import os
import sys
import json
import time
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DATA_DIR, EMBEDDING_DIR,
    ESM2_MODEL_NAME, ESM2_REPR_LAYER, ESM2_BATCH_SIZE, MAX_SEQ_LENGTH
)


def load_dataset():
    """Load multi-species GPCR dataset."""
    json_file = os.path.join(DATA_DIR, "gpcr_multispecies_dataset.json")
    if not os.path.exists(json_file):
        print(f"ERROR: {json_file} not found. Run 01_fetch_multispecies_gpcr.py first.")
        sys.exit(1)
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("Phase 1c: Compute ESM-2 per-residue embeddings")
    print(f"Model: {ESM2_MODEL_NAME}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    # Load data
    dataset = load_dataset()
    print(f"Total receptors: {len(dataset)}")

    # Check existing
    existing = set()
    for f in os.listdir(EMBEDDING_DIR):
        if f.endswith(".pt"):
            existing.add(f.replace(".pt", ""))
    remaining = [d for d in dataset if d["entry_name"] not in existing]
    print(f"Already computed: {len(existing)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All embeddings already computed!")
        return

    # Load ESM-2 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {ESM2_MODEL_NAME}...")

    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    print("Model loaded.")

    # Process one-by-one to avoid OOM on long sequences
    log = {}
    total_done = 0
    cpu_fallback_count = 0

    for i, rec in enumerate(remaining):
        entry_name = rec["entry_name"]
        seq = rec["sequence"][:MAX_SEQ_LENGTH]
        batch_data = [(entry_name, seq)]

        success = False
        for run_device in [device, torch.device("cpu")]:
            try:
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(run_device)
                run_model = model if run_device == device else model.cpu()

                with torch.no_grad():
                    results = run_model(batch_tokens, repr_layers=[ESM2_REPR_LAYER],
                                        return_contacts=False)

                token_reps = results["representations"][ESM2_REPR_LAYER]
                seq_len = len(seq)
                rep = token_reps[0, 1:seq_len + 1].cpu()  # (seq_len, 1280)

                out_path = os.path.join(EMBEDDING_DIR, f"{entry_name}.pt")
                torch.save(rep, out_path)
                log[entry_name] = {"shape": list(rep.shape), "seq_length": seq_len}

                if run_device.type == "cpu" and device.type == "cuda":
                    cpu_fallback_count += 1
                    model.to(device)  # Move back to GPU

                success = True
                break

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower() or "CUBLAS" in str(e):
                    if run_device == device:
                        torch.cuda.empty_cache()
                        continue  # Retry on CPU
                log[entry_name] = {"error": str(e)}
                break
            except Exception as e:
                log[entry_name] = {"error": str(e)}
                break

        if success:
            total_done += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
            print(f"  Progress: {i+1}/{len(remaining)} (done={total_done}, cpu_fallback={cpu_fallback_count})")

        # Periodic GPU cache cleanup
        if device.type == "cuda" and (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # Save log
    log_file = os.path.join(DATA_DIR, "embedding_log.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Embedding computation complete: {total_done} receptors")
    errors = sum(1 for v in log.values() if "error" in v)
    print(f"Errors: {errors}")
    print(f"Saved log: {log_file}")


if __name__ == "__main__":
    main()
