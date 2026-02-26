#!/usr/bin/env python3
"""
Prepare FASTA input for PRECOG server.
"""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from leakageguard.data.dataset import GPCRDataset

def main():
    dataset = GPCRDataset().load()
    output_path = os.path.join(PROJECT_DIR, "results", "precog_input.fasta")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing {len(dataset.receptors)} sequences to {output_path}...")
    
    with open(output_path, "w") as f:
        for name, seq in zip(dataset.entry_names, dataset.sequences):
            # PRECOG uses "Accession" or "Gene Name". 
            # We use entry_name as header for easy mapping back.
            # Format: >entry_name
            # SEQUENCE
            f.write(f">{name}\n{seq}\n")
            
    print("Done.")
    print(f"Upload {output_path} to https://precog.russelllab.org/")

if __name__ == "__main__":
    main()
