# Server Offline Run Guide (China Mainland)

All scripts are designed to run offline after initial data download.
Required data files are already included in this repository.

## Quick Start (No Internet Required)

```bash
# 1. Clone repository
git clone https://github.com/nblvguohao/ncs.git
cd ncs

# 2. Install dependencies
conda env create -f environment.yml
conda activate ncs

# 3. Verify data files exist
ls -la data/
# Expected files:
# - gpcrdb_coupling_dataset.csv
# - gpcrdb_coupling_dataset.json  
# - gpcrdb_residues_cache.json
# - esm2_bw_embeddings.npz
# - esm2_35m_bw_embeddings.npz
# - kinase/kinase_dataset.csv
# - kinase/kinase_processed.csv

# 4. Run experiments (all offline)
```

## Experiments

### 1. Kinase Cross-Family Generalization (CPU, ~30min)
```bash
python scripts/run_kinase_generalization.py --n-folds 5 --n-repeats 10
# Results: results/kinase_generalization/
```

### 2. ESM-2 Attention Analysis (CPU, ~6-8h or GPU ~1-2h)
```bash
# CPU (small model)
python scripts/run_esm2_attention.py --model esm2_t6_8M_UR50D --device cpu

# GPU (larger model, if available)
python scripts/run_esm2_attention.py --model esm2_t12_35M_UR50D --device cuda
# Results: results/esm2_attention/, figures/esm2_attention/
```

### 3. AlphaFold-Multimer Validation (GPU Required)

#### Step 1: Prepare FASTA inputs (already done, offline)
```bash
# FASTA files already exist in results/alphafold_validation/fasta_inputs/
# No network needed
ls results/alphafold_validation/fasta_inputs/*.fasta
```

#### Step 2: Run AF-Multimer predictions (GPU, ~20-80h)
```bash
# Install ColabFold if not available
pip install colabfold[alphafold]

# Run batch prediction
bash results/alphafold_validation/run_colabfold.sh
# Or manually:
colabfold_batch --model-type alphafold2_multimer_v3 \
  --num-recycle 3 --num-models 5 --amber \
  results/alphafold_validation/fasta_inputs/ \
  results/alphafold_validation/af_predictions/
```

#### Step 3: Analyze predicted contacts (CPU)
```bash
python scripts/run_alphafold_validation.py --step analyze
# Results: results/alphafold_validation/
```

## Data Dependencies

All required data is included in the repository:

| File | Purpose | Size |
|------|---------|------|
| `data/gpcrdb_coupling_dataset.csv` | GPCR coupling labels | 120KB |
| `data/gpcrdb_coupling_dataset.json` | Rich coupling metadata | 171KB |
| `data/gpcrdb_residues_cache.json` | BW site annotations | 42MB |
| `data/esm2_bw_embeddings.npz` | ESM-2 embeddings (650M) | 6.9MB |
| `data/esm2_35m_bw_embeddings.npz` | ESM-2 embeddings (35M) | 425KB |
| `data/kinase/kinase_dataset.csv` | UniProt kinase download | 1.2MB |
| `data/kinase/kinase_processed.csv` | Processed kinase data | 1.1MB |

## Network Requirements

- **Kinase script**: Uses cached UniProt data; no network needed
- **ESM-2 script**: Downloads ESM-2 models on first run (requires internet once)
- **AF-Multimer**: Requires GPU but no network after ColabFold installation

## Expected Outputs

After running all experiments:

```
results/
├── kinase_generalization/
│   ├── fig_kinase_gpcr_comparison.png
│   └── kinase_benchmark_results.csv
├── esm2_attention/
│   ├── attention_stats.csv
│   ├── layer_enrichment.csv
│   └── ...
└── alphafold_validation/
    ├── af_predictions/          # AF-Multimer outputs
    ├── af_validation_results.csv
    └── fig_af_validation_heatmap.png

figures/
└── esm2_attention/
    ├── fig_esm2_attention_analysis.png
    └── fig_attention_per_fdr_site.png
```

## Troubleshooting

### GPCRdb Timeout
If you see errors like:
```
Error: HTTPSConnectionPool(host='gpcrdb.org', port=443): Read timed out
```
This means the script is trying to fetch data online. Ensure:
1. `data/gpcrdb_residues_cache.json` exists
2. Scripts use local cache (they should by default)

### ESM-2 Model Download
First run of ESM-2 will download models (~100MB-1GB). After that, it's cached locally.

### AF-Multimer GPU Memory
If GPU memory is insufficient, reduce batch size or use smaller models:
```bash
colabfold_batch --model-type alphafold2_multimer_v3 \
  --num-models 3 --max-recycle 2 \
  --batch-size 1 ...
```

## Summary

- **Kinase generalization**: ✅ Fully offline, CPU-only
- **ESM-2 attention**: ✅ Offline after initial model download
- **AF-Multimer**: ✅ Offline FASTA prep, GPU required for prediction

All data files are committed to the repository for seamless offline execution in China Mainland.
