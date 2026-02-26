# SE3-GNN Minimal Server Package

This folder contains the minimum files required to run the GPCR SE(3)-GNN pipeline on a server.

## Included

- `code/01_fetch_multispecies_gpcr.py`
- `code/02_download_af2_structures.py`
- `code/03_compute_esm2_embeddings.py`
- `code/04_build_3d_graphs.py`
- `code/07_run_training.py`
- `code/contrastive_training.py`
- `code/se3_gnn_model.py`
- `code/analyze_results.py`
- `configs/config.py`
- `requirements.txt`
- `environment.yml`
- `run_on_server.sh`

## Quick Start (Server)

```bash
conda create -n gpcr_gnn python=3.11 -y
conda activate gpcr_gnn
pip install -r requirements.txt

# Data pipeline
python code/01_fetch_multispecies_gpcr.py
python code/02_download_af2_structures.py
python code/03_compute_esm2_embeddings.py
python code/04_build_3d_graphs.py

# Training
python code/07_run_training.py --mode standard --epochs 30 --batch_size 2
python code/07_run_training.py --mode zeroshot --epochs 50 --batch_size 2

# Analysis
python code/analyze_results.py
```

## Notes

- If GPU memory is insufficient, reduce `--batch_size` to `1`.
- If `No graphs found` appears, run scripts `01` to `04` first.
- Standard outputs are written to the `results/` directory.
