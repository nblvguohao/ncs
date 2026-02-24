# GPCR–G Protein Coupling Leakage Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A leakage-aware benchmarking framework for GPCR–G protein coupling prediction, with structure-aligned determinants analysis and melanopsin (OPN4) case study.

## Key Features

- **Multi-target prediction**: Independent classifiers for all four G protein families (Gs, Gi/o, Gq/11, G12/13)
- **Leakage-aware evaluation**: Subfamily-grouped k-fold cross-validation (5-fold × 10 repeats) replacing single train/test splits
- **Structure-aligned features**: Ballesteros–Weinstein (BW) site physicochemical encodings at 29 G protein contact positions
- **Comprehensive model suite**: LR, RF, GBM, SVM, XGBoost, MLP, and Ensemble
- **Standalone leakage diagnostic**: `leakage_test.py` for auditing any coupling predictor
- **Reproducible**: All results regenerable from public data sources (GPCRdb, GtoPdb, PDB)

## Quick Start

```bash
git clone https://github.com/nblvguohao/ncs.git
cd ncs
conda env create -f environment.yml
conda activate gpcr_ncs
```

### Run the main benchmark

```bash
# Full multi-target benchmark (Gs, Gi, Gq, G12) with grouped k-fold CV
python scripts/run_benchmark.py --target all --n-folds 5 --n-repeats 10

# Single target (faster)
python scripts/run_benchmark.py --target Gq --n-folds 5 --n-repeats 5
```

### Run multi-label evaluation

```bash
python scripts/run_multilabel.py --n-folds 5 --n-repeats 5
```

### Run leakage diagnostic

```bash
python scripts/leakage_test.py --target Gq
```

## Repository Structure

```
ncs/
├── configs/
│   └── default.yaml              # Experiment configuration
│
├── src/                           # Core library
│   ├── data/
│   │   └── dataset.py            # Multi-label GPCR dataset loader
│   ├── features/
│   │   ├── handcrafted.py        # 99d sequence-level features
│   │   └── bw_site.py            # 145d BW-site physicochemical features
│   ├── splits/
│   │   └── strategies.py         # Random, subfamily, sequence-cluster splits + grouped k-fold CV
│   ├── models/
│   │   └── classifiers.py        # Model registry (LR, RF, GBM, SVM, XGB, MLP, Ensemble)
│   └── evaluation/
│       └── metrics.py            # Bootstrap CI, DeLong test, CV aggregation
│
├── scripts/                       # Executable pipelines
│   ├── run_benchmark.py          # Main benchmark (single-split + grouped CV)
│   ├── run_multilabel.py         # Multi-label G protein prediction
│   └── leakage_test.py           # Standalone leakage diagnostic
│
├── data/                          # Input data (cached from GPCRdb)
│   ├── gpcrdb_coupling_dataset.csv
│   └── gpcrdb_residues_cache.json
│
├── results/                       # Generated CSV results
├── figures/                       # Generated figures
├── environment.yml                # Conda environment
└── README.md
```

## Methodological Upgrades (vs. initial BIB submission)

| Aspect | BIB version | NCS version |
|--------|-------------|-------------|
| Evaluation | Single 80/20 split + bootstrap CI | 5-fold × 10-repeat grouped CV |
| Targets | Gq/11 only | All 4 families (Gs, Gi/o, Gq/11, G12/13) |
| Models | LR, RF, GBM, SVM, XGB, Ensemble | + MLP neural network |
| Features | Handcrafted, BW-site, ESM-2 | + combined, + ESM-2 at BW sites |
| Splits | Random, subfamily, seqcluster | + grouped k-fold CV for all |
| Code | Monolithic scripts | Modular `src/` library + `scripts/` |

## Dataset

233 human GPCRs (Class A, B1, C) from GPCRdb with multi-label coupling annotations:

| Target | Positive | Negative |
|--------|----------|----------|
| Gs     | 48       | 182      |
| Gi/o   | 109      | 121      |
| Gq/11  | 91       | 139      |
| G12/13 | 4        | 226      |

## Citation

If you use this code, please cite:

```
Lv G, Wang X, Gu L. Addressing data leakage in GPCR–G protein coupling prediction:
a benchmarking protocol with structure-aligned determinants and melanopsin case study.
(2025, submitted to Nature Computational Science)
```

## License

MIT License. See [LICENSE](LICENSE) for details.
