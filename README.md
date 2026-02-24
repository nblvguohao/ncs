# LeakageGuard

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)]()

**Leakage-aware benchmarking for GPCR–G protein coupling prediction.**

LeakageGuard quantifies phylogenetic data leakage in coupling predictors, evaluates models under rigorous no-leak cross-validation, and analyses structure-aligned selectivity determinants at Ballesteros–Weinstein G protein contact sites.

> **Paper**: Lv G, Wang X, Gu L. *Addressing data leakage in GPCR–G protein coupling prediction: a benchmarking protocol with structure-aligned determinants and melanopsin case study.* (2025, submitted to Nature Computational Science)

---

## Installation

```bash
# From source (recommended)
git clone https://github.com/nblvguohao/ncs.git
cd ncs
pip install -e .

# With ESM-2 support
pip install -e ".[esm]"

# With dev tools (pytest, black, ruff)
pip install -e ".[dev]"

# Or via conda
conda env create -f environment.yml
conda activate gpcr_ncs
pip install -e .
```

## Quick Start

### Command-line interface

```bash
# Leakage diagnostic — compare random vs no-leak performance
leakageguard diagnose --target Gq

# Full repeated CV benchmark
leakageguard benchmark --target Gq --n-folds 5 --n-repeats 10

# Multi-target evaluation (Gs, Gi/o, Gq/11)
leakageguard multilabel --n-folds 5 --n-repeats 5

# Dataset summary
leakageguard info
```

### Python API

```python
import leakageguard as lg

# Load dataset
dataset = lg.GPCRDataset().load()
y = dataset.get_labels("Gq")

# Extract features
X_hc, names_hc = lg.build_feature_matrix(dataset.sequences, dataset.families)
bw_cache = lg.load_bw_cache()
X_bw, names_bw = lg.build_bw_feature_matrix(dataset.entry_names, bw_cache)

# No-leak cross-validation
folds = lg.grouped_kfold_cv(y, dataset.families, n_folds=5, seed=42)
for train_idx, test_idx in folds:
    # train and evaluate your model here
    pass

# Build all 7 models
models = lg.build_models(include_mlp=True)

# Evaluation with bootstrap CIs
from leakageguard import bootstrap_metrics
result = bootstrap_metrics(y_true, y_prob, n_boot=1000)
print(f"AUC = {result['auc'][0]:.3f} [{result['auc'][1]:.3f}–{result['auc'][2]:.3f}]")
```

### ESM-2 attention analysis

```python
from leakageguard.features.esm2 import extract_esm2_attention

# Extract attention weights at BW contact sites
attn_matrix, site_names = extract_esm2_attention(
    sequences, entry_names, bw_cache,
    model_name="esm2_t6_8M_UR50D", device="cuda",
)
```

## Key Results

| Feature Set | Random CV | Subfamily CV (no-leak) | ΔAUC |
|-------------|-----------|------------------------|------|
| Handcrafted (99d) | 0.670 ± 0.063 | 0.487 ± 0.100 | 0.18 |
| **BW-site (145d)** | **0.835 ± 0.059** | **0.599 ± 0.119** | **0.24** |
| Combined (244d) | 0.759 ± 0.062 | 0.521 ± 0.108 | 0.24 |

**Multi-target (BW-site Ensemble, no-leak CV):**

| Target | AUC-ROC | PR-AUC |
|--------|---------|--------|
| G_s | 0.820 ± 0.104 | 0.666 ± 0.176 |
| G_i/o | 0.677 ± 0.107 | 0.635 ± 0.140 |
| G_q/11 | 0.599 ± 0.128 | 0.547 ± 0.168 |
| **Macro** | **0.699** | **0.616** |

## Repository Structure

```
ncs/
├── leakageguard/                  # pip-installable Python package
│   ├── __init__.py               # Public API exports
│   ├── cli.py                    # CLI: diagnose, benchmark, multilabel, info
│   ├── data/
│   │   └── dataset.py            # Multi-label GPCR dataset (Gs/Gi/Gq/G12)
│   ├── features/
│   │   ├── handcrafted.py        # 99d sequence-level features
│   │   ├── bw_site.py            # 145d BW-site physicochemical features
│   │   └── esm2.py              # ESM-2 embeddings + attention extraction
│   ├── splits/
│   │   └── strategies.py         # Grouped k-fold CV, subfamily/seqcluster splits
│   ├── models/
│   │   └── classifiers.py        # 7-model registry (LR/RF/GBM/SVM/XGB/MLP/Ensemble)
│   ├── evaluation/
│   │   └── metrics.py            # Bootstrap CI, DeLong test, CV aggregation
│   └── plotting/
│       └── nature_style.py       # Nature Computational Science figure style
│
├── scripts/                       # Executable pipelines
│   ├── run_benchmark.py          # Full grouped CV benchmark
│   ├── run_multilabel.py         # Multi-target evaluation
│   ├── run_esm2_leakage.py       # ESM-2 attention + leakage gradient
│   ├── render_figures.py         # Nature-format figure generation
│   ├── pymol_render.py           # PyMOL structural figure scripts
│   └── leakage_test.py           # Standalone leakage diagnostic
│
├── tests/                         # pytest test suite (19 tests)
├── data/                          # Input data (cached from GPCRdb)
├── configs/                       # Experiment configuration
├── figures/nature/                # Publication-ready figures (PDF + 600dpi PNG)
├── pyproject.toml                 # Package metadata and dependencies
├── environment.yml                # Conda environment
└── LICENSE                        # MIT License
```

## Dataset

230 human GPCRs (Class A, B1, C) from GPCRdb with multi-label coupling annotations:

| Target | Positive | Negative |
|--------|----------|----------|
| Gs     | 48       | 182      |
| Gi/o   | 109      | 121      |
| Gq/11  | 91       | 139      |
| G12/13 | 4        | 226      |

## Citation

```bibtex
@article{lv2025leakageguard,
  title={Addressing data leakage in GPCR--G protein coupling prediction:
         a benchmarking protocol with structure-aligned determinants
         and melanopsin case study},
  author={Lv, Guohao and Wang, Xiaosong and Gu, Lichuan},
  journal={Nature Computational Science},
  year={2025},
  note={Submitted}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
