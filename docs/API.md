# LeakageGuard API Reference

## Top-Level Imports

```python
import leakageguard as lg
```

All public symbols are available from the top-level package:

| Symbol | Module | Description |
|--------|--------|-------------|
| `GPCRDataset` | `data.dataset` | Dataset loader for GPCRdb coupling data |
| `COUPLING_TARGETS` | `data.dataset` | `["Gs", "Gi", "Gq", "G12"]` |
| `build_feature_matrix` | `features.handcrafted` | 99d sequence-level features |
| `extract_handcrafted_features` | `features.handcrafted` | Single-sequence feature dict |
| `build_bw_feature_matrix` | `features.bw_site` | 145d BW-site physicochemical features |
| `extract_bw_features` | `features.bw_site` | Single-receptor BW features |
| `load_bw_cache` | `features.bw_site` | Load GPCRdb residue cache JSON |
| `GP_CONTACT_SITES` | `features.bw_site` | 29 BW contact positions (list) |
| `random_split` | `splits.strategies` | Random stratified train/test split |
| `subfamily_split` | `splits.strategies` | Subfamily-level no-leak split |
| `seqcluster_split` | `splits.strategies` | Sequence-identity cluster split |
| `grouped_kfold_cv` | `splits.strategies` | Subfamily-grouped k-fold CV |
| `repeated_grouped_kfold_cv` | `splits.strategies` | Repeated grouped k-fold CV |
| `seqcluster_kfold_cv` | `splits.strategies` | Sequence-cluster k-fold CV |
| `build_models` | `models.classifiers` | Build all 7 classifiers |
| `MODEL_REGISTRY` | `models.classifiers` | Dict of model name → constructor |
| `bootstrap_metrics` | `evaluation.metrics` | Bootstrap confidence intervals |
| `compute_fold_metrics` | `evaluation.metrics` | Per-fold AUC/PR-AUC/F1 |
| `aggregate_cv_results` | `evaluation.metrics` | Aggregate fold results |
| `delong_permutation_test` | `evaluation.metrics` | Permutation-based AUC comparison |

---

## `leakageguard.data.dataset`

### `GPCRDataset(data_dir=None)`

Load and manage the GPCR coupling dataset.

```python
dataset = lg.GPCRDataset(data_dir="data/").load()
```

**Attributes** (after `.load()`):
- `entry_names` — list of GPCRdb entry names (e.g. `"oprm_human"`)
- `sequences` — list of amino acid sequences
- `families` — list of subfamily strings (e.g. `"001_001_001_001"`)
- `df` — full pandas DataFrame

**Methods**:
- `.load()` → self — Load CSV and residue cache
- `.get_labels(target)` → np.ndarray — Binary labels for `"Gs"`, `"Gi"`, `"Gq"`, or `"G12"`
- `.summary()` — Print dataset statistics

---

## `leakageguard.features.handcrafted`

### `build_feature_matrix(sequences, families=None)`

Build a (n_samples, 99) feature matrix from amino acid sequences.

**Returns**: `(X: np.ndarray, feature_names: list[str])`

### `extract_handcrafted_features(sequence, family=None)`

Extract features for a single sequence. Returns a dict of feature name → value.

---

## `leakageguard.features.bw_site`

### `build_bw_feature_matrix(entry_names, bw_cache, sites=None)`

Build a (n_samples, 145) feature matrix using physicochemical properties at 29 BW contact sites.

**Parameters**:
- `entry_names` — list of GPCRdb entry names
- `bw_cache` — dict from `load_bw_cache()`
- `sites` — optional list of BW positions (default: `GP_CONTACT_SITES`)

**Returns**: `(X: np.ndarray, feature_names: list[str])`

### `load_bw_cache(path=None)`

Load the GPCRdb residue annotations JSON cache. Default path: `data/gpcrdb_residues_cache.json`.

### `GP_CONTACT_SITES`

List of 29 Ballesteros–Weinstein positions at the GPCR–G protein interface:
```
["34.50", "34.51", "34.52", "34.53", "34.54", "34.55", "34.56", "34.57",
 "3.49", "3.50", "3.53", "3.54", "3.55", "3.56",
 "5.61", "5.64", "5.65", "5.67", "5.68", "5.69", "5.71",
 "6.29", "6.32", "6.33", "6.36",
 "8.47", "8.48", "8.49", "8.51"]
```

---

## `leakageguard.features.esm2`

> Requires optional dependencies: `pip install leakageguard[esm]`

### `extract_esm2_embeddings(sequences, entry_names, model_name, batch_size, device)`

Mean-pooled ESM-2 sequence embeddings. Returns `(n_samples, embed_dim)`.

### `extract_esm2_bw_embeddings(sequences, entry_names, bw_cache, model_name, sites, batch_size, device)`

Per-residue ESM-2 embeddings at BW contact sites. Returns `(X, feature_names)` where X is `(n_samples, n_sites * embed_dim)`.

### `extract_esm2_attention(sequences, entry_names, bw_cache, model_name, sites, batch_size, device)`

Attention weights at BW positions, averaged over layers and heads. Returns `(attn_matrix, site_names)` where attn_matrix is `(n_samples, n_sites)`.

---

## `leakageguard.splits.strategies`

### `random_split(y, test_size=0.2, seed=42)`

Stratified random split. Returns `(train_indices, test_indices)`.

### `subfamily_split(y, families, test_size=0.2, seed=42)`

Split by subfamily — no subfamily appears in both train and test. Returns `(train_idx, test_idx)`.

### `seqcluster_split(y, sequences, threshold=0.3, test_size=0.2, seed=42)`

Cluster by k-mer Jaccard similarity, then split clusters. Returns `(train_idx, test_idx)`.

### `grouped_kfold_cv(y, families, n_folds=5, seed=42)`

Subfamily-grouped k-fold CV. Returns list of `(train_idx, test_idx)` tuples.

### `repeated_grouped_kfold_cv(y, families, n_folds=5, n_repeats=10, seed=42)`

Repeated grouped k-fold. Returns list of `(repeat, fold, train_idx, test_idx)` tuples.

### `seqcluster_kfold_cv(y, sequences, threshold=0.3, n_folds=5, seed=42)`

Sequence-cluster grouped k-fold. Returns list of `(train_idx, test_idx)` tuples.

---

## `leakageguard.models.classifiers`

### `build_models(include_xgb=True, include_mlp=True)`

Returns a dict of model name → sklearn estimator:

| Key | Model | Notes |
|-----|-------|-------|
| `"LR"` | Logistic Regression | `class_weight="balanced"` |
| `"RF"` | Random Forest | 200 trees, balanced |
| `"GBM"` | Gradient Boosting | 200 estimators |
| `"SVM"` | SVM (RBF) | probability=True, balanced |
| `"XGB"` | XGBoost | optional (`include_xgb=True`) |
| `"MLP"` | MLP Neural Network | (128, 64), early stopping |
| `"Ensemble"` | Soft Voting | RF + GBM + SVM |

---

## `leakageguard.evaluation.metrics`

### `bootstrap_metrics(y_true, y_prob, n_boot=1000, seed=42)`

Compute AUC-ROC, PR-AUC, and F1 with bootstrap 95% CIs.

**Returns**: `{"auc": (mean, lo, hi), "prauc": (mean, lo, hi), "f1": (mean, lo, hi)}`

### `compute_fold_metrics(y_true, y_prob)`

Compute metrics for a single CV fold. Returns `{"auc": float, "prauc": float, "f1": float}`.

### `aggregate_cv_results(fold_results)`

Aggregate a list of fold metric dicts. Returns `{"auc": {"mean": float, "std": float}, ...}`.

### `delong_permutation_test(y_true, y_prob_a, y_prob_b, n_perm=10000)`

Permutation test for AUC difference between two models. Returns `(delta_auc, p_value)`.

---

## `leakageguard.plotting.nature_style`

### `set_nature_style()`

Apply Nature Computational Science rcParams globally (Helvetica 7pt, no top/right spines, color-blind palette).

### `nature_single_col(height_ratio=0.75)`

Create a 89mm (3.50in) wide figure. Returns `matplotlib.figure.Figure`.

### `nature_double_col(height_ratio=0.45)`

Create a 183mm (7.20in) wide figure. Returns `matplotlib.figure.Figure`.

### `add_panel_label(ax, label, x, y, fontsize)`

Add bold panel label (a, b, c…) in Nature style.

### `save_nature_fig(fig, path, formats=("pdf", "png"))`

Save in PDF (300dpi) + PNG (600dpi) for submission.

### `NATURE_PALETTE`

8-colour colour-blind-safe palette (Wong 2011).

---

## CLI Reference

```bash
leakageguard --version
leakageguard diagnose  --target Gq --n-folds 5 [--data-dir data/]
leakageguard benchmark --target Gq --n-folds 5 --n-repeats 10 [--skip-seqcluster]
leakageguard multilabel --n-folds 5 --n-repeats 5
leakageguard info [--data-dir data/]
```
