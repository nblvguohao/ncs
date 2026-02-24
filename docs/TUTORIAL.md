# LeakageGuard Tutorial

A step-by-step walkthrough for auditing GPCR–G protein coupling predictors under leakage-aware conditions.

---

## 1. Installation

```bash
git clone https://github.com/nblvguohao/ncs.git
cd ncs
pip install -e .
```

Verify:

```bash
leakageguard --version
# leakageguard 0.1.0
```

---

## 2. Explore the Dataset

```bash
leakageguard info
```

Or in Python:

```python
import leakageguard as lg

dataset = lg.GPCRDataset().load()
# Loaded 230 receptors
#   Gs: 48 positive, 182 negative
#   Gi: 109 positive, 121 negative
#   Gq: 91 positive, 139 negative
#   G12: 4 positive, 226 negative
```

---

## 3. Run a Quick Leakage Diagnostic

The fastest way to check for leakage is the CLI:

```bash
leakageguard diagnose --target Gq --n-folds 5
```

This compares AUC under random (leaky) vs subfamily (no-leak) CV for both handcrafted and BW-site features. If ΔAUC > 0.05, significant phylogenetic leakage is present.

---

## 4. Extract Features

### Handcrafted sequence features (99d)

```python
X_hc, names_hc = lg.build_feature_matrix(dataset.sequences, dataset.families)
print(X_hc.shape)  # (230, 99)
```

### BW-site physicochemical features (145d)

```python
bw_cache = lg.load_bw_cache()
X_bw, names_bw = lg.build_bw_feature_matrix(dataset.entry_names, bw_cache)
print(X_bw.shape)  # (230, 145)
```

### ESM-2 embeddings (optional, requires `pip install leakageguard[esm]`)

```python
from leakageguard.features.esm2 import extract_esm2_embeddings

X_esm = extract_esm2_embeddings(
    dataset.sequences, dataset.entry_names,
    model_name="esm2_t6_8M_UR50D", device="cpu",
)
print(X_esm.shape)  # (230, 320)
```

---

## 5. Set Up No-Leak Cross-Validation

```python
y = dataset.get_labels("Gq")

# Single round of 5-fold subfamily-grouped CV
folds = lg.grouped_kfold_cv(y, dataset.families, n_folds=5, seed=42)
print(f"{len(folds)} folds")

# Full benchmark: 5-fold × 10 repeats
cv_folds = lg.repeated_grouped_kfold_cv(
    y, dataset.families, n_folds=5, n_repeats=10, seed=42
)
print(f"{len(cv_folds)} total folds")  # 50
```

---

## 6. Train and Evaluate Models

```python
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

models = lg.build_models(include_mlp=True)
results = {name: [] for name in models}

for train_idx, test_idx in folds:
    X_tr, X_te = X_bw[train_idx], X_bw[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    for name, template in models.items():
        model = clone(template)
        model.fit(X_tr_s, y_tr)
        y_prob = model.predict_proba(X_te_s)[:, 1]

        fold_m = lg.compute_fold_metrics(y_te, y_prob)
        results[name].append(fold_m)

# Aggregate
for name, fold_list in results.items():
    agg = lg.aggregate_cv_results(fold_list)
    print(f"{name:10s}  AUC = {agg['auc']['mean']:.3f} ± {agg['auc']['std']:.3f}")
```

Expected output (BW-site, subfamily CV):
```
LR          AUC = 0.501 ± 0.118
RF          AUC = 0.636 ± 0.096
GBM         AUC = 0.596 ± 0.110
SVM         AUC = 0.531 ± 0.128
XGB         AUC = 0.608 ± 0.105
MLP         AUC = 0.598 ± 0.115
Ensemble    AUC = 0.599 ± 0.119
```

---

## 7. Quantify the Leakage Gradient

Compare performance at multiple sequence-identity thresholds:

```python
from sklearn.model_selection import StratifiedKFold

thresholds_results = {}

# Random CV (leaky baseline)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_folds = list(skf.split(X_bw, y))
# ... evaluate as above ...

# Sequence-cluster CV at decreasing thresholds
for thresh in [0.6, 0.5, 0.4, 0.3, 0.2]:
    folds = lg.seqcluster_kfold_cv(y, dataset.sequences, threshold=thresh)
    # ... evaluate as above ...

# Subfamily CV (strictest no-leak)
subfamily_folds = lg.grouped_kfold_cv(y, dataset.families)
# ... evaluate as above ...
```

The resulting AUC curve from random → subfamily CV reveals the **leakage gradient**.

---

## 8. Compare Two Models Statistically

```python
delta, p_value = lg.delong_permutation_test(y_true, y_prob_model_a, y_prob_model_b)
print(f"ΔAUC = {delta:.3f}, p = {p_value:.4f}")
```

---

## 9. Generate Publication Figures

```python
from leakageguard.plotting import set_nature_style, nature_single_col
import matplotlib.pyplot as plt

set_nature_style()
fig = nature_single_col()
ax = fig.add_subplot(111)
ax.bar(["Random", "Subfamily"], [0.835, 0.599], color=["#D55E00", "#0072B2"])
ax.set_ylabel("AUC-ROC")
fig.savefig("leakage_comparison.pdf", dpi=300)
```

Or render all 6 manuscript figures at once:

```bash
python scripts/render_figures.py --output-dir figures/nature
```

---

## 10. Full Benchmark via CLI

```bash
# Single-target Gq benchmark (5-fold × 10 repeats)
leakageguard benchmark --target Gq --n-folds 5 --n-repeats 10

# Multi-target evaluation (Gs, Gi/o, Gq/11)
leakageguard multilabel --n-folds 5 --n-repeats 5
```

---

## 11. Audit Your Own Predictor

To evaluate a custom coupling predictor:

1. Prepare predictions as a CSV with columns `entry_name` and `y_prob`
2. Load the dataset and your predictions
3. Run no-leak CV using `grouped_kfold_cv`

```python
import pandas as pd
import leakageguard as lg

dataset = lg.GPCRDataset().load()
y = dataset.get_labels("Gq")

# Your predictions
preds = pd.read_csv("my_predictions.csv")
y_prob = preds.set_index("entry_name").loc[dataset.entry_names, "y_prob"].values

# Evaluate under no-leak conditions
folds = lg.grouped_kfold_cv(y, dataset.families, n_folds=5)
# Compare held-out predictions per fold ...

# Or use bootstrap CIs on the full set
result = lg.bootstrap_metrics(y, y_prob, n_boot=1000)
print(f"AUC = {result['auc'][0]:.3f} [{result['auc'][1]:.3f}–{result['auc'][2]:.3f}]")
```

---

## Summary

| Task | Command / Function |
|------|-------------------|
| Quick leakage check | `leakageguard diagnose` |
| Full benchmark | `leakageguard benchmark` |
| Multi-target | `leakageguard multilabel` |
| Dataset info | `leakageguard info` |
| Load data | `lg.GPCRDataset().load()` |
| BW features | `lg.build_bw_feature_matrix()` |
| No-leak CV | `lg.grouped_kfold_cv()` |
| All models | `lg.build_models()` |
| Bootstrap CIs | `lg.bootstrap_metrics()` |
| Nature figures | `set_nature_style()` |
