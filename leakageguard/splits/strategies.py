"""
Data splitting strategies for leakage-aware GPCR coupling evaluation.

Includes:
  - Random stratified split
  - Subfamily-level split (zero subfamily overlap)
  - Sequence-identity clustering split
  - Grouped k-fold cross-validation (subfamily-based)
  - Repeated grouped k-fold CV
"""
import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold


def _get_kmer_set(seq, k=3):
    return set(seq[i : i + k] for i in range(len(seq) - k + 1))


def _get_subfamily(family_slug):
    parts = family_slug.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else family_slug


# ── Single-split strategies ──────────────────────────────────────────────

def random_split(y, test_size=0.2, seed=42):
    """Stratified random 80/20 split."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return tr, te


def subfamily_split(y, families, test_size=0.2, seed=42):
    """Subfamily-level split: entire subfamilies held out as test."""
    rng = np.random.RandomState(seed)
    sf_map = defaultdict(list)
    for i, f in enumerate(families):
        sf_map[_get_subfamily(f)].append(i)
    sfs = list(sf_map.keys())
    rng.shuffle(sfs)
    target = int(len(y) * test_size)
    te, tr = [], []
    n_te = 0
    for sf in sfs:
        members = sf_map[sf]
        if n_te < target:
            te.extend(members)
            n_te += len(members)
        else:
            tr.extend(members)
    return np.array(tr), np.array(te)


def seqcluster_split(y, sequences, threshold=0.3, test_size=0.2, seed=42):
    """Sequence-identity clustering split using k-mer Jaccard distance."""
    rng = np.random.RandomState(seed)
    n = len(sequences)
    kmer_sets = [_get_kmer_set(s, 3) for s in sequences]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(kmer_sets[i] & kmer_sets[j])
            union = len(kmer_sets[i] | kmer_sets[j])
            d = 1 - (inter / union if union > 0 else 0)
            dist[i, j] = dist[j, i] = d
    Z = linkage(squareform(dist), method="average")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")
    cl_map = defaultdict(list)
    for i, c in enumerate(labels):
        cl_map[c].append(i)
    cls = list(cl_map.keys())
    rng.shuffle(cls)
    target = int(n * test_size)
    te, tr = [], []
    n_te = 0
    for c in cls:
        members = cl_map[c]
        if n_te < target:
            te.extend(members)
            n_te += len(members)
        else:
            tr.extend(members)
    return np.array(tr), np.array(te)


# ── Cross-validation strategies ──────────────────────────────────────────

def grouped_kfold_cv(y, families, n_folds=5, seed=42):
    """Subfamily-grouped k-fold CV.

    Yields (train_idx, test_idx) tuples where no subfamily appears
    in both train and test within the same fold.
    """
    rng = np.random.RandomState(seed)
    sf_map = defaultdict(list)
    for i, f in enumerate(families):
        sf_map[_get_subfamily(f)].append(i)

    # Assign subfamilies to folds, roughly balanced by sample count
    sfs = list(sf_map.keys())
    rng.shuffle(sfs)
    fold_sizes = [0] * n_folds
    fold_sfs = [[] for _ in range(n_folds)]

    # Greedy assignment: add each subfamily to the smallest fold
    for sf in sorted(sfs, key=lambda s: -len(sf_map[s])):
        smallest = int(np.argmin(fold_sizes))
        fold_sfs[smallest].append(sf)
        fold_sizes[smallest] += len(sf_map[sf])

    # Generate fold indices
    folds = []
    for fold_idx in range(n_folds):
        te_idx = []
        for sf in fold_sfs[fold_idx]:
            te_idx.extend(sf_map[sf])
        tr_idx = [i for i in range(len(y)) if i not in set(te_idx)]
        folds.append((np.array(tr_idx), np.array(te_idx)))

    return folds


def repeated_grouped_kfold_cv(y, families, n_folds=5, n_repeats=10, seed=42):
    """Repeated subfamily-grouped k-fold CV.

    Yields (repeat_idx, fold_idx, train_idx, test_idx) tuples.
    """
    results = []
    for rep in range(n_repeats):
        rep_seed = seed + rep * 1000
        folds = grouped_kfold_cv(y, families, n_folds=n_folds, seed=rep_seed)
        for fold_idx, (tr, te) in enumerate(folds):
            results.append((rep, fold_idx, tr, te))
    return results


def seqcluster_kfold_cv(y, sequences, threshold=0.3, n_folds=5, seed=42):
    """Sequence-cluster-grouped k-fold CV.

    Groups receptors by k-mer Jaccard clustering, then performs
    grouped k-fold where no cluster appears in both train and test.
    """
    rng = np.random.RandomState(seed)
    n = len(sequences)
    kmer_sets = [_get_kmer_set(s, 3) for s in sequences]

    # Build distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(kmer_sets[i] & kmer_sets[j])
            union = len(kmer_sets[i] | kmer_sets[j])
            d = 1 - (inter / union if union > 0 else 0)
            dist[i, j] = dist[j, i] = d

    Z = linkage(squareform(dist), method="average")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")

    cl_map = defaultdict(list)
    for i, c in enumerate(labels):
        cl_map[c].append(i)

    cls = list(cl_map.keys())
    rng.shuffle(cls)

    # Greedy assign clusters to folds
    fold_sizes = [0] * n_folds
    fold_cls = [[] for _ in range(n_folds)]
    for c in sorted(cls, key=lambda x: -len(cl_map[x])):
        smallest = int(np.argmin(fold_sizes))
        fold_cls[smallest].append(c)
        fold_sizes[smallest] += len(cl_map[c])

    folds = []
    for fold_idx in range(n_folds):
        te_idx = []
        for c in fold_cls[fold_idx]:
            te_idx.extend(cl_map[c])
        tr_idx = [i for i in range(n) if i not in set(te_idx)]
        folds.append((np.array(tr_idx), np.array(te_idx)))

    return folds
