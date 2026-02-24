"""
LeakageGuard — Leakage-aware benchmarking for GPCR–G protein coupling prediction.

Provides tools to quantify phylogenetic data leakage, evaluate coupling
predictors under rigorous no-leak conditions, and analyse structure-aligned
selectivity determinants at Ballesteros–Weinstein G protein contact sites.
"""

__version__ = "0.1.0"
__author__ = "Guohao Lv, Xiaosong Wang, Lichuan Gu"

from .data.dataset import GPCRDataset, COUPLING_TARGETS
from .features.handcrafted import build_feature_matrix, extract_handcrafted_features
from .features.bw_site import (
    build_bw_feature_matrix,
    extract_bw_features,
    load_bw_cache,
    GP_CONTACT_SITES,
)
from .splits.strategies import (
    random_split,
    subfamily_split,
    seqcluster_split,
    grouped_kfold_cv,
    repeated_grouped_kfold_cv,
    seqcluster_kfold_cv,
)
from .models.classifiers import build_models, MODEL_REGISTRY
from .evaluation.metrics import (
    bootstrap_metrics,
    compute_fold_metrics,
    aggregate_cv_results,
    delong_permutation_test,
)

__all__ = [
    # Data
    "GPCRDataset", "COUPLING_TARGETS",
    # Features
    "build_feature_matrix", "extract_handcrafted_features",
    "build_bw_feature_matrix", "extract_bw_features", "load_bw_cache",
    "GP_CONTACT_SITES",
    # Splits
    "random_split", "subfamily_split", "seqcluster_split",
    "grouped_kfold_cv", "repeated_grouped_kfold_cv", "seqcluster_kfold_cv",
    # Models
    "build_models", "MODEL_REGISTRY",
    # Evaluation
    "bootstrap_metrics", "compute_fold_metrics", "aggregate_cv_results",
    "delong_permutation_test",
]
