"""
Model registry for GPCR coupling prediction.

Includes classical ML models + MLP neural network baseline.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_SEED = 42


def build_models(include_mlp=True, include_xgb=True, seed=RANDOM_SEED):
    """Build all models for benchmarking.

    Returns
    -------
    dict : model_name -> sklearn estimator
    """
    models = {}

    models["LR"] = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced",
        random_state=seed, solver="lbfgs",
    )
    models["RF"] = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=2,
        class_weight="balanced", random_state=seed,
    )
    models["GBM"] = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    models["SVM"] = SVC(
        kernel="rbf", C=10, gamma="scale", probability=True,
        class_weight="balanced", random_state=seed,
    )

    if include_xgb and HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1.5, random_state=seed,
            use_label_encoder=False, eval_metric="logloss",
            verbosity=0,
        )

    if include_mlp:
        models["MLP"] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=32,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )

    # Ensemble (RF + GBM + SVM)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=2,
        class_weight="balanced", random_state=seed,
    )
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    svm = SVC(
        kernel="rbf", C=10, gamma="scale", probability=True,
        class_weight="balanced", random_state=seed,
    )
    models["Ensemble"] = VotingClassifier(
        estimators=[("rf", rf), ("gbm", gbm), ("svm", svm)],
        voting="soft", weights=[2, 2, 1],
    )

    return models


MODEL_REGISTRY = list(build_models(include_mlp=True, include_xgb=True).keys())
