"""
Smoke tests for LeakageGuard package.

Run with: pytest tests/ -v
"""
import numpy as np
import pytest


# ── Data ─────────────────────────────────────────────────────────────────

class TestDataset:
    def test_import(self):
        from leakageguard.data.dataset import GPCRDataset, COUPLING_TARGETS
        assert len(COUPLING_TARGETS) == 4

    def test_coupling_targets(self):
        from leakageguard import COUPLING_TARGETS
        assert "Gq" in COUPLING_TARGETS
        assert "Gs" in COUPLING_TARGETS


# ── Features ─────────────────────────────────────────────────────────────

class TestHandcrafted:
    def test_extract_single(self):
        from leakageguard.features.handcrafted import extract_handcrafted_features
        seq = "MGNCLHRAERDQRLENAQKLLEERLK" * 10  # 260 aa fake GPCR
        feat = extract_handcrafted_features(seq, "001_001_001_001")
        assert isinstance(feat, dict)
        assert len(feat) > 50
        assert "charge_ratio" in feat

    def test_build_matrix(self):
        from leakageguard.features.handcrafted import build_feature_matrix
        seqs = ["AAACCC" * 20, "KKKDDD" * 20, "LLLFFF" * 20]
        X, names = build_feature_matrix(seqs)
        assert X.shape[0] == 3
        assert X.shape[1] == len(names)
        assert X.shape[1] > 50


class TestBWSite:
    def test_encode_residue(self):
        from leakageguard.features.bw_site import _encode_residue
        assert _encode_residue("K") == [1, 0, 0, 0, 0]  # positive
        assert _encode_residue("D") == [0, 1, 0, 0, 0]  # negative
        assert _encode_residue("F") == [0, 0, 1, 1, 0]  # hydrophobic + aromatic
        assert _encode_residue(None) == [0, 0, 0, 0, 1]  # gap

    def test_contact_sites(self):
        from leakageguard.features.bw_site import GP_CONTACT_SITES
        assert len(GP_CONTACT_SITES) == 29
        assert "34.50" in GP_CONTACT_SITES
        assert "5.71" in GP_CONTACT_SITES


# ── Splits ───────────────────────────────────────────────────────────────

class TestSplits:
    def test_random_split(self):
        from leakageguard.splits.strategies import random_split
        y = np.array([0]*80 + [1]*20)
        tr, te = random_split(y, test_size=0.2)
        assert len(tr) + len(te) == 100
        assert len(te) == 20

    def test_grouped_kfold(self):
        from leakageguard.splits.strategies import grouped_kfold_cv
        y = np.array([0]*60 + [1]*40)
        families = [f"001_001_{i:03d}_001" for i in range(100)]
        folds = grouped_kfold_cv(y, families, n_folds=5, seed=42)
        assert len(folds) == 5
        # No overlap between train and test
        for tr, te in folds:
            assert len(set(tr) & set(te)) == 0
            assert len(tr) + len(te) == 100

    def test_repeated_grouped_kfold(self):
        from leakageguard.splits.strategies import repeated_grouped_kfold_cv
        y = np.array([0]*60 + [1]*40)
        families = [f"001_001_{i:03d}_001" for i in range(100)]
        results = repeated_grouped_kfold_cv(y, families, n_folds=5, n_repeats=3)
        assert len(results) == 15  # 5 folds × 3 repeats


# ── Models ───────────────────────────────────────────────────────────────

class TestModels:
    def test_build_models(self):
        from leakageguard.models.classifiers import build_models
        models = build_models(include_mlp=True, include_xgb=True)
        assert "LR" in models
        assert "RF" in models
        assert "MLP" in models
        assert "Ensemble" in models
        assert len(models) >= 6

    def test_model_fit_predict(self):
        from leakageguard.models.classifiers import build_models
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=50, n_features=10, random_state=42)
        models = build_models(include_mlp=False, include_xgb=False)
        for name, model in models.items():
            model.fit(X, y)
            proba = model.predict_proba(X)
            assert proba.shape == (50, 2)


# ── Evaluation ───────────────────────────────────────────────────────────

class TestEvaluation:
    def test_bootstrap_metrics(self):
        from leakageguard.evaluation.metrics import bootstrap_metrics
        y_true = np.array([0]*30 + [1]*20)
        y_prob = np.random.RandomState(42).uniform(0, 1, 50)
        result = bootstrap_metrics(y_true, y_prob, n_boot=100)
        assert "auc" in result
        assert len(result["auc"]) == 3  # mean, lo, hi

    def test_compute_fold_metrics(self):
        from leakageguard.evaluation.metrics import compute_fold_metrics
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.8, 0.7, 0.2, 0.9])
        m = compute_fold_metrics(y_true, y_prob)
        assert m["auc"] > 0.5
        assert 0 <= m["f1"] <= 1

    def test_aggregate_cv(self):
        from leakageguard.evaluation.metrics import aggregate_cv_results
        fold_results = [
            {"auc": 0.7, "prauc": 0.6, "f1": 0.5},
            {"auc": 0.8, "prauc": 0.7, "f1": 0.6},
        ]
        agg = aggregate_cv_results(fold_results)
        assert abs(agg["auc"]["mean"] - 0.75) < 1e-6
        assert agg["auc"]["std"] > 0


# ── Plotting ─────────────────────────────────────────────────────────────

class TestPlotting:
    def test_nature_style(self):
        from leakageguard.plotting.nature_style import set_nature_style, NATURE_PALETTE
        set_nature_style()
        assert len(NATURE_PALETTE) == 8

    def test_figure_creation(self):
        from leakageguard.plotting.nature_style import nature_single_col, nature_double_col
        import matplotlib
        matplotlib.use("Agg")
        fig1 = nature_single_col()
        assert abs(fig1.get_figwidth() - 3.50) < 0.01
        import matplotlib.pyplot as plt
        plt.close(fig1)

        fig2 = nature_double_col()
        assert abs(fig2.get_figwidth() - 7.20) < 0.01
        plt.close(fig2)


# ── CLI ──────────────────────────────────────────────────────────────────

class TestCLI:
    def test_import_cli(self):
        from leakageguard.cli import main
        assert callable(main)

    def test_version(self):
        from leakageguard import __version__
        assert __version__ == "0.1.0"


# ── Package top-level ────────────────────────────────────────────────────

class TestPackage:
    def test_all_exports(self):
        import leakageguard
        assert hasattr(leakageguard, "GPCRDataset")
        assert hasattr(leakageguard, "build_models")
        assert hasattr(leakageguard, "grouped_kfold_cv")
        assert hasattr(leakageguard, "bootstrap_metrics")
        assert hasattr(leakageguard, "GP_CONTACT_SITES")
