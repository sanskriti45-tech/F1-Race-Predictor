from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

import src.config as config_module
from src.config import Settings
from src.evaluation.backtest import backtest_summary, walk_forward_backtest
from src.features.dataset import build_training_dataset, feature_columns
from src.models.predict import save_model_bundle
from src.models.train import compute_feature_importance, train_baseline_gbm
from tests.fixtures import make_large_synthetic_races_df

from pathlib import Path

DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "src" / "app" / "dashboard.py")


def _patch_settings(tmp_path, monkeypatch):
    """Repoint the module-level `settings` object src.app.dashboard reads
    at import time. AppTest re-executes dashboard.py's top-level code on
    every .run(), including `from src.config import settings` — so
    reassigning the attribute on the already-imported src.config module
    (rather than mutating the frozen Settings instance) is picked up on
    the next run."""
    fake_settings = Settings(
        project_root=tmp_path,
        fastf1_cache_dir=tmp_path / "cache",
        data_raw_dir=tmp_path / "raw",
        data_processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        log_level="INFO",
    )
    monkeypatch.setattr(config_module, "settings", fake_settings)
    return fake_settings


@pytest.fixture
def empty_project(tmp_path, monkeypatch):
    return _patch_settings(tmp_path, monkeypatch)


@pytest.fixture
def full_project(tmp_path, monkeypatch):
    settings = _patch_settings(tmp_path, monkeypatch)
    settings.ensure_dirs()

    races_df = make_large_synthetic_races_df(n_races=24, seed=9)
    races_df.to_csv(settings.data_processed_dir / "races.csv", index=False)

    dataset = build_training_dataset(races_df)
    feats = feature_columns(dataset)
    model = train_baseline_gbm(dataset, feats)
    importance = compute_feature_importance(model, dataset, feats)
    save_model_bundle(model, feats, settings.models_dir / "baseline_gbm.joblib", feature_importance=importance)

    predictions = walk_forward_backtest(dataset, min_train_races=10, retrain_every=1)
    predictions.to_csv(settings.reports_dir / "backtest_predictions.csv", index=False)
    summary = backtest_summary(predictions)
    with open(settings.reports_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, default=float)

    return settings


class TestDashboardWithNoDataYet:
    def test_runs_without_exception_and_shows_missing_data_message(self, empty_project):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run()
        assert not at.exception
        error_texts = " ".join(e.value for e in at.error)
        assert "No processed data" in error_texts

    def test_disclaimer_always_shown(self, empty_project):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run()
        warning_texts = " ".join(w.value for w in at.warning)
        assert "uncertain estimates" in warning_texts


class TestDashboardWithFullData:
    def test_runs_without_exception(self, full_project):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run()
        assert not at.exception

    def test_shows_predicted_finish_table(self, full_project):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run()
        assert not at.exception
        assert len(at.dataframe) > 0

    def test_status_expander_reports_all_three_artifacts_present(self, full_project):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run()
        markdown_texts = " ".join(m.value for m in at.markdown)
        assert "Historical races table" in markdown_texts
        assert "Trained model" in markdown_texts
        assert "Backtest report" in markdown_texts
        assert "not yet built" not in markdown_texts
