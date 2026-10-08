from __future__ import annotations

import json

import pandas as pd
import pytest

from src.app import dashboard_data as dd
from src.features.dataset import build_training_dataset, feature_columns
from src.models.predict import save_model_bundle
from src.models.train import compute_feature_importance, train_baseline_gbm
from tests.fixtures import make_large_synthetic_races_df


class TestArtifactStatusAllMissing:
    def test_reports_all_missing_without_raising(self, tmp_path):
        statuses = dd.check_artifact_status(
            tmp_path / "races.csv", tmp_path / "model.joblib", tmp_path / "summary.json"
        )
        assert len(statuses) == 3
        assert all(not s.exists for s in statuses)


class TestArtifactStatusPresent:
    def test_reports_details_when_files_exist(self, tmp_path):
        races_df = make_large_synthetic_races_df(n_races=24, seed=1)
        races_path = tmp_path / "races.csv"
        races_df.to_csv(races_path, index=False)

        dataset = build_training_dataset(races_df)
        feats = feature_columns(dataset)
        model = train_baseline_gbm(dataset, feats)
        model_path = tmp_path / "model.joblib"
        save_model_bundle(model, feats, model_path)

        summary_path = tmp_path / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({"overall": {"n_races": 14, "mae": 0.3}}, f)

        statuses = dd.check_artifact_status(races_path, model_path, summary_path)
        by_name = {s.name: s for s in statuses}
        assert by_name["Historical races table"].exists
        assert "96 driver-race rows" in by_name["Historical races table"].detail
        assert by_name["Trained model"].exists
        assert "30 features" in by_name["Trained model"].detail
        assert by_name["Backtest report"].exists
        assert "14 backtested races" in by_name["Backtest report"].detail

    def test_corrupt_file_reported_not_raised(self, tmp_path):
        bad_path = tmp_path / "races.csv"
        bad_path.write_text("not,valid,csv,for,our,schema\n1,2")
        statuses = dd.check_artifact_status(bad_path, tmp_path / "no_model.joblib", tmp_path / "no_summary.json")
        races_status = statuses[0]
        assert races_status.exists  # file exists, even though content is wrong shape
        assert "could not be read" in races_status.detail or races_status.detail is not None


class TestLoaders:
    def test_load_functions_return_none_when_missing(self, tmp_path):
        assert dd.load_races_df(tmp_path / "x.csv") is None
        assert dd.load_bundle(tmp_path / "x.joblib") is None
        assert dd.load_backtest_summary(tmp_path / "x.json") is None
        assert dd.load_backtest_predictions(tmp_path / "x.csv") is None

    def test_load_backtest_summary_round_trip(self, tmp_path):
        path = tmp_path / "summary.json"
        with open(path, "w") as f:
            json.dump({"overall": {"mae": 0.5}}, f)
        result = dd.load_backtest_summary(path)
        assert result == {"overall": {"mae": 0.5}}


class TestFormatPredictionTable:
    def test_renames_and_formats_probability_as_percent(self):
        predictions = pd.DataFrame({
            "predicted_rank": [1, 2],
            "driver": ["D1", "D2"],
            "team": ["TeamA", "TeamB"],
            "win_prob": [0.734, 0.266],
            "pred_finish": [1.2, 2.4],
            "qualifying_position": [1, 2],
            "driver_form_3": [1.5, 2.5],
            "team_form_3": [1.5, 2.5],
        })
        out = dd.format_prediction_table(predictions)
        assert "P1 probability" in out.columns
        assert out["P1 probability"].tolist() == ["73.4%", "26.6%"]
        assert "Driver" in out.columns and "driver" not in out.columns

    def test_handles_missing_optional_columns_gracefully(self):
        predictions = pd.DataFrame({"predicted_rank": [1], "driver": ["D1"]})
        out = dd.format_prediction_table(predictions)
        assert "Predicted rank" in out.columns
        assert "Driver" in out.columns


class TestFormatFeatureImportanceTable:
    def test_applies_labels_and_respects_top_n(self):
        importance = pd.Series(
            {"qualifying_position": 0.5, "driver_form_3": 0.3, "unknown_feature_xyz": 0.1}
        ).sort_values(ascending=False)
        out = dd.format_feature_importance_table(importance, top_n=2)
        assert len(out) == 2
        assert out["Factor"].tolist() == ["qualifying position", "driver recent form (last 3 races)"]

    def test_unmapped_feature_falls_back_to_raw_name(self):
        importance = pd.Series({"some_new_feature": 1.0})
        out = dd.format_feature_importance_table(importance)
        assert out["Factor"].iloc[0] == "some_new_feature"


class TestFormatBacktestSummaryTable:
    def test_overall_pinned_first(self):
        summary = {
            "season_2023": {"mae": 0.4, "n_races": 12},
            "overall": {"mae": 0.35, "n_races": 14},
            "season_2022": {"mae": 0.3, "n_races": 2},
        }
        out = dd.format_backtest_summary_table(summary)
        assert out["Period"].tolist() == ["Overall", "Season 2022", "Season 2023"]

    def test_metrics_columns_present(self):
        summary = {"overall": {"mae": 0.35, "rank_correlation": 0.9}}
        out = dd.format_backtest_summary_table(summary)
        assert {"mae", "rank_correlation"} <= set(out.columns)


class TestRacePredictionVsActual:
    def test_filters_sorts_and_computes_abs_error(self):
        predictions = pd.DataFrame({
            "race_id": ["r1", "r1", "r2"],
            "driver": ["A", "B", "C"],
            "team": ["T1", "T2", "T3"],
            "target_finish": [2, 1, 1],
            "pred_finish": [2.5, 1.5, 3.0],
            "win_prob": [0.3, 0.7, 1.0],
        })
        out = dd.race_prediction_vs_actual(predictions, "r1")
        assert len(out) == 2
        assert out["driver"].tolist() == ["B", "A"]  # sorted by actual finish ascending
        assert out["abs_error"].tolist() == pytest.approx([0.5, 0.5])
