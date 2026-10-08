from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.app.api import _is_allowed_origin, build_next_race_payload
from src.data.fastf1_client import UpcomingRaceInfo
from src.features.dataset import build_training_dataset, feature_columns
from src.models.predict import ModelBundle, save_model_bundle
from src.models.train import compute_feature_importance, train_baseline_gbm
from tests.fixtures import make_large_synthetic_races_df


class TestCorsAllowlist:
    def test_localhost_allowed(self):
        assert _is_allowed_origin("http://localhost:5500")
        assert _is_allowed_origin("http://127.0.0.1:5500")

    def test_file_origin_null_allowed(self):
        assert _is_allowed_origin("null")

    def test_external_origin_rejected(self):
        assert not _is_allowed_origin("https://evil.example.com")
        assert not _is_allowed_origin(None)


class TestPayloadHonestEmptyState:
    def test_no_data_on_disk_and_fastf1_unreachable(self, tmp_path, monkeypatch):
        import src.config as config_module
        fake_settings = config_module.Settings(
            project_root=tmp_path, fastf1_cache_dir=tmp_path / "cache",
            data_raw_dir=tmp_path / "raw", data_processed_dir=tmp_path / "processed",
            models_dir=tmp_path / "models", reports_dir=tmp_path / "reports",
        )
        monkeypatch.setattr(config_module, "settings", fake_settings)
        monkeypatch.setattr("src.app.api.settings", fake_settings)

        with patch("src.app.api.get_upcoming_race", side_effect=RuntimeError("network blocked")):
            payload = build_next_race_payload()

        assert payload["fastf1Available"] is False
        assert payload["historicalDataLoaded"] is False
        assert payload["modelLoaded"] is False
        assert payload["predictionGenerated"] is False
        assert payload["nextRace"] is None
        assert payload["predictions"] == []
        assert payload["backtest"] is None
        assert payload["history"] == []
        assert "network blocked" in payload["error"]
        # json-serializable, matching what actually goes over the wire
        json.dumps(payload, default=str)


class TestPayloadPopulatedViaRealPipeline:
    """Builds real races.csv + a real trained model bundle via the actual
    backend functions (synthetic input data, since this sandbox can't
    reach FastF1 — but every number is computed by the real pipeline,
    nothing is hand-written), then mocks only the two genuinely
    network-dependent calls (get_upcoming_race, load_session) to point at
    that real data, and checks the API layer wires it all together
    correctly without adding any prediction logic of its own."""

    @pytest.fixture
    def populated_env(self, tmp_path, monkeypatch):
        import src.config as config_module
        fake_settings = config_module.Settings(
            project_root=tmp_path, fastf1_cache_dir=tmp_path / "cache",
            data_raw_dir=tmp_path / "raw", data_processed_dir=tmp_path / "processed",
            models_dir=tmp_path / "models", reports_dir=tmp_path / "reports",
        )
        fake_settings.ensure_dirs()
        monkeypatch.setattr(config_module, "settings", fake_settings)
        monkeypatch.setattr("src.app.api.settings", fake_settings)

        races_df = make_large_synthetic_races_df(n_races=24, seed=13)
        races_df.to_csv(fake_settings.data_processed_dir / "races.csv", index=False)

        dataset = build_training_dataset(races_df)
        feats = feature_columns(dataset)
        model = train_baseline_gbm(dataset, feats)
        importance = compute_feature_importance(model, dataset, feats)
        save_model_bundle(model, feats, fake_settings.models_dir / "baseline_gbm.joblib", feature_importance=importance)

        target_season, target_round = 2023, 1  # exists in the synthetic fixture
        race_rows = races_df[(races_df["season"] == target_season) & (races_df["round"] == target_round)]

        fake_race_info = UpcomingRaceInfo(
            season=target_season, round=target_round, event="Synthetic Test GP",
            circuit=race_rows["circuit"].iloc[0], date=datetime(2023, 3, 5, tzinfo=timezone.utc),
            qualifying_completed=True, retrieved_at=datetime.now(timezone.utc),
        )
        fake_quali_results = pd.DataFrame({
            "Abbreviation": race_rows["driver"].tolist(),
            "TeamName": race_rows["team"].tolist(),
            "Position": race_rows["qualifying_position"].tolist(),
        })
        fake_session_result = MagicMock(ok=True, results=fake_quali_results, error=None, warnings=[])

        return fake_race_info, fake_session_result, race_rows

    def test_full_real_pipeline_payload(self, populated_env):
        fake_race_info, fake_session_result, race_rows = populated_env
        with patch("src.app.api.get_upcoming_race", return_value=fake_race_info), \
             patch("src.app.api.load_session", return_value=fake_session_result):
            payload = build_next_race_payload()

        assert payload["fastf1Available"] is True
        assert payload["historicalDataLoaded"] is True
        assert payload["modelLoaded"] is True
        assert payload["predictionGenerated"] is True
        assert payload["nextRace"]["name"] == "Synthetic Test GP"
        assert len(payload["predictions"]) == len(race_rows)

        drivers_in_payload = {p["driver"] for p in payload["predictions"]}
        assert drivers_in_payload == set(race_rows["driver"])

        # every row has a real predicted_rank and win_prob the model computed
        ranks = sorted(p["predicted_rank"] for p in payload["predictions"])
        assert ranks == list(range(1, len(race_rows) + 1))
        assert all(isinstance(p["win_prob"], float) for p in payload["predictions"])
        probs_sum = sum(p["win_prob"] for p in payload["predictions"])
        assert probs_sum == pytest.approx(1.0, abs=1e-6)

        assert len(payload["featureValues"]) > 0
        assert all("feature" in f and "value" in f for f in payload["featureValues"])

        json.dumps(payload, default=str)  # must be JSON-serializable end to end

    def test_predictions_match_direct_predict_upcoming_race_call(self, populated_env):
        """Cross-check: the API's numbers must be IDENTICAL to calling the
        real backend function directly — proving the API adds no logic."""
        from src.models.predict import load_model_bundle, predict_upcoming_race
        import src.config as config_module
        from src.app.cli import entrants_from_qualifying

        fake_race_info, fake_session_result, race_rows = populated_env
        settings = config_module.settings

        races_df = pd.read_csv(settings.data_processed_dir / "races.csv", parse_dates=["date"])
        bundle = load_model_bundle(settings.models_dir / "baseline_gbm.joblib")
        entrants = entrants_from_qualifying(fake_session_result.results)
        direct = predict_upcoming_race(
            races_df, bundle, season=fake_race_info.season, round_number=fake_race_info.round,
            circuit=fake_race_info.circuit, entrants=entrants, qualifying_completed=True,
        )

        with patch("src.app.api.get_upcoming_race", return_value=fake_race_info), \
             patch("src.app.api.load_session", return_value=fake_session_result):
            payload = build_next_race_payload()

        api_by_driver = {p["driver"]: p for p in payload["predictions"]}
        for _, row in direct.iterrows():
            api_row = api_by_driver[row["driver"]]
            assert api_row["predicted_rank"] == row["predicted_rank"]
            assert api_row["pred_finish"] == pytest.approx(row["pred_finish"])
            assert api_row["win_prob"] == pytest.approx(row["win_prob"])
