from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.app.cli import entrants_from_qualifying, main
from src.features.dataset import build_training_dataset, feature_columns
from src.models.predict import save_model_bundle
from src.models.train import train_baseline_gbm
from tests.fixtures import make_large_synthetic_races_df


class TestEntrantsFromQualifying:
    def test_maps_fastf1_columns_to_entrant_dicts(self):
        quali_results = pd.DataFrame({
            "Abbreviation": ["VER", "HAM"],
            "TeamName": ["RedBull", "Mercedes"],
            "Position": [1, 2],
        })
        entrants = entrants_from_qualifying(quali_results)
        assert entrants == [
            {"driver": "VER", "team": "RedBull", "qualifying_position": 1, "grid_position": 1},
            {"driver": "HAM", "team": "Mercedes", "qualifying_position": 2, "grid_position": 2},
        ]


@pytest.fixture
def populated_project(tmp_path, monkeypatch):
    """A tmp project dir with a real races.csv and a real trained model
    bundle on disk, with src.app.cli's `settings` reference patched to
    point at it — so main() exercises the real file-loading code paths.
    (settings is a frozen dataclass, so we patch the module-level name
    cli.py imports rather than mutating the Settings instance itself.)"""
    import src.app.cli as cli_module

    races_df = make_large_synthetic_races_df(n_races=24, seed=5)
    dataset = build_training_dataset(races_df)
    feats = feature_columns(dataset)
    model = train_baseline_gbm(dataset, feats)

    processed_dir = tmp_path / "processed"
    models_dir_path = tmp_path / "models"
    processed_dir.mkdir()
    races_path = processed_dir / "races.csv"
    races_df.to_csv(races_path, index=False)
    save_model_bundle(model, feats, models_dir_path / "baseline_gbm.joblib")

    class _FakeSettings:
        data_processed_dir = processed_dir
        models_dir = models_dir_path

    monkeypatch.setattr(cli_module, "settings", _FakeSettings())
    return races_df


class TestCLIMissingData:
    def test_missing_races_csv_returns_1_and_reports_clearly(self, tmp_path, monkeypatch, capsys):
        import src.app.cli as cli_module

        class _FakeSettings:
            data_processed_dir = tmp_path / "nowhere"
            models_dir = tmp_path / "nowhere_models"

        monkeypatch.setattr(cli_module, "settings", _FakeSettings())

        exit_code = main(["--season", "2023", "--round", "1"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "Cannot generate a prediction" in captured.err

    def test_requires_next_race_or_explicit_season_and_round(self):
        with pytest.raises(SystemExit):
            main([])


class TestCLIExplicitRace:
    def test_full_flow_with_mocked_fastf1_session(self, populated_project, capsys):
        races_df = populated_project
        target = races_df[(races_df["season"] == 2023) & (races_df["round"] == 6)]

        fake_quali_results = pd.DataFrame({
            "Abbreviation": target["driver"].tolist(),
            "TeamName": target["team"].tolist(),
            "Position": target["qualifying_position"].tolist(),
        })
        fake_session = MagicMock()
        fake_session.event = {"EventName": "Race 6", "Location": "C2"}

        fake_result = MagicMock()
        fake_result.ok = True
        fake_result.results = fake_quali_results
        fake_result.error = None
        fake_result.warnings = []
        fake_result.session = fake_session

        with patch("src.app.cli.load_session", return_value=fake_result):
            exit_code = main(["--season", "2023", "--round", "6"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "PREDICTED FINISH" in captured.out
        assert "No future race-result data was used." in captured.out
        assert "P1 probability" in captured.out

    def test_qualifying_load_failure_returns_1_not_a_crash(self, populated_project, capsys):
        fake_result = MagicMock()
        fake_result.ok = False
        fake_result.results = None
        fake_result.error = "simulated network failure"
        fake_result.warnings = []
        fake_result.session = None

        with patch("src.app.cli.load_session", return_value=fake_result):
            exit_code = main(["--season", "2023", "--round", "6"])

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "Qualifying has not completed" in captured.err
