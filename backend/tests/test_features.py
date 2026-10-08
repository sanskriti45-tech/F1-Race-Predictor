"""
Tests for src.features.* using the synthetic races_df in tests/fixtures.py.
Expected values below were computed by hand from that fixture — see the
comments in each test for the arithmetic.
"""
from __future__ import annotations

import math

from src.features.circuit_form import circuit_history_features
from src.features.cutoff import races_before, races_before_race_id
from src.features.dataset import build_training_dataset, feature_columns
from src.features.driver_form import driver_form_features
from src.features.team_form import team_form_features
from tests.fixtures import make_synthetic_races_df


class TestCutoff:
    def test_races_before_excludes_target_round_and_later(self):
        df = make_synthetic_races_df()
        history = races_before(df, 2022, 2)
        assert set(history["round"].unique()) == {1}
        assert set(history["season"].unique()) == {2022}

    def test_races_before_includes_earlier_seasons_entirely(self):
        df = make_synthetic_races_df()
        history = races_before(df, 2023, 1)
        # everything from 2022 (all 3 rounds), nothing from 2023
        assert set(history["season"].unique()) == {2022}
        assert set(history["round"].unique()) == {1, 2, 3}

    def test_races_before_race_id_matches_explicit_call(self):
        df = make_synthetic_races_df()
        a = races_before_race_id(df, "2022_3")
        b = races_before(df, 2022, 3)
        assert a.equals(b)

    def test_first_race_of_dataset_has_empty_history(self):
        df = make_synthetic_races_df()
        history = races_before(df, 2022, 1)
        assert history.empty


class TestDriverForm:
    def test_ver_form_before_2023_round1(self):
        df = make_synthetic_races_df()
        history = races_before(df, 2023, 1)  # all of 2022: finishes 2,1,1 / quali 3,1,2 / points 18,25,25
        feats = driver_form_features(history, "VER", windows=(3, 5))

        assert math.isclose(feats["driver_form_3"], (2 + 1 + 1) / 3)
        assert math.isclose(feats["driver_quali_form_3"], (3 + 1 + 2) / 3)
        assert feats["driver_points_3"] == 68
        assert feats["driver_finish_rate_3"] == 1.0
        assert feats["driver_form_3_available"] is True

        # only 3 prior races exist, window is 5 -> partial/"growing" window,
        # same numeric average as window 3, but flagged unavailable
        assert math.isclose(feats["driver_form_5"], feats["driver_form_3"])
        assert feats["driver_form_5_available"] is False
        assert feats["driver_prior_starts"] == 3

    def test_cold_start_driver_has_nan_not_zero(self):
        df = make_synthetic_races_df()
        history = races_before(df, 2022, 1)  # empty history
        feats = driver_form_features(history, "VER", windows=(3, 5))
        assert math.isnan(feats["driver_form_3"])
        assert feats["driver_prior_starts"] == 0
        assert feats["driver_form_3_available"] is False


class TestTeamForm:
    def test_redbull_form_before_2023_round1_counts_races_not_rows(self):
        df = make_synthetic_races_df()
        history = races_before(df, 2023, 1)
        feats = team_form_features(history, "RedBull", windows=(3, 5))
        assert math.isclose(feats["team_form_3"], (2 + 1 + 1) / 3)
        assert feats["team_points_3"] == 68
        assert feats["team_prior_races"] == 3


class TestCircuitForm:
    def test_ver_bahrain_history_before_2023_bahrain(self):
        df = make_synthetic_races_df()
        history = races_before(df, 2023, 1)  # only 2022 Bahrain (round 1) is at this circuit
        feats = circuit_history_features(history, "VER", "RedBull", "Bahrain")
        assert feats["circuit_driver_avg_finish"] == 2.0
        assert feats["circuit_driver_prior_starts"] == 1
        assert feats["circuit_team_avg_finish"] == 2.0

    def test_no_prior_starts_at_circuit_is_nan_with_zero_count(self):
        df = make_synthetic_races_df()
        history = races_before(df, 2022, 1)
        feats = circuit_history_features(history, "VER", "RedBull", "Bahrain")
        assert math.isnan(feats["circuit_driver_avg_finish"])
        assert feats["circuit_driver_prior_starts"] == 0


class TestSeasonFormViaDataset:
    def test_first_race_of_a_new_season_has_no_season_form_yet(self):
        df = make_synthetic_races_df()
        dataset = build_training_dataset(df)
        row = dataset[(dataset["race_id"] == "2023_1") & (dataset["driver"] == "VER")].iloc[0]
        assert math.isnan(row["season_avg_finish"])
        assert row["season_points"] == 0.0
        assert row["season_races_so_far"] == 0


class TestBuildTrainingDataset:
    def test_one_row_per_driver_per_race(self):
        df = make_synthetic_races_df()
        dataset = build_training_dataset(df)
        # 4 races x 2 drivers each = 8 rows
        assert len(dataset) == 8

    def test_target_finish_matches_source_finish_position(self):
        df = make_synthetic_races_df()
        dataset = build_training_dataset(df)
        row = dataset[(dataset["race_id"] == "2022_2") & (dataset["driver"] == "VER")].iloc[0]
        assert row["target_finish"] == 1  # VER finished P1 at 2022 Monza per fixture

    def test_feature_columns_excludes_identifiers_and_label(self):
        df = make_synthetic_races_df()
        dataset = build_training_dataset(df)
        feats = feature_columns(dataset)
        for excluded in ("race_id", "season", "round", "driver", "team", "circuit", "target_finish"):
            assert excluded not in feats
        assert "driver_form_3" in feats
