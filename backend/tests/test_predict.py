from __future__ import annotations

import math

import pytest

from src.features.cutoff import races_before
from src.features.dataset import build_feature_row, build_training_dataset, feature_columns
from src.models.predict import (
    ModelBundle,
    PredictionCutoffError,
    load_model_bundle,
    predict_upcoming_race,
    save_model_bundle,
)
from src.models.train import train_baseline_gbm
from tests.fixtures import make_large_synthetic_races_df


@pytest.fixture(scope="module")
def races_and_dataset():
    races_df = make_large_synthetic_races_df(n_races=24, seed=3)
    dataset = build_training_dataset(races_df)
    return races_df, dataset


@pytest.fixture(scope="module")
def trained_bundle(races_and_dataset):
    _, dataset = races_and_dataset
    feats = feature_columns(dataset)
    model = train_baseline_gbm(dataset, feats)  # train on everything for these prediction tests
    return ModelBundle(
        model=model, feature_cols=feats, feature_importance=None,
        model_params=model.get_params(), saved_at=None,
    )


def _entrants_for_race(races_df, season, round_number):
    race_rows = races_df[(races_df["season"] == season) & (races_df["round"] == round_number)]
    return [
        {
            "driver": r["driver"],
            "team": r["team"],
            "qualifying_position": r["qualifying_position"],
            "grid_position": r["grid_position"],
        }
        for _, r in race_rows.iterrows()
    ], race_rows["circuit"].iloc[0]


class TestPredictUpcomingRaceConsistency:
    def test_matches_training_row_feature_construction(self, races_and_dataset, trained_bundle):
        """The strongest possible check: predicting a HISTORICAL race
        through the 'upcoming race' code path must produce byte-identical
        input features to what build_training_dataset computed for that
        same race — since both ultimately call build_feature_row with the
        same history/driver/team/circuit/qualifying inputs. If this ever
        drifts, live predictions would silently stop matching what the
        model was actually trained on.
        """
        races_df, dataset = races_and_dataset
        season, round_number = 2023, 6
        entrants, circuit = _entrants_for_race(races_df, season, round_number)

        predicted = predict_upcoming_race(
            races_df, trained_bundle,
            season=season, round_number=round_number, circuit=circuit,
            entrants=entrants, qualifying_completed=True,
        )

        training_rows = dataset[(dataset["season"] == season) & (dataset["round"] == round_number)]

        for _, pred_row in predicted.iterrows():
            train_row = training_rows[training_rows["driver"] == pred_row["driver"]].iloc[0]
            for feat in trained_bundle.feature_cols:
                a, b = pred_row[feat], train_row[feat]
                if pd_isna(a) and pd_isna(b):
                    continue
                assert math.isclose(float(a), float(b), rel_tol=1e-9), (
                    f"Feature {feat} mismatch for {pred_row['driver']}: {a} vs {b}"
                )


def pd_isna(x) -> bool:
    import pandas as pd
    return pd.isna(x)


class TestPredictUpcomingRaceValidation:
    def test_refuses_when_qualifying_not_completed(self, races_and_dataset, trained_bundle):
        races_df, _ = races_and_dataset
        with pytest.raises(PredictionCutoffError):
            predict_upcoming_race(
                races_df, trained_bundle,
                season=2023, round_number=6, circuit="C1",
                entrants=[{"driver": "D1", "team": "TeamA", "qualifying_position": 1}],
                qualifying_completed=False,
            )

    def test_refuses_on_empty_entrants(self, races_and_dataset, trained_bundle):
        races_df, _ = races_and_dataset
        with pytest.raises(PredictionCutoffError):
            predict_upcoming_race(
                races_df, trained_bundle,
                season=2023, round_number=6, circuit="C1",
                entrants=[], qualifying_completed=True,
            )

    def test_refuses_on_missing_qualifying_position(self, races_and_dataset, trained_bundle):
        races_df, _ = races_and_dataset
        with pytest.raises(PredictionCutoffError):
            predict_upcoming_race(
                races_df, trained_bundle,
                season=2023, round_number=6, circuit="C1",
                entrants=[
                    {"driver": "D1", "team": "TeamA", "qualifying_position": 1},
                    {"driver": "D2", "team": "TeamA", "qualifying_position": None},
                ],
                qualifying_completed=True,
            )


class TestPredictUpcomingRaceOutput:
    def test_predicted_rank_matches_ascending_pred_finish(self, races_and_dataset, trained_bundle):
        races_df, _ = races_and_dataset
        entrants, circuit = _entrants_for_race(races_df, 2023, 6)
        predicted = predict_upcoming_race(
            races_df, trained_bundle,
            season=2023, round_number=6, circuit=circuit,
            entrants=entrants, qualifying_completed=True,
        )
        assert list(predicted["predicted_rank"]) == list(range(1, len(predicted) + 1))
        assert predicted["pred_finish"].is_monotonic_increasing

    def test_win_probabilities_sum_to_one(self, races_and_dataset, trained_bundle):
        races_df, _ = races_and_dataset
        entrants, circuit = _entrants_for_race(races_df, 2023, 6)
        predicted = predict_upcoming_race(
            races_df, trained_bundle,
            season=2023, round_number=6, circuit=circuit,
            entrants=entrants, qualifying_completed=True,
        )
        assert math.isclose(predicted["win_prob"].sum(), 1.0, abs_tol=1e-6)

    def test_cold_start_race_with_no_history_still_predicts(self, races_and_dataset, trained_bundle):
        # season/round with no prior races at all in the dataset
        races_df, _ = races_and_dataset
        predicted = predict_upcoming_race(
            races_df, trained_bundle,
            season=1999, round_number=1, circuit="BrandNewCircuit",
            entrants=[
                {"driver": "NEW1", "team": "NewTeam", "qualifying_position": 1, "grid_position": 1},
                {"driver": "NEW2", "team": "NewTeam", "qualifying_position": 2, "grid_position": 2},
            ],
            qualifying_completed=True,
        )
        assert len(predicted) == 2
        assert predicted["pred_finish"].notna().all()


class TestModelPersistence:
    def test_save_and_load_round_trip_preserves_predictions(self, races_and_dataset, trained_bundle, tmp_path):
        races_df, _ = races_and_dataset
        entrants, circuit = _entrants_for_race(races_df, 2023, 6)

        before = predict_upcoming_race(
            races_df, trained_bundle,
            season=2023, round_number=6, circuit=circuit,
            entrants=entrants, qualifying_completed=True,
        )

        path = tmp_path / "model.joblib"
        save_model_bundle(trained_bundle.model, trained_bundle.feature_cols, path)
        loaded = load_model_bundle(path)

        after = predict_upcoming_race(
            races_df, loaded,
            season=2023, round_number=6, circuit=circuit,
            entrants=entrants, qualifying_completed=True,
        )
        assert (before["pred_finish"] == after["pred_finish"]).all()
        assert loaded.feature_cols == trained_bundle.feature_cols

    def test_load_missing_file_raises_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_bundle(tmp_path / "does_not_exist.joblib")
