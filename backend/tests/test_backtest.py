from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from src.evaluation.backtest import (
    backtest_summary,
    walk_forward_backtest,
    worst_predictions,
    worst_races,
)
from src.features.dataset import build_training_dataset
from src.models.train import train_baseline_gbm
from tests.fixtures import make_large_synthetic_races_df

N_RACES = 24
MIN_TRAIN = 10


@pytest.fixture(scope="module")
def dataset():
    races_df = make_large_synthetic_races_df(n_races=N_RACES, seed=11)
    return build_training_dataset(races_df)


@pytest.fixture(scope="module")
def predictions(dataset):
    return walk_forward_backtest(dataset, min_train_races=MIN_TRAIN, retrain_every=1)


class TestWalkForwardBacktest:
    def test_covers_exactly_the_expected_number_of_races(self, predictions):
        assert predictions["race_id"].nunique() == N_RACES - MIN_TRAIN

    def test_every_backtested_race_has_all_drivers(self, predictions, dataset):
        n_drivers_per_race = dataset.groupby("race_id").size().iloc[0]
        rows_per_race = predictions.groupby("race_id").size()
        assert (rows_per_race == n_drivers_per_race).all()

    def test_predictions_and_win_probs_present_and_finite(self, predictions):
        assert predictions["pred_finish"].notna().all()
        assert predictions["win_prob"].between(0, 1).all()

    def test_too_few_races_raises(self, dataset):
        with pytest.raises(ValueError):
            walk_forward_backtest(dataset, min_train_races=N_RACES)  # nothing left to test on

    def test_retrain_every_controls_how_often_the_model_is_refit(self, dataset):
        with patch(
            "src.evaluation.backtest.train_baseline_gbm", wraps=train_baseline_gbm
        ) as mock_train:
            walk_forward_backtest(dataset, min_train_races=MIN_TRAIN, retrain_every=5)
        n_out_of_sample = N_RACES - MIN_TRAIN
        expected_retrains = math.ceil(n_out_of_sample / 5)
        assert mock_train.call_count == expected_retrains

    def test_retrain_every_one_retrains_before_every_race(self, dataset):
        with patch(
            "src.evaluation.backtest.train_baseline_gbm", wraps=train_baseline_gbm
        ) as mock_train:
            walk_forward_backtest(dataset, min_train_races=MIN_TRAIN, retrain_every=1)
        assert mock_train.call_count == N_RACES - MIN_TRAIN


class TestBacktestCannotLeakFuture:
    def test_corrupting_a_late_race_does_not_change_an_earlier_backtested_prediction(self, dataset):
        """The most direct leakage check for the backtest loop itself:
        predictions for an early out-of-sample race must come from a
        model trained only on races strictly before it, so corrupting a
        much later race's results must not change that early prediction
        at all.
        """
        SENTINEL = -777777.0
        clean_predictions = walk_forward_backtest(dataset, min_train_races=MIN_TRAIN, retrain_every=1)

        race_order = (
            dataset[["race_id", "season", "round"]]
            .drop_duplicates()
            .sort_values(["season", "round"])
            .reset_index(drop=True)
        )
        late_race_id = race_order["race_id"].iloc[-1]  # last race chronologically
        early_race_id = race_order["race_id"].iloc[MIN_TRAIN]  # first backtested race

        corrupted_dataset = dataset.copy()
        mask = corrupted_dataset["race_id"] == late_race_id
        corrupted_dataset.loc[mask, "target_finish"] = SENTINEL

        corrupted_predictions = walk_forward_backtest(
            corrupted_dataset, min_train_races=MIN_TRAIN, retrain_every=1
        )

        clean_early = clean_predictions[clean_predictions["race_id"] == early_race_id][
            "pred_finish"
        ].reset_index(drop=True)
        corrupted_early = corrupted_predictions[
            corrupted_predictions["race_id"] == early_race_id
        ]["pred_finish"].reset_index(drop=True)

        pd_testing_equal = clean_early.equals(corrupted_early)
        assert pd_testing_equal, "Corrupting a late race changed an earlier backtested prediction."

    def test_target_race_is_never_in_its_own_training_set(self, dataset):
        """Different failure mode than the test above: this catches a
        model being trained on the very race it's about to predict (an
        off-by-one in the slice bound), which corrupting a late race and
        checking an early race's predictions would NOT catch, since
        self-inclusion doesn't propagate to other races' predictions.
        Verified by deliberately breaking iloc[:i] into iloc[:i+1] in this
        module and confirming this test fails while the test above does
        not — that off-by-one is a real, distinct bug class.
        """
        seen_train_race_ids = []
        seen_target_race_ids = []

        original_train = train_baseline_gbm

        def _capturing_train(train_df, feature_cols, **kwargs):
            seen_train_race_ids.append(set(train_df["race_id"]))
            return original_train(train_df, feature_cols, **kwargs)

        with patch("src.evaluation.backtest.train_baseline_gbm", side_effect=_capturing_train):
            preds = walk_forward_backtest(dataset, min_train_races=MIN_TRAIN, retrain_every=1)

        race_order = (
            dataset[["race_id", "season", "round"]]
            .drop_duplicates()
            .sort_values(["season", "round"])
            .reset_index(drop=True)
        )
        target_race_ids = race_order["race_id"].iloc[MIN_TRAIN:].tolist()

        assert len(seen_train_race_ids) == len(target_race_ids)
        for train_ids, target_id in zip(seen_train_race_ids, target_race_ids):
            assert target_id not in train_ids, (
                f"Race {target_id} was present in its own training set."
            )


class TestBacktestSummary:
    def test_has_overall_and_per_season_keys(self, predictions):
        summary = backtest_summary(predictions)
        assert "overall" in summary
        seasons_in_data = sorted(predictions["season"].unique())
        for s in seasons_in_data:
            assert f"season_{s}" in summary

    def test_overall_metrics_are_numeric_and_sane(self, predictions):
        summary = backtest_summary(predictions)
        overall = summary["overall"]
        assert overall["mae"] >= 0
        assert overall["n_races"] == N_RACES - MIN_TRAIN
        assert overall["n_rows"] == len(predictions)


class TestFailureCaseInspection:
    def test_worst_predictions_sorted_descending_by_abs_error(self, predictions):
        worst = worst_predictions(predictions, n=5)
        errors = worst["abs_error"].tolist()
        assert errors == sorted(errors, reverse=True)
        assert len(worst) == 5

    def test_worst_predictions_abs_error_matches_manual_computation(self, predictions):
        worst = worst_predictions(predictions, n=1)
        row = worst.iloc[0]
        assert math.isclose(row["abs_error"], abs(row["target_finish"] - row["pred_finish"]))

    def test_worst_races_aggregates_by_race_not_by_row(self, predictions):
        worst = worst_races(predictions, n=3)
        assert len(worst) == 3
        assert "mean_abs_error" in worst.columns
        # each race_id should appear at most once
        assert worst["race_id"].is_unique
