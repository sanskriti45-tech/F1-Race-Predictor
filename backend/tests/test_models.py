from __future__ import annotations

import math

import pytest

from src.features.dataset import build_training_dataset, feature_columns
from src.models.baseline import (
    baseline_qualifying_predictor,
    baseline_recent_form_predictor,
    win_probabilities_from_predictions,
)
from src.models.train import (
    chronological_split,
    compare_against_baselines,
    evaluate_predictions,
    predict_with_model,
    train_baseline_gbm,
)
from tests.fixtures import make_large_synthetic_races_df, make_synthetic_races_df


@pytest.fixture(scope="module")
def large_dataset():
    races_df = make_large_synthetic_races_df(n_races=24, seed=7)
    return build_training_dataset(races_df)


class TestChronologicalSplit:
    def test_races_never_span_multiple_splits(self, large_dataset):
        train_df, val_df, test_df = chronological_split(large_dataset, train_frac=0.6, val_frac=0.2)
        train_ids = set(train_df["race_id"])
        val_ids = set(val_df["race_id"])
        test_ids = set(test_df["race_id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_splits_are_chronologically_ordered(self, large_dataset):
        train_df, val_df, test_df = chronological_split(large_dataset, train_frac=0.6, val_frac=0.2)
        max_train_key = train_df[["season", "round"]].apply(tuple, axis=1).max()
        min_val_key = val_df[["season", "round"]].apply(tuple, axis=1).min()
        max_val_key = val_df[["season", "round"]].apply(tuple, axis=1).max()
        min_test_key = test_df[["season", "round"]].apply(tuple, axis=1).min()
        assert max_train_key < min_val_key
        assert max_val_key < min_test_key

    def test_all_rows_accounted_for(self, large_dataset):
        train_df, val_df, test_df = chronological_split(large_dataset, train_frac=0.6, val_frac=0.2)
        assert len(train_df) + len(val_df) + len(test_df) == len(large_dataset)

    def test_too_few_races_raises(self):
        small_dataset = build_training_dataset(make_synthetic_races_df())  # only 4 races
        # train_frac=0.6, val_frac=0.2 on 4 races is fine (n>=3); force failure with n<3 races
        tiny = small_dataset[small_dataset["race_id"].isin(["2022_1", "2022_2"])]
        with pytest.raises(ValueError):
            chronological_split(tiny)

    def test_invalid_fractions_raise(self, large_dataset):
        with pytest.raises(ValueError):
            chronological_split(large_dataset, train_frac=0.7, val_frac=0.4)  # sums to >= 1


class TestTrainingAndPrediction:
    def test_model_trains_and_predicts_without_error(self, large_dataset):
        train_df, _, test_df = chronological_split(large_dataset)
        feats = feature_columns(large_dataset)
        model = train_baseline_gbm(train_df, feats)
        predicted = predict_with_model(model, test_df, feats)
        assert "pred_finish" in predicted.columns
        assert len(predicted) == len(test_df)
        assert predicted["pred_finish"].notna().all()

    def test_evaluate_predictions_returns_expected_keys(self, large_dataset):
        train_df, _, test_df = chronological_split(large_dataset)
        feats = feature_columns(large_dataset)
        model = train_baseline_gbm(train_df, feats)
        predicted = predict_with_model(model, test_df, feats)
        metrics = evaluate_predictions(predicted)
        for key in ("mae", "rank_correlation", "top1_agreement", "top3_agreement", "n_rows"):
            assert key in metrics
        assert metrics["mae"] >= 0
        assert metrics["n_rows"] == len(test_df)


class TestBaselineComparison:
    def test_returns_all_three_approaches(self, large_dataset):
        results = compare_against_baselines(large_dataset)
        assert set(results.keys()) == {
            "gradient_boosting", "baseline_qualifying_position", "baseline_recent_form",
        }

    def test_gbm_has_calibration_metrics_attached(self, large_dataset):
        results = compare_against_baselines(large_dataset)
        assert "brier_score_win" in results["gradient_boosting"]
        assert "log_loss_win" in results["gradient_boosting"]
        assert math.isfinite(results["gradient_boosting"]["brier_score_win"])

    def test_qualifying_baseline_matches_raw_qualifying_position(self, large_dataset):
        train_df, _, test_df = chronological_split(large_dataset)
        preds = baseline_qualifying_predictor(test_df)
        assert (preds == test_df["qualifying_position"]).all()

    def test_recent_form_baseline_has_no_nans(self, large_dataset):
        # fallback to qualifying_position should eliminate cold-start NaNs
        train_df, _, test_df = chronological_split(large_dataset)
        preds = baseline_recent_form_predictor(test_df)
        assert preds.notna().all()


class TestWinProbabilities:
    def test_probabilities_sum_to_one_per_race(self, large_dataset):
        train_df, _, test_df = chronological_split(large_dataset)
        feats = feature_columns(large_dataset)
        model = train_baseline_gbm(train_df, feats)
        predicted = predict_with_model(model, test_df, feats)
        probs = win_probabilities_from_predictions(predicted, "pred_finish")
        totals = probs.groupby(predicted["race_id"]).sum()
        assert all(math.isclose(t, 1.0, abs_tol=1e-6) for t in totals)

    def test_lower_predicted_finish_gets_higher_probability(self):
        import pandas as pd
        df = pd.DataFrame({
            "race_id": ["r1", "r1", "r1"],
            "pred_finish": [1.2, 3.5, 2.0],
        })
        probs = win_probabilities_from_predictions(df, "pred_finish")
        # lowest pred_finish (1.2) should have the highest win probability
        assert probs.iloc[0] > probs.iloc[2] > probs.iloc[1]
