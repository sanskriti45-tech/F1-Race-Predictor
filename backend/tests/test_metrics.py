from __future__ import annotations

import math

import pandas as pd

from src.evaluation.metrics import (
    brier_score_win,
    log_loss_win,
    mae,
    race_level_rank_correlation,
    reliability_table,
    topk_agreement,
)


def _race(race_id, true_positions, pred_positions):
    n = len(true_positions)
    return pd.DataFrame({
        "race_id": [race_id] * n,
        "target_finish": true_positions,
        "pred_finish": pred_positions,
    })


class TestMAE:
    def test_exact_value(self):
        df = _race("r1", [1, 2, 3, 4], [1, 3, 2, 4])
        # |1-1| + |2-3| + |3-2| + |4-4| = 0+1+1+0 = 2 / 4 = 0.5
        assert math.isclose(mae(df), 0.5)

    def test_zero_for_perfect_predictions(self):
        df = _race("r1", [1, 2, 3], [1, 2, 3])
        assert mae(df) == 0.0


class TestRankCorrelation:
    def test_perfect_agreement_is_one(self):
        df = _race("r1", [1, 2, 3, 4], [1, 2, 3, 4])
        assert math.isclose(race_level_rank_correlation(df), 1.0)

    def test_perfect_reversal_is_minus_one(self):
        df = _race("r1", [1, 2, 3, 4], [4, 3, 2, 1])
        assert math.isclose(race_level_rank_correlation(df), -1.0)

    def test_averages_across_multiple_races(self):
        perfect = _race("r1", [1, 2, 3], [1, 2, 3])
        reversed_ = _race("r2", [1, 2, 3], [3, 2, 1])
        df = pd.concat([perfect, reversed_], ignore_index=True)
        assert math.isclose(race_level_rank_correlation(df), 0.0, abs_tol=1e-9)

    def test_no_variance_race_is_excluded_not_zero(self):
        # a race with only one unique predicted value has undefined correlation
        no_variance = _race("r1", [1, 2, 3], [5, 5, 5])
        assert math.isnan(race_level_rank_correlation(no_variance))


class TestTopKAgreement:
    def test_top1_hit(self):
        df = _race("r1", [1, 2, 3], [1, 2, 3])  # predicted winner (lowest pred) is actual winner
        assert topk_agreement(df, k=1) == 1.0

    def test_top1_miss(self):
        df = _race("r1", [1, 2, 3], [3, 2, 1])  # predicted winner is actually last
        assert topk_agreement(df, k=1) == 0.0

    def test_top3_more_forgiving_than_top1(self):
        df = _race("r1", [1, 2, 3, 4], [3, 1, 2, 4])  # actual winner (true=1) predicted 3rd
        assert topk_agreement(df, k=1) == 0.0
        assert topk_agreement(df, k=3) == 1.0

    def test_averages_hit_rate_across_races(self):
        hit = _race("r1", [1, 2], [1, 2])
        miss = _race("r2", [1, 2], [2, 1])
        df = pd.concat([hit, miss], ignore_index=True)
        assert topk_agreement(df, k=1) == 0.5


class TestWinProbabilityMetrics:
    def test_brier_score_perfect_prediction_is_zero(self):
        df = pd.DataFrame({"target_finish": [1, 2, 3], "prob": [1.0, 0.0, 0.0]})
        assert brier_score_win(df, "prob") == 0.0

    def test_brier_score_worst_case_is_one(self):
        df = pd.DataFrame({"target_finish": [1, 2], "prob": [0.0, 1.0]})
        # driver who actually won (target_finish==1) was given prob 0 -> (0-1)^2=1
        # driver who didn't win was given prob 1 -> (1-0)^2=1 ; mean = 1.0
        assert brier_score_win(df, "prob") == 1.0

    def test_log_loss_is_finite_even_with_extreme_probs(self):
        df = pd.DataFrame({"target_finish": [1, 2], "prob": [0.0, 1.0]})
        loss = log_loss_win(df, "prob")
        assert math.isfinite(loss)
        assert loss > 0

    def test_reliability_table_has_expected_columns(self):
        df = pd.DataFrame({
            "target_finish": [1, 2, 3, 4, 1, 2, 3, 4],
            "prob": [0.4, 0.3, 0.2, 0.1, 0.5, 0.25, 0.15, 0.1],
        })
        table = reliability_table(df, "prob", n_bins=3)
        assert {"bucket", "n", "mean_predicted_prob", "actual_win_rate"} <= set(table.columns)
        assert table["n"].sum() == len(df)
