"""
Walk-forward backtesting (Section 9 of the project guide).

Phase 3's chronological_split gives ONE train/val/test cut — useful for
quick iteration, but it only produces out-of-sample predictions for
whatever races land in the single test fold. A walk-forward backtest
instead advances race-by-race (or in periodic retrain blocks) through the
*entire* dataset after some minimum training history, producing an
out-of-sample prediction for almost every historical race and a much
larger, more trustworthy sample of held-out performance.

Leakage note: this module does NOT need to re-derive its own cutoff
logic. Every row in `dataset` (the output of
src.features.dataset.build_training_dataset) already has its features
computed strictly before that row's own race — see src/features/cutoff.py.
So "train on races before race i" here just means "take the dataset rows
for those race_ids", and correctness falls straight out of Phase 2's
guarantees. What this module is responsible for getting right is not
re-introducing leakage by, e.g., predicting a race using a model trained
on a training set that includes that race or a later one — the tests in
tests/test_backtest.py check exactly that.
"""
from __future__ import annotations

import pandas as pd

from src.config import get_logger
from src.evaluation.metrics import (
    brier_score_win,
    log_loss_win,
    mae,
    race_level_rank_correlation,
    topk_agreement,
)
from src.features.dataset import feature_columns
from src.models.baseline import win_probabilities_from_predictions
from src.models.train import predict_with_model, train_baseline_gbm

logger = get_logger(__name__)


def _race_order(dataset: pd.DataFrame) -> pd.DataFrame:
    return (
        dataset[["race_id", "season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )


def walk_forward_backtest(
    dataset: pd.DataFrame,
    feature_cols: list[str] | None = None,
    min_train_races: int = 20,
    retrain_every: int = 1,
) -> pd.DataFrame:
    """Produce out-of-sample predictions for every race after the first
    `min_train_races`, retraining the model every `retrain_every` races
    (retrain_every=1 retrains before every single race — the most
    faithful simulation of "retrain right before each new race weekend";
    larger values trade fidelity for speed on large datasets).

    Returns a DataFrame with one row per (race_id, driver) covering every
    backtested race, with `pred_finish` and `win_prob` columns attached,
    plus everything else from `dataset` (including `target_finish`).
    """
    feature_cols = feature_cols or feature_columns(dataset)
    race_order = _race_order(dataset)
    n = len(race_order)

    if n <= min_train_races:
        raise ValueError(
            f"Dataset has only {n} distinct races, need more than "
            f"min_train_races={min_train_races} to backtest anything."
        )

    predictions = []
    model = None
    for i in range(min_train_races, n):
        if model is None or (i - min_train_races) % retrain_every == 0:
            train_race_ids = set(race_order["race_id"].iloc[:i])
            train_df = dataset[dataset["race_id"].isin(train_race_ids)]
            model = train_baseline_gbm(train_df, feature_cols)
            logger.info(
                "Backtest: retrained before race %s (%d/%d), trained on %d prior races",
                race_order["race_id"].iloc[i], i + 1, n, len(train_race_ids),
            )

        target_race_id = race_order["race_id"].iloc[i]
        target_df = dataset[dataset["race_id"] == target_race_id]
        predicted = predict_with_model(model, target_df, feature_cols)
        predicted["win_prob"] = win_probabilities_from_predictions(predicted, "pred_finish")
        predictions.append(predicted)

    result = pd.concat(predictions, ignore_index=True)
    logger.info(
        "Walk-forward backtest complete: %d out-of-sample races, %d rows",
        n - min_train_races, len(result),
    )
    return result


def backtest_summary(predictions_df: pd.DataFrame) -> dict:
    """Aggregate metrics overall AND per season (Section 9: 'report
    metrics per season and in aggregate', so a single pooled number can't
    hide a season where the model quietly fell apart, e.g. after a
    regulation change)."""

    def _metrics_for(df: pd.DataFrame) -> dict:
        m = {
            "mae": mae(df),
            "rank_correlation": race_level_rank_correlation(df),
            "top1_agreement": topk_agreement(df, k=1),
            "top3_agreement": topk_agreement(df, k=3),
            "brier_score_win": brier_score_win(df, "win_prob"),
            "log_loss_win": log_loss_win(df, "win_prob"),
            "n_races": df["race_id"].nunique(),
            "n_rows": len(df),
        }
        return m

    summary = {"overall": _metrics_for(predictions_df)}
    for season, season_df in predictions_df.groupby("season"):
        summary[f"season_{season}"] = _metrics_for(season_df)
    return summary


def worst_predictions(predictions_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The individual driver-race predictions with the largest absolute
    error — Section 9's requirement to investigate failure cases, not
    just report aggregate metrics. Look at these rows and ask why: DNF?
    A rookie's first race? A surprise strategy call the model has no
    feature for? That's the actual diagnostic work; this function just
    surfaces the candidates."""
    df = predictions_df.copy()
    df["abs_error"] = (df["target_finish"] - df["pred_finish"]).abs()
    return df.nlargest(n, "abs_error")[
        ["race_id", "season", "round", "driver", "team", "target_finish", "pred_finish", "abs_error"]
    ].reset_index(drop=True)


def worst_races(predictions_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Races (not individual drivers) with the highest mean absolute
    error across their whole field — surfaces races the model struggled
    with broadly (e.g. a chaotic weather race), as distinct from one
    driver having a fluke result in an otherwise well-predicted race."""
    df = predictions_df.copy()
    df["abs_error"] = (df["target_finish"] - df["pred_finish"]).abs()
    by_race = (
        df.groupby(["race_id", "season", "round"])["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "mean_abs_error"})
    )
    return by_race.nlargest(n, "mean_abs_error").reset_index(drop=True)
