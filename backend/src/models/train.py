"""
Chronological dataset splitting, model training, and baseline comparison
(Sections 7, 8.1, 8.2 of the project guide).
"""
from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import get_logger
from src.evaluation.metrics import (
    brier_score_win,
    log_loss_win,
    mae,
    race_level_rank_correlation,
    topk_agreement,
)
from src.features.dataset import feature_columns
from src.models.baseline import (
    baseline_qualifying_predictor,
    baseline_recent_form_predictor,
    build_gradient_boosting_model,
    win_probabilities_from_predictions,
)

logger = get_logger(__name__)


def chronological_split(
    dataset: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train/val/test by RACE, in chronological
    (season, round) order — never a random row-level split (Section 7:
    'Use chronological data splits, not random train/test splitting').
    Splitting is done at the race level, not the row level, because every
    driver-row for a given race must land in the same split; a random
    row-level split could put one driver from a race in train and another
    driver from that SAME race in test, which doesn't reflect how the
    system is actually used (predicting an entire field at once) and
    would make the test set easier than reality.
    """
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1, with train_frac > 0")

    race_order = (
        dataset[["race_id", "season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )
    n = len(race_order)
    if n < 3:
        raise ValueError(
            f"Need at least 3 distinct races to form a train/val/test split; got {n}. "
            "This is expected on small synthetic fixtures - use a larger dataset."
        )

    n_train = max(1, round(n * train_frac))
    n_val = max(0, round(n * val_frac))
    n_train = min(n_train, n - 2)  # always leave >=1 race each for val (if requested) and test
    n_val = min(n_val, n - n_train - 1)

    train_ids = set(race_order["race_id"].iloc[:n_train])
    val_ids = set(race_order["race_id"].iloc[n_train:n_train + n_val])
    test_ids = set(race_order["race_id"].iloc[n_train + n_val:])

    train_df = dataset[dataset["race_id"].isin(train_ids)].reset_index(drop=True)
    val_df = dataset[dataset["race_id"].isin(val_ids)].reset_index(drop=True)
    test_df = dataset[dataset["race_id"].isin(test_ids)].reset_index(drop=True)

    logger.info(
        "Chronological split: %d train races / %d val races / %d test races "
        "(%d / %d / %d rows)",
        len(train_ids), len(val_ids), len(test_ids),
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def train_baseline_gbm(train_df: pd.DataFrame, feature_cols: list[str], label_col: str = "target_finish"):
    """Fit the baseline gradient-boosting model on the training fold only.
    No feature scaling/imputation is needed or performed here beyond the
    model's built-in NaN handling — see build_gradient_boosting_model's
    docstring for why that matters for leakage-free handling of cold-start
    features."""
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df[label_col].astype(float)
    model = build_gradient_boosting_model()
    model.fit(X_train, y_train)
    logger.info("Trained baseline GBM on %d rows, %d features", len(X_train), len(feature_cols))
    return model


def predict_with_model(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a copy of df with a `pred_finish` column added."""
    X = df[feature_cols].astype(float)
    out = df.copy()
    out["pred_finish"] = model.predict(X)
    return out


def evaluate_predictions(df: pd.DataFrame, pred_col: str = "pred_finish") -> dict:
    """Compute the Section 8.2 metric set for one set of predictions
    already attached to df as `pred_col`."""
    return {
        "mae": mae(df, pred_col=pred_col),
        "rank_correlation": race_level_rank_correlation(df, pred_col=pred_col),
        "top1_agreement": topk_agreement(df, k=1, pred_col=pred_col),
        "top3_agreement": topk_agreement(df, k=3, pred_col=pred_col),
        "n_rows": len(df),
    }


def compute_feature_importance(
    model, df: pd.DataFrame, feature_cols: list[str], label_col: str = "target_finish",
    n_repeats: int = 5, random_state: int = 42,
) -> pd.Series:
    """Permutation importance, sorted descending. HistGradientBoostingRegressor
    has no built-in `feature_importances_` (unlike a plain
    RandomForest/GradientBoostingRegressor), so this is how the CLI's
    "Top factors" list (Section 2's product vision) is actually derived
    from the data rather than hard-coded — a literal requirement from that
    section ("Do not hard-code this example output").

    Run this on a held-out set (validation or test fold), not the
    training set, so importance reflects generalization rather than
    memorization.
    """
    X = df[feature_cols].astype(float)
    y = df[label_col].astype(float)
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="neg_mean_absolute_error"
    )
    importance = pd.Series(result.importances_mean, index=feature_cols)
    return importance.sort_values(ascending=False)


def compare_against_baselines(
    dataset: pd.DataFrame,
    feature_cols: list[str] | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict:
    """Section 8.2's required comparison: train the GBM on the
    chronological training split, evaluate it plus the two simple
    heuristic baselines (qualifying position, recent form) all on the
    SAME held-out test split, and return a metrics dict per approach so
    they're directly comparable.

    Also attaches a heuristic win-probability layer and its calibration
    metrics for the GBM's predictions (Section 8.3) — explicitly labeled
    as unvalidated until this has been run against real historical data
    (see win_probabilities_from_predictions' docstring).
    """
    feature_cols = feature_cols or feature_columns(dataset)
    train_df, val_df, test_df = chronological_split(dataset, train_frac, val_frac)

    model = train_baseline_gbm(train_df, feature_cols)
    test_with_gbm = predict_with_model(model, test_df, feature_cols)

    test_with_quali = test_df.copy()
    test_with_quali["pred_finish"] = baseline_qualifying_predictor(test_df)

    test_with_form = test_df.copy()
    test_with_form["pred_finish"] = baseline_recent_form_predictor(test_df)

    results = {
        "gradient_boosting": evaluate_predictions(test_with_gbm),
        "baseline_qualifying_position": evaluate_predictions(test_with_quali),
        "baseline_recent_form": evaluate_predictions(test_with_form),
    }

    test_with_gbm["win_prob"] = win_probabilities_from_predictions(test_with_gbm, "pred_finish")
    results["gradient_boosting"]["brier_score_win"] = brier_score_win(test_with_gbm, "win_prob")
    results["gradient_boosting"]["log_loss_win"] = log_loss_win(test_with_gbm, "win_prob")

    logger.info("Baseline comparison results: %s", results)
    return results
