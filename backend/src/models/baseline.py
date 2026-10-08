"""
Baseline model + simple comparison baselines (Section 8.1, 8.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def build_gradient_boosting_model(random_state: int = 42, **overrides) -> HistGradientBoostingRegressor:
    """Section 8.1: 'Start with a simple, reproducible configuration before
    hyperparameter tuning.' HistGradientBoostingRegressor is chosen over a
    library like LightGBM/XGBoost specifically because it natively handles
    NaN feature values (our cold-start features, e.g. driver_form_5 for a
    rookie's first few races, are legitimately NaN — see Section 5.6) with
    no separate imputation step, which would otherwise be its own source
    of train/test leakage if not fit strictly on the training fold.
    """
    defaults = dict(
        max_iter=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_state,
    )
    defaults.update(overrides)
    return HistGradientBoostingRegressor(**defaults)


def baseline_qualifying_predictor(df: pd.DataFrame, quali_col: str = "qualifying_position") -> pd.Series:
    """Simplest possible baseline: predict the driver finishes where they
    qualified. Section 8.2 requires comparing the trained model against
    baselines like this one — a model that can't beat "just use qualifying
    position" isn't adding value yet."""
    return df[quali_col]


def baseline_recent_form_predictor(
    df: pd.DataFrame,
    form_col: str = "driver_form_3",
    fallback_col: str = "qualifying_position",
) -> pd.Series:
    """Predict a driver's recent average finishing position; fall back to
    qualifying position for cold-start rows where recent form is NaN
    (a driver's first few races), so the baseline itself doesn't need
    NaN-handling downstream."""
    return df[form_col].fillna(df[fallback_col])


def win_probabilities_from_predictions(
    df: pd.DataFrame, pred_col: str, race_id_col: str = "race_id"
) -> pd.Series:
    """Turn per-driver predicted finishing positions into per-race P1 win
    probabilities via a softmax over negative predicted position (a lower
    predicted position -> higher win probability), normalized within each
    race so probabilities sum to 1 across that race's field.

    This is a HEURISTIC probability layer, not a calibrated one. Section
    8.3 is explicit: do not label an arbitrary model score a "probability"
    without validating calibration on held-out data. Use
    src.evaluation.metrics.reliability_table / brier_score_win / log_loss_win
    on a held-out test set before trusting these numbers, and treat them as
    provisional until that's been done with real (not synthetic) historical
    data.
    """
    probs = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(race_id_col):
        scores = -group[pred_col].to_numpy(dtype=float)
        scores = scores - scores.max()  # numerical stability
        exp_scores = np.exp(scores)
        probs.loc[group.index] = exp_scores / exp_scores.sum()
    return probs
