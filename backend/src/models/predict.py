"""
Upcoming-race prediction mode (Section 10 of the project guide).

This is the module that makes the CLI's `--next-race` command work:
given a trained model and the historical races table, build features for
a race that HASN'T happened yet (or hasn't been scored yet) and produce a
predicted finishing order plus win probabilities.

The critical constraint carried over from every earlier phase: an
upcoming race's features are built with EXACTLY the same
`races_before(...)` cutoff logic as every historical training row (via
src.features.dataset.build_feature_row) — there is no separate,
possibly-inconsistent code path for "live" predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.config import get_logger
from src.features.cutoff import races_before
from src.features.dataset import build_feature_row
from src.models.baseline import win_probabilities_from_predictions

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    """Everything Section 8.1 requires saving: 'the exact model
    configuration and feature schema' — plus, for the CLI's 'Top factors'
    output, whatever feature-importance ranking was computed at save time
    (see src.models.train.compute_feature_importance). `saved_at` and
    `model_params` make the artifact self-describing rather than a bare
    pickle nobody can audit later."""

    model: object
    feature_cols: list[str]
    feature_importance: Optional[pd.Series]
    model_params: dict
    saved_at: datetime


def save_model_bundle(
    model, feature_cols: list[str], path: str | Path, feature_importance: Optional[pd.Series] = None
) -> None:
    bundle = ModelBundle(
        model=model,
        feature_cols=list(feature_cols),
        feature_importance=feature_importance,
        model_params=model.get_params(),
        saved_at=datetime.now(timezone.utc),
    )
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path)
    logger.info("Saved model bundle to %s (%d features)", save_path, len(feature_cols))


def load_model_bundle(path: str | Path) -> ModelBundle:
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"No saved model at {load_path}")
    bundle: ModelBundle = joblib.load(load_path)
    logger.info(
        "Loaded model bundle from %s (saved_at=%s, %d features)",
        load_path, bundle.saved_at.isoformat(), len(bundle.feature_cols),
    )
    return bundle


class PredictionCutoffError(Exception):
    """Raised when a prediction cannot honestly be made — e.g. qualifying
    hasn't happened yet but the model needs qualifying_position. Section
    10: 'If qualifying has not occurred, either refuse a qualifying-based
    prediction or use a separate pre-qualifying model.' This MVP takes the
    'refuse' branch and says so clearly rather than guessing a qualifying
    position."""


def predict_upcoming_race(
    races_df: pd.DataFrame,
    bundle: ModelBundle,
    *,
    season: int,
    round_number: int,
    circuit: str,
    entrants: list[dict],
    qualifying_completed: bool,
) -> pd.DataFrame:
    """Predict finishing order + win probabilities for an upcoming race.

    `entrants` is a list of dicts, one per driver on the grid, each with
    at minimum {"driver": ..., "team": ..., "qualifying_position": ...,
    "grid_position": ...}. Qualifying/grid position are allowed to be
    None ONLY if `qualifying_completed=False`; this function then
    refuses to predict rather than silently treating "unknown" as
    "average" (Section 10's 'report missing data, don't fabricate').

    Every feature is built via races_before(races_df, season, round)
    exactly as in training — see build_feature_row's docstring for why
    that matters.
    """
    if not qualifying_completed:
        raise PredictionCutoffError(
            f"Qualifying has not completed for {season} round {round_number} ({circuit}). "
            "This model requires qualifying_position as an input feature (Section 5.1), so "
            "a pre-qualifying prediction is refused rather than fabricated. A separate "
            "pre-qualifying model (Section 16, optional extension) would be needed for that."
        )

    if not entrants:
        raise PredictionCutoffError(
            f"No entrant list provided for {season} round {round_number} ({circuit})."
        )

    missing_quali = [e["driver"] for e in entrants if e.get("qualifying_position") is None]
    if missing_quali:
        raise PredictionCutoffError(
            f"Missing qualifying_position for: {missing_quali}. Refusing to predict rather "
            "than fabricating a value for these drivers."
        )

    history = races_before(races_df, season, round_number)
    if history.empty:
        logger.warning(
            "No historical data before %s round %s — every driver/team/circuit feature "
            "will be cold-start NaN. Predictions will rely entirely on qualifying_position.",
            season, round_number,
        )

    rows = []
    for e in entrants:
        row = build_feature_row(
            history,
            driver=e["driver"],
            team=e["team"],
            circuit=circuit,
            season=season,
            qualifying_position=e["qualifying_position"],
            grid_position=e.get("grid_position", e["qualifying_position"]),
        )
        rows.append(row)

    features_df = pd.DataFrame(rows)
    X = features_df[bundle.feature_cols].astype(float)
    features_df["pred_finish"] = bundle.model.predict(X)
    features_df["race_id_tmp"] = f"{season}_{round_number}"
    features_df["win_prob"] = win_probabilities_from_predictions(
        features_df, "pred_finish", race_id_col="race_id_tmp"
    )
    features_df = features_df.drop(columns=["race_id_tmp"])

    result = features_df.sort_values("pred_finish").reset_index(drop=True)
    result.insert(0, "predicted_rank", range(1, len(result) + 1))

    logger.info(
        "Generated prediction for %s round %s (%s): %d drivers",
        season, round_number, circuit, len(result),
    )
    return result
