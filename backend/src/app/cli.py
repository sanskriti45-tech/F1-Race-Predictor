"""
CLI for the F1 race predictor (Section 2 product vision, Section 10
upcoming-race mode).

    python -m src.app.cli --next-race
    python -m src.app.cli --season 2026 --round 5

No dashboard here on purpose — Section 11 says build the dashboard only
after the data pipeline and backtest are trustworthy, and Section 18's
first prompt is explicit: "Do not add a dashboard yet."
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from src.app.labels import label_for
from src.config import get_logger, settings
from src.data.fastf1_client import get_upcoming_race, load_session
from src.models.predict import (
    ModelBundle,
    PredictionCutoffError,
    load_model_bundle,
    predict_upcoming_race,
)

logger = get_logger(__name__)


def _load_races_df() -> pd.DataFrame:
    path = settings.data_processed_dir / "races.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No processed historical races table found at {path}. Run "
            "scripts/build_dataset_and_train.py first (requires live FastF1 "
            "network access) to build it."
        )
    return pd.read_csv(path, parse_dates=["date"])


def _load_bundle() -> ModelBundle:
    path = settings.models_dir / "baseline_gbm.joblib"
    return load_model_bundle(path)  # raises FileNotFoundError with a clear message if missing


def entrants_from_qualifying(quali_results: pd.DataFrame) -> list[dict]:
    """Turn a FastF1 qualifying `results` frame into the entrant list
    predict_upcoming_race expects. grid_position falls back to
    qualifying_position — Section 5.1 notes grid penalties should be
    applied "if the required information is available"; this MVP has no
    penalty data source wired up yet, so it documents that gap here
    rather than pretending to apply penalties it doesn't have."""
    entrants = []
    for _, r in quali_results.iterrows():
        entrants.append({
            "driver": r["Abbreviation"],
            "team": r["TeamName"],
            "qualifying_position": r.get("Position"),
            "grid_position": r.get("Position"),
        })
    return entrants


def _print_report(predictions: pd.DataFrame, event: str, cutoff_note: str, importance) -> None:
    print(f"NEXT RACE: {event}")
    print(f"Prediction generated from data available at {cutoff_note}\n")
    print("PREDICTED FINISH")
    for _, row in predictions.iterrows():
        print(
            f"{int(row['predicted_rank'])}. {row['driver']:6s} — "
            f"P1 probability: {row['win_prob'] * 100:.0f}%"
        )

    availability_cols = [c for c in predictions.columns if c.endswith("_available")]
    if availability_cols:
        n_missing = int((~predictions[availability_cols]).sum().sum())
        if n_missing > 0:
            print(
                f"\nNote: {n_missing} rolling-form feature value(s) were cold-start "
                "(insufficient history) for one or more drivers."
            )

    if importance is not None and len(importance) > 0:
        print("\nTop factors:")
        for feat in importance.head(5).index:
            print(f"- {label_for(feat)}")

    print("\nNo future race-result data was used.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="F1 race finishing-position and win-probability predictor"
    )
    parser.add_argument(
        "--next-race", action="store_true", help="Predict the next upcoming race on the calendar"
    )
    parser.add_argument("--season", type=int, help="Explicit season, use with --round")
    parser.add_argument("--round", type=int, dest="round_number", help="Explicit round number, use with --season")
    args = parser.parse_args(argv)

    if not args.next_race and (args.season is None or args.round_number is None):
        parser.error("Specify either --next-race or both --season and --round")

    try:
        races_df = _load_races_df()
        bundle = _load_bundle()
    except FileNotFoundError as exc:
        print(f"Cannot generate a prediction: {exc}", file=sys.stderr)
        return 1

    try:
        if args.next_race:
            info = get_upcoming_race()
            season, round_number, event, circuit = info.season, info.round, info.event, info.circuit
            qualifying_completed = info.qualifying_completed
            cutoff_note = f"{info.retrieved_at.isoformat()} (after qualifying, before race start)"
        else:
            season, round_number = args.season, args.round_number
            quali_result = load_session(season, round_number, "Q")
            if quali_result.session is not None:
                event = str(quali_result.session.event.get("EventName", f"round {round_number}"))
                circuit = str(quali_result.session.event.get("Location", event))
            else:
                event = circuit = f"round {round_number}"
            qualifying_completed = quali_result.ok
            cutoff_note = f"{datetime.now(timezone.utc).isoformat()} (explicit season/round request)"

        if not qualifying_completed:
            print(
                f"Qualifying has not completed (or could not be loaded) for {season} round "
                f"{round_number} ({event}) — refusing to fabricate a qualifying-based prediction.",
                file=sys.stderr,
            )
            return 1

        quali_result = load_session(season, round_number, "Q")
        if not quali_result.ok or quali_result.results is None or len(quali_result.results) == 0:
            print(
                f"Could not load usable qualifying results for {season} round {round_number}: "
                f"{quali_result.error or quali_result.warnings}",
                file=sys.stderr,
            )
            return 1

        entrants = entrants_from_qualifying(quali_result.results)
        predictions = predict_upcoming_race(
            races_df, bundle,
            season=season, round_number=round_number, circuit=circuit,
            entrants=entrants, qualifying_completed=True,
        )
    except PredictionCutoffError as exc:
        print(f"Cannot generate a prediction: {exc}", file=sys.stderr)
        return 1

    _print_report(predictions, event, cutoff_note, bundle.feature_importance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
