"""
End-to-end pipeline: fetch historical seasons via FastF1 -> normalize into
the races table -> build the leakage-safe training dataset -> train the
baseline GBM -> save the races table and model bundle for the CLI to use.

This is the script that actually needs live network access to FastF1's
backends (see README's network-access note) — run it yourself in an
unrestricted environment:

    python scripts/build_dataset_and_train.py --seasons 2022 2023

Everything it calls (fastf1_client, loaders, dataset, train, predict) is
already unit-tested against synthetic/mocked data; this script's own job
is just orchestration, so it's kept deliberately thin.
"""
from __future__ import annotations

import argparse
import sys

import fastf1

from src.config import get_logger, settings
from src.data.fastf1_client import load_race_and_qualifying
from src.data.loaders import build_race_rows, combine_races
from src.data.validation import validate_session_result
from src.features.dataset import build_training_dataset, feature_columns
from src.models.predict import save_model_bundle
from src.models.train import chronological_split, compute_feature_importance, train_baseline_gbm

logger = get_logger(__name__)


def fetch_season(season: int) -> list:
    """Fetch every completed race + qualifying pair for one season and
    normalize each into a per-race row frame. Skips (with a logged
    warning, not a crash) any event that fails validation — e.g. a race
    that hasn't happened yet this season, or one FastF1 can't fully load."""
    settings.ensure_dirs()
    try:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not fetch %s schedule: %s", season, exc)
        return []

    race_frames = []
    for _, event in schedule.iterrows():
        round_number = int(event["RoundNumber"])
        if round_number == 0:  # testing events sometimes slip through
            continue
        race_result, quali_result = load_race_and_qualifying(season, round_number)

        race_report = validate_session_result(race_result)
        if not race_report.ok:
            logger.warning(
                "Skipping %s round %s (%s): %s", season, round_number, event["EventName"], race_report.issues
            )
            continue

        try:
            rows = build_race_rows(
                race_result.results, quali_result.results,
                season=season, round_number=round_number,
                event=str(event["EventName"]), circuit=str(event.get("Location", event["EventName"])),
                date=event["EventDate"],
            )
            race_frames.append(rows)
        except ValueError as exc:
            logger.warning("Skipping %s round %s: %s", season, round_number, exc)

    return race_frames


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the F1 dataset and train the baseline model")
    parser.add_argument("--seasons", type=int, nargs="+", required=True, help="Seasons to fetch, e.g. 2022 2023")
    args = parser.parse_args(argv)

    all_race_frames = []
    for season in args.seasons:
        logger.info("Fetching season %s...", season)
        all_race_frames.extend(fetch_season(season))

    if not all_race_frames:
        print(
            "No usable race data was fetched for any requested season. This usually means "
            "either the seasons are invalid, or the environment's network can't reach "
            "FastF1's backends (see README).",
            file=sys.stderr,
        )
        return 1

    races_df = combine_races(all_race_frames)
    settings.ensure_dirs()
    races_path = settings.data_processed_dir / "races.csv"
    races_df.to_csv(races_path, index=False)
    logger.info("Saved %d rows to %s", len(races_df), races_path)

    dataset = build_training_dataset(races_df)
    feats = feature_columns(dataset)
    train_df, val_df, _test_df = chronological_split(dataset)
    model = train_baseline_gbm(train_df, feats)

    importance = (
        compute_feature_importance(model, val_df, feats) if len(val_df) > 0
        else compute_feature_importance(model, train_df, feats)
    )

    model_path = settings.models_dir / "baseline_gbm.joblib"
    save_model_bundle(model, feats, model_path, feature_importance=importance)

    print(f"Done. {len(races_df)} historical driver-race rows saved to {races_path}")
    print(f"Model saved to {model_path}")
    print("\nTop features by permutation importance:")
    print(importance.head(10).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
