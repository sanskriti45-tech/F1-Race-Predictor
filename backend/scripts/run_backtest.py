"""
Run the Phase 4 walk-forward backtest over the full processed dataset and
save the results to reports/, so the Phase 6 dashboard has something to
display without re-running the backtest on every page load.

Requires data/processed/races.csv to already exist (see
scripts/build_dataset_and_train.py).

    python scripts/run_backtest.py --min-train-races 10
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.config import get_logger, settings
from src.evaluation.backtest import (
    backtest_summary,
    walk_forward_backtest,
    worst_predictions,
    worst_races,
)
from src.features.dataset import build_training_dataset

logger = get_logger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the walk-forward backtest and save reports")
    parser.add_argument("--min-train-races", type=int, default=20)
    parser.add_argument("--retrain-every", type=int, default=1)
    args = parser.parse_args(argv)

    races_path = settings.data_processed_dir / "races.csv"
    if not races_path.exists():
        print(
            f"No processed races table at {races_path}. Run "
            "scripts/build_dataset_and_train.py first.",
            file=sys.stderr,
        )
        return 1

    races_df = pd.read_csv(races_path, parse_dates=["date"])
    dataset = build_training_dataset(races_df)

    predictions = walk_forward_backtest(
        dataset, min_train_races=args.min_train_races, retrain_every=args.retrain_every
    )
    summary = backtest_summary(predictions)
    worst_rows = worst_predictions(predictions, n=20)
    worst_race_rows = worst_races(predictions, n=10)

    settings.ensure_dirs()
    predictions.to_csv(settings.reports_dir / "backtest_predictions.csv", index=False)
    worst_rows.to_csv(settings.reports_dir / "worst_predictions.csv", index=False)
    worst_race_rows.to_csv(settings.reports_dir / "worst_races.csv", index=False)
    with open(settings.reports_dir / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"Backtest complete: {predictions['race_id'].nunique()} out-of-sample races.")
    print(json.dumps(summary["overall"], indent=2, default=float))
    print(f"\nReports saved to {settings.reports_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
