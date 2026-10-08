import pandas as pd
import traceback

from src.config import settings
from src.features.dataset import build_training_dataset, feature_columns
from src.models.train import train_baseline_gbm

races_df = pd.read_csv(
    settings.data_processed_dir / "races.csv",
    parse_dates=["date"],
)

dataset = build_training_dataset(races_df)
feature_cols = feature_columns(dataset)

race_order = (
    dataset[["race_id", "season", "round"]]
    .drop_duplicates()
    .sort_values(["season", "round"])
    .reset_index(drop=True)
)

print(f"Total races: {len(race_order)}")
print()

for i in range(20, len(race_order)):
    train_race_ids = set(race_order["race_id"].iloc[:i])
    train_df = dataset[dataset["race_id"].isin(train_race_ids)]

    race_id = race_order["race_id"].iloc[i]

    print(
        f"Testing fold {i + 1}/{len(race_order)}: "
        f"train_races={i}, "
        f"train_rows={len(train_df)}, "
        f"target_race={race_id}"
    )

    try:
        train_baseline_gbm(train_df, feature_cols)
    except Exception:
        print()
        print("!!! FIRST FAILING FOLD !!!")
        print(f"Fold: {i + 1}")
        print(f"Training races: {i}")
        print(f"Training rows: {len(train_df)}")
        print(f"Target race: {race_id}")
        traceback.print_exc()
        break
else:
    print()
    print("ALL BACKTEST TRAINING FOLDS SUCCEEDED")