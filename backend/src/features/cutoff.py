"""
The ONE place that decides "what races happened before this one".

Every feature builder in src/features/ must go through `races_before` to
get its input data. Centralizing this is deliberate: Section 6 of the
project guide calls leakage prevention "non-negotiable", and the easiest
way to make that true in practice is to make it impossible to accidentally
bypass — there is exactly one function that knows how to compare
chronological position, and every rolling-stat function is required to
call it rather than re-implement its own filtering.

Ordering key: (season, round). Never raw calendar `date` — see the
docstring in src/data/loaders.py for why.
"""
from __future__ import annotations

import pandas as pd


def races_before(races_df: pd.DataFrame, season: int, round_number: int) -> pd.DataFrame:
    """Return only the rows of races_df strictly chronologically before
    (season, round_number). Excludes the target race itself entirely —
    including its qualifying_position and grid_position — because those
    are only knowable in full for the target race at exactly the cutoff
    this function's caller decides, and rolling "form" features must never
    include the very race they're trying to predict.
    """
    is_earlier_season = races_df["season"] < season
    is_same_season_earlier_round = (races_df["season"] == season) & (races_df["round"] < round_number)
    return races_df[is_earlier_season | is_same_season_earlier_round]


def races_before_race_id(races_df: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Convenience wrapper keyed by race_id (must be present exactly once
    in races_df, per the f"{season}_{round}" convention from loaders.py)."""
    target_rows = races_df[races_df["race_id"] == race_id]
    if target_rows.empty:
        raise KeyError(f"race_id {race_id!r} not found in races_df")
    season = int(target_rows["season"].iloc[0])
    round_number = int(target_rows["round"].iloc[0])
    return races_before(races_df, season, round_number)
