"""
Team recent-form features (Section 5.3 of the project guide).

Same cutoff contract as driver_form.py: `history` must already be the
output of src.features.cutoff.races_before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.driver_form import DEFAULT_WINDOWS


def team_form_features(history: pd.DataFrame, team: str, windows=DEFAULT_WINDOWS) -> dict:
    """Rolling team form over the last N *races* (not N driver-entries —
    a team fields two cars per race, so we average both cars' results
    within each of the last N races rather than treating "last N rows" as
    "last N races", which would silently shrink the effective window).
    """
    team_races = history[history["team"] == team].sort_values(["season", "round"])
    race_ids_in_order = team_races["race_id"].drop_duplicates().tolist()

    out: dict = {}
    for w in windows:
        recent_race_ids = race_ids_in_order[-w:]
        recent = team_races[team_races["race_id"].isin(recent_race_ids)]
        n_races = len(recent_race_ids)
        out[f"team_form_{w}"] = float(recent["finish_position"].mean()) if n_races > 0 else np.nan
        out[f"team_quali_form_{w}"] = float(recent["qualifying_position"].mean()) if n_races > 0 else np.nan
        out[f"team_points_{w}"] = float(recent["points"].sum()) if n_races > 0 else np.nan
        out[f"team_form_{w}_available"] = n_races >= w

    out["team_prior_races"] = len(race_ids_in_order)
    return out
