"""
Driver recent-form features (Section 5.2 of the project guide).

Every function here takes `history`, which MUST already be the output of
src.features.cutoff.races_before (or races_before_race_id) — i.e. it must
not contain the target race. These functions do not re-check the cutoff
themselves; that separation is intentional so cutoff logic lives in
exactly one place (see cutoff.py's docstring).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (3, 5)


def driver_form_features(history: pd.DataFrame, driver: str, windows=DEFAULT_WINDOWS) -> dict:
    """Rolling driver form over the last N races (N in `windows`), computed
    from `history` only. If the driver has fewer than N prior races, the
    average is computed over however many are actually available (a
    "growing window") and `driver_form_{w}_available` is set to False so
    the partial-window average is distinguishable downstream from a full
    one. True cold start (zero prior races) returns NaN, per Section 5.6's
    requirement for an explicit missing-value strategy rather than a
    silent default.
    """
    driver_races = (
        history[history["driver"] == driver]
        .sort_values(["season", "round"])
    )

    out: dict = {}
    for w in windows:
        recent = driver_races.tail(w)
        n = len(recent)
        out[f"driver_form_{w}"] = float(recent["finish_position"].mean()) if n > 0 else np.nan
        out[f"driver_quali_form_{w}"] = float(recent["qualifying_position"].mean()) if n > 0 else np.nan
        out[f"driver_points_{w}"] = float(recent["points"].sum()) if n > 0 else np.nan
        out[f"driver_finish_rate_{w}"] = (
            float(recent["classified_finish"].mean()) if n > 0 else np.nan
        )
        out[f"driver_form_{w}_available"] = n >= w

    out["driver_prior_starts"] = len(driver_races)
    return out
