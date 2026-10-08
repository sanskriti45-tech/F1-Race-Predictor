"""
Leakage-safe training dataset construction (Sections 5.5, 6, 7 of the
project guide).

build_training_dataset() is the single function that turns the unified
races table (src.data.loaders.combine_races output) into one row per
driver per race, with every feature computed strictly from
src.features.cutoff.races_before(season, round) — i.e. strictly before
that race. The target race's own qualifying_position and grid_position
ARE included as features (Section 5.1 — these are known before the race
itself starts), but its finish_position/points/status are used ONLY as
the label, never as an input feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_logger
from src.features.circuit_form import circuit_history_features
from src.features.cutoff import races_before
from src.features.driver_form import DEFAULT_WINDOWS, driver_form_features
from src.features.team_form import team_form_features

logger = get_logger(__name__)


def _season_form_features(history: pd.DataFrame, season: int, driver: str, team: str) -> dict:
    """Section 5.5: current-season form up to the cutoff. `history` is
    already filtered to strictly-before-target-race by the caller; here we
    further restrict to the same season, since season form is explicitly
    "current-season", not all-time."""
    season_history = history[history["season"] == season]
    driver_rows = season_history[season_history["driver"] == driver]
    return {
        "season_avg_finish": (
            float(driver_rows["finish_position"].mean()) if len(driver_rows) > 0 else np.nan
        ),
        "season_avg_quali": (
            float(driver_rows["qualifying_position"].mean()) if len(driver_rows) > 0 else np.nan
        ),
        "season_points": (
            float(driver_rows["points"].sum()) if len(driver_rows) > 0 else 0.0
        ),
        "season_races_so_far": int(len(driver_rows)),
    }


def build_feature_row(
    history: pd.DataFrame,
    *,
    driver: str,
    team: str,
    circuit: str,
    season: int,
    qualifying_position,
    grid_position,
    windows=DEFAULT_WINDOWS,
) -> dict:
    """Build the feature dict for ONE driver in ONE race, given `history`
    (already filtered to strictly-before-this-race via
    src.features.cutoff.races_before) and the race's known pre-race
    information (qualifying_position, grid_position — Section 5.1: these
    ARE known before the race itself and are legitimate features).

    This is the single row-building chokepoint used by both
    build_training_dataset (below, for historical races with a known
    label) and src.models.predict (for an upcoming race with no label
    yet) — keeping them on the same function guarantees a live prediction
    is built exactly the same way a training row was, which matters: any
    drift between the two would silently invalidate the model.
    """
    row = {
        "driver": driver,
        "team": team,
        "circuit": circuit,
        "qualifying_position": qualifying_position,
        "grid_position": grid_position,
    }
    row.update(driver_form_features(history, driver, windows))
    row.update(team_form_features(history, team, windows))
    row.update(circuit_history_features(history, driver, team, circuit))
    row.update(_season_form_features(history, season, driver, team))
    return row


def build_training_dataset(races_df: pd.DataFrame, windows=DEFAULT_WINDOWS) -> pd.DataFrame:
    """Build the full leakage-safe training dataset.

    One row per (race_id, driver). For each target race, in chronological
    (season, round) order, every rolling/form/circuit feature is computed
    from races_before(races_df, season, round) — the target race and
    everything after it is excluded from that driver/team's own history
    when computing ITS row. Rows for a given race never see each other's
    late-breaking info either, since all of them are computed from the
    same pre-race `history` slice.

    target_finish is attached last and is clearly marked as the label —
    it must never be fed back in as a feature.
    """
    if races_df.empty:
        return pd.DataFrame()

    races_df = races_df.sort_values(["season", "round", "driver"]).reset_index(drop=True)
    race_keys = races_df[["race_id", "season", "round"]].drop_duplicates().sort_values(
        ["season", "round"]
    )

    rows = []
    for _, race_key in race_keys.iterrows():
        race_id, season, round_number = race_key["race_id"], int(race_key["season"]), int(race_key["round"])
        history = races_before(races_df, season, round_number)
        target_race_rows = races_df[races_df["race_id"] == race_id]

        for _, target in target_race_rows.iterrows():
            driver, team, circuit = target["driver"], target["team"], target["circuit"]

            row = {"race_id": race_id, "season": season, "round": round_number}
            row.update(build_feature_row(
                history,
                driver=driver,
                team=team,
                circuit=circuit,
                season=season,
                qualifying_position=target["qualifying_position"],
                grid_position=target["grid_position"],
                windows=windows,
            ))

            # Label only — must never be used as an input feature anywhere above.
            row["target_finish"] = target["finish_position"]

            rows.append(row)

    dataset = pd.DataFrame(rows)
    logger.info(
        "Built training dataset: %d rows across %d races", len(dataset), len(race_keys)
    )
    return dataset


NON_FEATURE_COLUMNS = {"race_id", "season", "round", "driver", "team", "circuit", "target_finish"}


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Every column in a built dataset that is a model input feature —
    i.e. everything except identifiers/metadata and the label. Centralized
    here so training code (Phase 3) and leakage tests agree on exactly
    what "a feature" means without redefining the exclusion list twice."""
    return [c for c in dataset.columns if c not in NON_FEATURE_COLUMNS]
