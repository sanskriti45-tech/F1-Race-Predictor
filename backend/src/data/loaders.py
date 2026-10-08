"""
Build the unified, normalized "races" table from FastF1 session results.

This is the single source table every feature builder in src/features/
reads from. One row = one driver in one race. Schema:

    race_id               str    "2023_16"  (f"{season}_{round}")
    season                int
    round                 int    sequential within season, 1-indexed
    event                 str    e.g. "Monza"
    circuit               str    e.g. "Autodromo Nazionale Monza"
    date                  pd.Timestamp   race date (for display/logging only —
                                          chronological ORDER for leakage
                                          purposes always uses (season, round),
                                          never raw date; see features/dataset.py)
    driver                str    FastF1 driver Abbreviation, e.g. "VER"
    team                  str    FastF1 TeamName
    grid_position         float  starting grid position (post-penalties, as
                                  reported by FastF1's race results)
    qualifying_position   float  qualifying classification position
    finish_position       float  official race classification position
    status                str    e.g. "Finished", "+1 Lap", "Accident"
    points                float  points scored in this race
    classified_finish     bool   True if `status` represents a classified
                                  finish (see _is_classified_finish)

Deliberately NOT included here: anything from qualifying lap TIMES beyond
position (Section 5.1 lists qualifying_gap_to_pole as "if reliably
available" — FastF1's Q1/Q2/Q3 session structure makes a single
apples-to-apples pole gap nontrivial across formats/eras, so it's left as
a documented future extension rather than a half-correct MVP field).
"""
from __future__ import annotations

import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

RACES_TABLE_COLUMNS = [
    "race_id", "season", "round", "event", "circuit", "date",
    "driver", "team", "grid_position", "qualifying_position",
    "finish_position", "status", "points", "classified_finish",
]


def _is_classified_finish(status: str) -> bool:
    """A driver is a classified finisher if they finished the race or were
    lapped but still classified (status starts with '+', e.g. '+1 Lap').
    Anything else (Accident, Engine, Retired, DNF, DSQ, ...) is not."""
    if not isinstance(status, str):
        return False
    return status == "Finished" or status.startswith("+")


def build_race_rows(
    race_results: pd.DataFrame,
    quali_results: pd.DataFrame,
    *,
    season: int,
    round_number: int,
    event: str,
    circuit: str,
    date,
) -> pd.DataFrame:
    """Normalize one race's FastF1 `results` frame + its qualifying
    session's `results` frame into rows of the unified races table.

    Both inputs are expected to be FastF1 `session.results`-shaped frames
    (as returned inside a SessionLoadResult from fastf1_client). Caller is
    responsible for having already validated these via
    src.data.validation.validate_session_result before calling this.
    """
    if race_results is None or len(race_results) == 0:
        raise ValueError(f"race_results is empty for {season} round {round_number} ({event})")

    quali_pos_by_driver = {}
    if quali_results is not None and len(quali_results) > 0:
        quali_pos_by_driver = dict(zip(quali_results["Abbreviation"], quali_results["Position"]))
    else:
        logger.warning(
            "No qualifying results for %s round %s (%s) — qualifying_position will be null",
            season, round_number, event,
        )

    rows = []
    for _, r in race_results.iterrows():
        driver = r["Abbreviation"]
        rows.append({
            "race_id": f"{season}_{round_number}",
            "season": season,
            "round": round_number,
            "event": event,
            "circuit": circuit,
            "date": pd.Timestamp(date),
            "driver": driver,
            "team": r["TeamName"],
            "grid_position": r.get("GridPosition"),
            "qualifying_position": quali_pos_by_driver.get(driver),
            "finish_position": r.get("Position"),
            "status": r.get("Status"),
            "points": r.get("Points"),
            "classified_finish": _is_classified_finish(r.get("Status")),
        })

    df = pd.DataFrame(rows, columns=RACES_TABLE_COLUMNS)
    logger.info(
        "Built %d driver-rows for %s round %s (%s)", len(df), season, round_number, event
    )
    return df


def combine_races(race_row_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-race frames from build_race_rows into the master
    races table, sorted chronologically. (season, round) is the ordering
    key used everywhere for "before/after" leakage checks — never `date`,
    since date alone can't disambiguate sprint weekends or be trusted
    across inconsistent metadata sources."""
    if not race_row_frames:
        return pd.DataFrame(columns=RACES_TABLE_COLUMNS)
    combined = pd.concat(race_row_frames, ignore_index=True)
    combined = combined.sort_values(["season", "round", "driver"]).reset_index(drop=True)
    return combined
