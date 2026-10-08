"""
Shared test fixture: a small, fully synthetic multi-race races_df so
feature/leakage tests don't depend on live FastF1 data (which this
environment's network can't reach — see README).
"""
from __future__ import annotations

import pandas as pd

from src.data.loaders import RACES_TABLE_COLUMNS, _is_classified_finish


def make_synthetic_races_df() -> pd.DataFrame:
    """4 races across 2 seasons, 2 drivers (VER/HAM), 2 teams, 2 circuits.
    finish/points are hand-picked so tests can assert exact expected
    rolling-average values instead of just "not NaN"."""
    races = [
        # season, round, event, circuit, date, driver, team, grid, quali, finish, status, points
        (2022, 1, "Bahrain GP", "Bahrain", "2022-03-20", "VER", "RedBull", 3, 3, 2, "Finished", 18),
        (2022, 1, "Bahrain GP", "Bahrain", "2022-03-20", "HAM", "Mercedes", 5, 5, 4, "Finished", 12),
        (2022, 2, "Monza GP", "Monza", "2022-09-11", "VER", "RedBull", 1, 1, 1, "Finished", 25),
        (2022, 2, "Monza GP", "Monza", "2022-09-11", "HAM", "Mercedes", 4, 4, 3, "Finished", 15),
        (2022, 3, "Suzuka GP", "Suzuka", "2022-10-09", "VER", "RedBull", 2, 2, 1, "Finished", 25),
        (2022, 3, "Suzuka GP", "Suzuka", "2022-10-09", "HAM", "Mercedes", 6, 6, 5, "Finished", 10),
        (2023, 1, "Bahrain GP", "Bahrain", "2023-03-05", "VER", "RedBull", 1, 1, 1, "Finished", 25),
        (2023, 1, "Bahrain GP", "Bahrain", "2023-03-05", "HAM", "Mercedes", 3, 3, 2, "Finished", 18),
    ]
    rows = []
    for season, rnd, event, circuit, date, driver, team, grid, quali, finish, status, points in races:
        rows.append({
            "race_id": f"{season}_{rnd}",
            "season": season,
            "round": rnd,
            "event": event,
            "circuit": circuit,
            "date": pd.Timestamp(date),
            "driver": driver,
            "team": team,
            "grid_position": grid,
            "qualifying_position": quali,
            "finish_position": finish,
            "status": status,
            "points": points,
            "classified_finish": _is_classified_finish(status),
        })
    return pd.DataFrame(rows, columns=RACES_TABLE_COLUMNS)


def make_large_synthetic_races_df(n_races: int = 24, seed: int = 7) -> pd.DataFrame:
    """A bigger, procedurally generated synthetic dataset for tests that
    need enough races for a meaningful chronological train/val/test split
    and a gradient-boosting model that can actually fit (Phase 3).

    4 drivers on 2 teams, cycling through 6 circuits, 2 seasons x 12
    rounds. Finishing position is qualifying position plus noise
    (re-ranked into a valid 1..4 permutation each race) so there is real,
    learnable signal — qualifying position should correlate with finish —
    without being a deterministic 1:1 mapping. This is still a toy
    dataset; it is NOT a substitute for validating against real FastF1
    history (see README's network-access note), only for exercising the
    modeling/splitting code paths deterministically and quickly in CI.
    """
    import numpy as np

    drivers = ["D1", "D2", "D3", "D4"]
    teams = {"D1": "TeamA", "D2": "TeamA", "D3": "TeamB", "D4": "TeamB"}
    circuits = ["C1", "C2", "C3", "C4", "C5", "C6"]
    points_map = {1: 25, 2: 18, 3: 15, 4: 12}

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_races):
        season = 2022 + i // 12
        rnd = i % 12 + 1
        circuit = circuits[i % len(circuits)]
        date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=14 * i)

        # base skill ranking D1 > D2 > D3 > D4, with noise, re-ranked into a permutation
        skill_score = np.arange(4) + rng.normal(0, 0.7, size=4)
        quali_order = np.argsort(skill_score)
        quali_position = {drivers[d]: pos + 1 for pos, d in enumerate(quali_order)}

        finish_score = np.array([quali_position[d] for d in drivers]) + rng.normal(0, 0.6, size=4)
        finish_order = np.argsort(finish_score)
        finish_position = {drivers[d]: pos + 1 for pos, d in enumerate(finish_order)}

        for d in drivers:
            finish = finish_position[d]
            rows.append({
                "race_id": f"{season}_{rnd}",
                "season": season,
                "round": rnd,
                "event": f"Race {i + 1}",
                "circuit": circuit,
                "date": date,
                "driver": d,
                "team": teams[d],
                "grid_position": quali_position[d],
                "qualifying_position": quali_position[d],
                "finish_position": finish,
                "status": "Finished",
                "points": points_map.get(finish, 0),
                "classified_finish": True,
            })
    return pd.DataFrame(rows, columns=RACES_TABLE_COLUMNS)
