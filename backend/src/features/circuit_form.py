"""
Circuit-specific history features (Section 5.4 of the project guide).

Same cutoff contract as driver_form.py / team_form.py: `history` must
already be the output of src.features.cutoff.races_before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def circuit_history_features(history: pd.DataFrame, driver: str, team: str, circuit: str) -> dict:
    """Driver's and team's historical performance at this specific circuit,
    using only editions already covered by `history` (i.e. strictly
    earlier than the target race's cutoff). Cold start (no prior starts
    at this circuit) is explicit NaN + a zero start count, per Section 5.4's
    "explicit missing/default strategy" requirement — never silently
    defaulted to e.g. the field average.
    """
    circuit_races = history[history["circuit"] == circuit]

    driver_at_circuit = circuit_races[circuit_races["driver"] == driver]
    team_at_circuit = circuit_races[circuit_races["team"] == team]

    return {
        "circuit_driver_avg_finish": (
            float(driver_at_circuit["finish_position"].mean())
            if len(driver_at_circuit) > 0 else np.nan
        ),
        "circuit_driver_prior_starts": int(len(driver_at_circuit)),
        "circuit_team_avg_finish": (
            float(team_at_circuit["finish_position"].mean())
            if len(team_at_circuit) > 0 else np.nan
        ),
        "circuit_team_prior_starts": int(len(team_at_circuit)),
    }
