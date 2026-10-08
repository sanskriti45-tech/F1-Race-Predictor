"""
Feature column -> human-readable label, shared by src.app.cli and
src.app.dashboard so the "Top factors" / feature-importance display is
identical in both places and only ever renames what
src.models.train.compute_feature_importance actually ranked highest —
never decides which factors are shown (Section 2: "Do not hard-code this
example output").
"""
from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    "qualifying_position": "qualifying position",
    "grid_position": "starting grid position",
    "driver_form_3": "driver recent form (last 3 races)",
    "driver_form_5": "driver recent form (last 5 races)",
    "driver_quali_form_3": "driver recent qualifying form (last 3)",
    "driver_quali_form_5": "driver recent qualifying form (last 5)",
    "driver_points_3": "driver points (last 3 races)",
    "driver_points_5": "driver points (last 5 races)",
    "driver_finish_rate_3": "driver finish rate (last 3 races)",
    "driver_finish_rate_5": "driver finish rate (last 5 races)",
    "team_form_3": "team recent pace (last 3 races)",
    "team_form_5": "team recent pace (last 5 races)",
    "team_quali_form_3": "team recent qualifying pace (last 3)",
    "team_quali_form_5": "team recent qualifying pace (last 5)",
    "team_points_3": "team points (last 3 races)",
    "team_points_5": "team points (last 5 races)",
    "circuit_driver_avg_finish": "driver's history at this circuit",
    "circuit_team_avg_finish": "team's history at this circuit",
    "circuit_driver_prior_starts": "driver's prior starts at this circuit",
    "circuit_team_prior_starts": "team's prior starts at this circuit",
    "season_avg_finish": "current-season form",
    "season_avg_quali": "current-season qualifying form",
    "season_points": "current-season points",
}


def label_for(feature_col: str) -> str:
    return FEATURE_LABELS.get(feature_col, feature_col)
