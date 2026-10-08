"""
Data-loading and formatting for the Phase 6 dashboard, deliberately kept
free of any `streamlit` import so these functions can be unit-tested
directly (Streamlit UI code itself is best tested via
streamlit.testing.v1.AppTest as an end-to-end smoke test, not via these
finer-grained checks). src.app.dashboard is a thin layer of `st.*` calls
around the functions here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.app.labels import label_for
from src.models.predict import ModelBundle, load_model_bundle


@dataclass
class ArtifactStatus:
    """One row of the dashboard's data-freshness/status indicator
    (Section 11's explicit requirement)."""

    name: str
    exists: bool
    path: str
    last_modified: Optional[str] = None
    detail: Optional[str] = None


def check_artifact_status(races_path: Path, model_path: Path, backtest_summary_path: Path) -> list[ArtifactStatus]:
    """Never raises — a dashboard's freshness indicator is precisely the
    place that must keep working even when nothing else has been built
    yet (Section 10 / 11: report missing data, don't fabricate or crash)."""
    statuses = []

    if races_path.exists():
        try:
            races_df = pd.read_csv(races_path, parse_dates=["date"])
            detail = f"{len(races_df)} driver-race rows, {races_df['race_id'].nunique()} races"
        except Exception as exc:  # noqa: BLE001
            detail = f"exists but could not be read: {exc}"
        statuses.append(ArtifactStatus(
            "Historical races table", True, str(races_path),
            last_modified=_mtime_str(races_path), detail=detail,
        ))
    else:
        statuses.append(ArtifactStatus("Historical races table", False, str(races_path)))

    if model_path.exists():
        try:
            bundle = load_model_bundle(model_path)
            detail = f"{len(bundle.feature_cols)} features, saved {bundle.saved_at.isoformat()}"
        except Exception as exc:  # noqa: BLE001
            detail = f"exists but could not be loaded: {exc}"
        statuses.append(ArtifactStatus(
            "Trained model", True, str(model_path), last_modified=_mtime_str(model_path), detail=detail,
        ))
    else:
        statuses.append(ArtifactStatus("Trained model", False, str(model_path)))

    if backtest_summary_path.exists():
        try:
            with open(backtest_summary_path) as f:
                summary = json.load(f)
            detail = f"{summary.get('overall', {}).get('n_races', '?')} backtested races"
        except Exception as exc:  # noqa: BLE001
            detail = f"exists but could not be read: {exc}"
        statuses.append(ArtifactStatus(
            "Backtest report", True, str(backtest_summary_path),
            last_modified=_mtime_str(backtest_summary_path), detail=detail,
        ))
    else:
        statuses.append(ArtifactStatus("Backtest report", False, str(backtest_summary_path)))

    return statuses


def _mtime_str(path: Path) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc).isoformat()


def load_races_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


def load_bundle(path: Path) -> Optional[ModelBundle]:
    if not path.exists():
        return None
    return load_model_bundle(path)


def load_backtest_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_backtest_predictions(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def format_prediction_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Predicted finish + win probability + the driver/team recent-form
    columns Section 11 asks to show alongside the prediction ('Recent
    driver/team form'), in one display-ready table."""
    display_cols = {
        "predicted_rank": "Predicted rank",
        "driver": "Driver",
        "team": "Team",
        "win_prob": "P1 probability",
        "pred_finish": "Predicted finish (raw)",
        "qualifying_position": "Qualifying position",
        "driver_form_3": "Driver form (last 3)",
        "team_form_3": "Team form (last 3)",
    }
    available = [c for c in display_cols if c in predictions.columns]
    out = predictions[available].rename(columns=display_cols).copy()
    if "P1 probability" in out.columns:
        out["P1 probability"] = (out["P1 probability"] * 100).round(1).astype(str) + "%"
    return out


def format_feature_importance_table(importance: pd.Series, top_n: int = 10) -> pd.DataFrame:
    top = importance.head(top_n)
    return pd.DataFrame({
        "Factor": [label_for(f) for f in top.index],
        "Importance": top.values,
    })


def format_backtest_summary_table(summary: dict) -> pd.DataFrame:
    """One row per key in the summary dict ('overall', 'season_2022', ...),
    'overall' pinned first so it isn't lost among season rows."""
    rows = []
    ordered_keys = ["overall"] + sorted(k for k in summary if k != "overall")
    for key in ordered_keys:
        if key not in summary:
            continue
        row = {"Period": "Overall" if key == "overall" else key.replace("season_", "Season ")}
        row.update(summary[key])
        rows.append(row)
    return pd.DataFrame(rows)


def race_prediction_vs_actual(predictions: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Section 11's 'race-by-race prediction vs actual comparison' for
    one specific backtested race, sorted by actual finishing order."""
    race_df = predictions[predictions["race_id"] == race_id].copy()
    race_df["abs_error"] = (race_df["target_finish"] - race_df["pred_finish"]).abs()
    cols = ["driver", "team", "target_finish", "pred_finish", "abs_error", "win_prob"]
    available = [c for c in cols if c in race_df.columns]
    return race_df[available].sort_values("target_finish").reset_index(drop=True)
