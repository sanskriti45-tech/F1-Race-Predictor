"""
Thin local API layer exposing the real backend prediction pipeline over
HTTP, so the frontend can fetch() real data instead of the frontend
either embedding static JSON or recomputing anything itself.

Deliberately stdlib-only (http.server + json) — "the thinnest possible
API/data-serving layer around the existing prediction functions," per
the integration spec, with zero new dependency added to the project.

This module contains NO prediction logic of its own. Every number in
build_next_race_payload() comes from calling the real, already-tested
backend functions (get_upcoming_race, load_session, predict_upcoming_race,
load_model_bundle) and reading real files the backend pipeline already
produces (races.csv, the saved model bundle, backtest reports). If any
of those don't exist yet, the corresponding payload field is null/empty
and a status flag is false — never fabricated.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import pandas as pd

from src.app.cli import entrants_from_qualifying
from src.config import get_logger, settings
from src.data.fastf1_client import get_upcoming_race, load_session
from src.models.predict import (
    PredictionCutoffError,
    load_model_bundle,
    predict_upcoming_race,
)

logger = get_logger(__name__)


def build_next_race_payload() -> dict:
    """Assemble the frontend's expected JSON shape entirely from real
    backend state. Every status flag reflects something actually checked
    this call, not an assumption."""
    settings.ensure_dirs()
    races_path = settings.data_processed_dir / "races.csv"
    model_path = settings.models_dir / "baseline_gbm.joblib"
    backtest_summary_path = settings.reports_dir / "backtest_summary.json"
    backtest_predictions_path = settings.reports_dir / "backtest_predictions.csv"

    historical_data_loaded = races_path.exists()
    model_loaded = model_path.exists()
    races_df = pd.read_csv(races_path, parse_dates=["date"]) if historical_data_loaded else None
    bundle = load_model_bundle(model_path) if model_loaded else None

    fastf1_available = False
    next_race_info = None
    error = None
    try:
        next_race_info = get_upcoming_race()
        fastf1_available = True
    except Exception as exc:  # noqa: BLE001 - network/schedule failure is expected and honest, not a crash
        error = f"Could not identify next race via FastF1: {exc}"
        logger.warning(error)

    predictions = []
    prediction_generated = False
    if races_df is not None and bundle is not None and next_race_info is not None:
        if not next_race_info.qualifying_completed:
            error = error or "Qualifying has not completed for the next race yet."
        else:
            try:
                quali_result = load_session(next_race_info.season, next_race_info.round, "Q")
                if quali_result.ok and quali_result.results is not None and len(quali_result.results) > 0:
                    entrants = entrants_from_qualifying(quali_result.results)
                    pred_df = predict_upcoming_race(
                        races_df, bundle,
                        season=next_race_info.season, round_number=next_race_info.round,
                        circuit=next_race_info.circuit, entrants=entrants, qualifying_completed=True,
                    )
                    # to_json/loads round-trip gives correct JSON null for
                    # NaN cold-start feature values (pandas' default).
                    predictions = json.loads(pred_df.to_json(orient="records"))
                    prediction_generated = True
                else:
                    error = error or (
                        f"Could not load usable qualifying results: "
                        f"{quali_result.error or quali_result.warnings}"
                    )
            except PredictionCutoffError as exc:
                error = error or str(exc)
            except Exception as exc:  # noqa: BLE001
                error = error or f"Prediction generation failed: {exc}"
                logger.warning(error)

    feature_values = []
    if bundle is not None and bundle.feature_importance is not None:
        feature_values = [
            {"feature": k, "value": round(float(v), 4)}
            for k, v in bundle.feature_importance.head(10).items()
        ]

    backtest = None
    if backtest_summary_path.exists():
        with open(backtest_summary_path) as f:
            backtest = json.load(f)

    history = []
    if backtest_predictions_path.exists():
        hist_df = pd.read_csv(backtest_predictions_path)
        cols = [c for c in ["race_id", "driver", "team", "target_finish", "pred_finish", "win_prob"] if c in hist_df.columns]
        if cols:
            history = json.loads(hist_df[cols].tail(50).to_json(orient="records"))

    return {
        "dataFreshness": bundle.saved_at.isoformat() if bundle else None,
        "modelVersion": "baseline_gbm" if bundle else None,
        "fastf1Available": fastf1_available,
        "historicalDataLoaded": historical_data_loaded,
        "modelLoaded": model_loaded,
        "predictionGenerated": prediction_generated,
        "nextRace": ({
            "name": next_race_info.event,
            "circuit": next_race_info.circuit,
            "date": str(next_race_info.date),
            "cutoffNote": next_race_info.retrieved_at.isoformat() + " (after qualifying, before race start)",
            "statusNote": "Qualifying complete" if next_race_info.qualifying_completed else "Waiting for qualifying",
        } if next_race_info else None),
        "predictions": predictions,
        "featureValues": feature_values,
        "backtest": backtest,
        "history": history,
        "error": error,
    }


def _is_allowed_origin(origin: Optional[str]) -> bool:
    """Local-dev-only allowlist: localhost/127.0.0.1 at any port, plus the
    'null' origin a browser sends for a page opened via file://. Never a
    wildcard — this reflects back only an origin that already matched the
    allowlist, per 'do not disable browser security globally.'"""
    if not origin:
        return False
    return (
        origin == "null"
        or origin.startswith("http://localhost")
        or origin.startswith("http://127.0.0.1")
    )


class ApiHandler(BaseHTTPRequestHandler):
    """Routes: GET /api/health, GET /api/prediction/next-race. Nothing else."""

    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if _is_allowed_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json({"status": "ok"})
        elif self.path == "/api/prediction/next-race":
            try:
                self._send_json(build_next_race_payload())
            except Exception as exc:  # noqa: BLE001 - last-resort guard; should be unreachable
                logger.exception("Unhandled error building prediction payload")
                self._send_json({"error": str(exc)}, status=500)
        else:
            self._send_json({"error": "not found"}, status=404)

    def log_message(self, fmt: str, *args) -> None:  # route stdlib's default stderr logging through ours
        logger.info("%s - %s", self.address_string(), fmt % args)
