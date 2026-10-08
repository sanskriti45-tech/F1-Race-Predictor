"""
Sanity checks for raw session data pulled from FastF1.

Phase 1 scope: structural validation only (are the columns/shape we expect
present?). This deliberately does NOT check feature-engineering-level
concerns like leakage — that lives in tests/test_leakage.py once the
feature builder exists in Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import get_logger
from src.data.fastf1_client import SessionLoadResult

logger = get_logger(__name__)

REQUIRED_RESULTS_COLUMNS = {
    "DriverNumber",
    "Abbreviation",
    "TeamName",
    "Position",
    "GridPosition",
    "Status",
    "Points",
}

REQUIRED_LAPS_COLUMNS = {
    "Driver",
    "LapNumber",
    "LapTime",
}


@dataclass
class ValidationReport:
    ok: bool
    issues: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.issues.append(msg)
        self.ok = False


def validate_session_result(result: SessionLoadResult) -> ValidationReport:
    """Check that a loaded session has the shape later phases will rely on.

    Missing columns are reported, not silently patched — Section 5.6 of the
    project guide is explicit that missingness must be documented, never
    silently filled.
    """
    report = ValidationReport(ok=True)

    if not result.ok:
        report.add(f"Session load reported ok=False: {result.error or result.warnings}")
        return report

    if result.results is not None and len(result.results) > 0:
        missing = REQUIRED_RESULTS_COLUMNS - set(result.results.columns)
        if missing:
            report.add(f"results is missing expected columns: {sorted(missing)}")
        else:
            null_counts = result.results[list(REQUIRED_RESULTS_COLUMNS)].isna().sum()
            nulled = {c: int(n) for c, n in null_counts.items() if n > 0}
            if nulled:
                report.add(f"results has null values in required columns: {nulled}")
    else:
        report.add("results frame is empty or missing.")

    if result.session_type == "R":
        if result.laps is not None and len(result.laps) > 0:
            missing = REQUIRED_LAPS_COLUMNS - set(result.laps.columns)
            if missing:
                report.add(f"laps is missing expected columns: {sorted(missing)}")
        else:
            report.add("laps frame is empty or missing (unexpected for a Race session).")

    if report.ok:
        logger.info(
            "Validation passed for %s %s %s", result.season, result.event, result.session_type
        )
    else:
        logger.warning(
            "Validation found %d issue(s) for %s %s %s: %s",
            len(report.issues), result.season, result.event, result.session_type, report.issues,
        )

    return report
