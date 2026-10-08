"""
Thin, defensive wrapper around FastF1 session loading.

Responsibilities (Phase 1 scope only):
  - Enable FastF1's on-disk cache exactly once per process.
  - Load a single race or qualifying session for a given season/event.
  - Log the exact season, event, session type and retrieval timestamp.
  - Be robust to a session that FastF1 cannot fully load (missing fields,
    unavailable Ergast data for recent events, etc.) rather than crashing.
  - Never guess at data: if a session fails to load, say so and return
    an explicit failure result instead of fabricating anything.

This module does NOT do feature engineering, does NOT decide prediction
cutoffs, and does NOT persist processed tables. That is later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

import fastf1
import pandas as pd

from src.config import get_logger, settings

logger = get_logger(__name__)

SessionType = Literal["R", "Q", "FP1", "FP2", "FP3", "S", "SQ"]

_cache_enabled = False


def _ensure_cache_enabled() -> None:
    """Enable FastF1's cache once. FastF1 raises if you call this twice
    with a different path, and warns/no-ops harmlessly if called again
    with the same path, so we guard with a module-level flag to keep the
    log clean and make the "cache is on" state explicit and testable."""
    global _cache_enabled
    if _cache_enabled:
        return
    settings.ensure_dirs()
    fastf1.Cache.enable_cache(str(settings.fastf1_cache_dir))
    _cache_enabled = True
    logger.info("FastF1 cache enabled at %s", settings.fastf1_cache_dir)


def _safe_get_loaded_frame(session: "fastf1.core.Session", attr: str) -> Optional[pd.DataFrame]:
    """Read session.laps / session.results defensively.

    FastF1 exposes these as properties that raise DataNotLoadedError (not
    AttributeError) when session.load() didn't actually populate them --
    which happens when load() "succeeds" but the underlying data source
    (e.g. Ergast, live timing) returned nothing. A plain getattr(..., None)
    does NOT catch that, so we catch broadly here and treat it as "no data",
    consistent with how an empty DataFrame is already handled downstream.
    """
    try:
        return getattr(session, attr)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure here means "no data"
        logger.warning("session.%s was not accessible: %s", attr, exc)
        return None


@dataclass
class SessionLoadResult:
    """Outcome of attempting to load one FastF1 session.

    `ok` is the single source of truth for whether this session is usable
    downstream. Callers must check it before touching `laps` / `results` /
    `session`, since FastF1 can return an object with empty frames instead
    of raising (e.g. very recent races before Ergast has results).
    """

    season: int
    event: str
    session_type: SessionType
    ok: bool
    retrieved_at: datetime
    session: Optional["fastf1.core.Session"] = None
    laps: Optional[pd.DataFrame] = None
    results: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "season": self.season,
            "event": self.event,
            "session_type": self.session_type,
            "ok": self.ok,
            "retrieved_at": self.retrieved_at.isoformat(),
            "n_laps": None if self.laps is None else len(self.laps),
            "n_results": None if self.results is None else len(self.results),
            "error": self.error,
            "warnings": self.warnings,
        }


def load_session(
    season: int,
    event: str,
    session_type: SessionType,
    *,
    load_telemetry: bool = False,
    load_weather: bool = False,
) -> SessionLoadResult:
    """Load one FastF1 session and return a SessionLoadResult.

    Never raises for ordinary data-availability problems (missing Ergast
    results for a very recent race, a cancelled session, etc.) — those are
    reported via `ok=False` / `error` / `warnings` so a batch loader can
    skip a bad session instead of crashing a whole historical pull.
    Programming errors (bad season/event/session_type) still raise, since
    those indicate a caller bug, not a data-availability issue.
    """
    _ensure_cache_enabled()
    retrieved_at = datetime.now(timezone.utc)
    warnings: list[str] = []

    logger.info(
        "Loading session: season=%s event=%s session_type=%s retrieved_at=%s",
        season, event, session_type, retrieved_at.isoformat(),
    )

    try:
        session = fastf1.get_session(season, event, session_type)
    except Exception as exc:  # noqa: BLE001 - bad identifiers are a caller bug
        logger.error("get_session failed for %s %s %s: %s", season, event, session_type, exc)
        raise

    try:
        session.load(telemetry=load_telemetry, weather=load_weather, laps=True, messages=False)
    except Exception as exc:  # noqa: BLE001 - genuine data-availability failure
        msg = f"session.load() raised for {season} {event} {session_type}: {exc}"
        logger.warning(msg)
        return SessionLoadResult(
            season=season, event=event, session_type=session_type, ok=False,
            retrieved_at=retrieved_at, error=msg,
        )

    laps = _safe_get_loaded_frame(session, "laps")
    results = _safe_get_loaded_frame(session, "results")

    if results is None or len(results) == 0:
        warnings.append(
            "No results data returned (expected for very recent sessions where "
            "Ergast has not published results yet, or for a cancelled session)."
        )
    if laps is None or len(laps) == 0:
        warnings.append("No lap data returned.")

    ok = not (
        (results is None or len(results) == 0)
        and (laps is None or len(laps) == 0)
    )

    if not ok:
        logger.warning(
            "Session %s %s %s loaded but produced no usable data.",
            season, event, session_type,
        )
    for w in warnings:
        logger.warning(w)

    return SessionLoadResult(
        season=season,
        event=event,
        session_type=session_type,
        ok=ok,
        retrieved_at=retrieved_at,
        session=session,
        laps=laps,
        results=results,
        warnings=warnings,
    )


def load_race_and_qualifying(season: int, event: str) -> tuple[SessionLoadResult, SessionLoadResult]:
    """Convenience helper: load both the race and qualifying session for
    one event. Used by Phase 1's verification slice and later by the
    historical dataset builder (Phase 2)."""
    race = load_session(season, event, "R")
    qualifying = load_session(season, event, "Q")
    return race, qualifying


@dataclass
class UpcomingRaceInfo:
    """What Phase 5's 'identify the next race' step (Section 10) needs to
    know about the next Grand Prix on the calendar."""

    season: int
    round: int
    event: str
    circuit: str
    date: "datetime"
    qualifying_completed: bool
    retrieved_at: datetime


def get_upcoming_race(reference_time: Optional[datetime] = None) -> UpcomingRaceInfo:
    """Identify the next race on the calendar at or after `reference_time`
    (defaults to now, UTC), using FastF1's event schedule.

    Raises RuntimeError (not a silent None / fabricated race) if the
    schedule can't be fetched or no upcoming race is found — Section 10 is
    explicit: 'If required data is missing, report that instead of
    fabricating a prediction.' Whether qualifying has already happened for
    that event is inferred from the event's scheduled qualifying session
    time versus `reference_time`; callers (src.models.predict) use this to
    decide whether a qualifying-based prediction can be made at all.
    """
    _ensure_cache_enabled()
    reference_time = reference_time or datetime.now(timezone.utc)
    retrieved_at = datetime.now(timezone.utc)

    try:
        schedule = fastf1.get_event_schedule(reference_time.year, include_testing=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not fetch the {reference_time.year} event schedule from FastF1: {exc}"
        ) from exc

    if schedule is None or len(schedule) == 0:
        raise RuntimeError(f"FastF1 returned an empty event schedule for {reference_time.year}")

    upcoming = schedule[schedule["EventDate"] >= pd.Timestamp(reference_time).tz_localize(None)]
    if len(upcoming) == 0:
        raise RuntimeError(
            f"No upcoming race found in the {reference_time.year} schedule after {reference_time.isoformat()}"
        )

    next_event = upcoming.sort_values("EventDate").iloc[0]

    quali_completed = False
    if "Session4DateUtc" in next_event and pd.notna(next_event.get("Session4DateUtc")):
        # FastF1 labels qualifying as "Session4" for a conventional weekend;
        # a sprint weekend's layout differs, which is exactly the kind of
        # "don't assume all sessions contain identical fields" case Section
        # 4.1 warns about — if we can't confidently tell, we conservatively
        # report qualifying as NOT completed rather than guessing.
        quali_completed = pd.Timestamp(next_event["Session4DateUtc"]) <= pd.Timestamp(reference_time).tz_localize(None)

    info = UpcomingRaceInfo(
        season=int(next_event["EventDate"].year),
        round=int(next_event["RoundNumber"]),
        event=str(next_event["EventName"]),
        circuit=str(next_event.get("Location", next_event["EventName"])),
        date=next_event["EventDate"],
        qualifying_completed=quali_completed,
        retrieved_at=retrieved_at,
    )
    logger.info(
        "Identified upcoming race: %s %s (round %s), qualifying_completed=%s, retrieved_at=%s",
        info.season, info.event, info.round, info.qualifying_completed, retrieved_at.isoformat(),
    )
    return info
