from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.fastf1_client import get_upcoming_race


def _fake_schedule():
    return pd.DataFrame({
        "RoundNumber": [1, 2, 3],
        "EventName": ["Season Opener", "Race Two", "Race Three"],
        "Location": ["CircuitA", "CircuitB", "CircuitC"],
        "EventDate": pd.to_datetime(["2026-03-01", "2026-03-15", "2026-03-29"]),
        # Session4 = qualifying, conventionally, one day before the race
        "Session4DateUtc": pd.to_datetime(["2026-02-28", "2026-03-14", "2026-03-28"]),
    })


class TestGetUpcomingRace:
    def test_picks_earliest_event_on_or_after_reference_time(self):
        reference = datetime(2026, 3, 10, tzinfo=timezone.utc)
        with patch(
            "src.data.fastf1_client.fastf1.get_event_schedule", return_value=_fake_schedule()
        ):
            info = get_upcoming_race(reference_time=reference)
        assert info.round == 2
        assert info.event == "Race Two"
        assert info.circuit == "CircuitB"

    def test_qualifying_completed_true_when_quali_time_has_passed(self):
        # reference is AFTER round 2's qualifying (2026-03-14) but before its race (2026-03-15)
        reference = datetime(2026, 3, 14, 20, tzinfo=timezone.utc)
        with patch(
            "src.data.fastf1_client.fastf1.get_event_schedule", return_value=_fake_schedule()
        ):
            info = get_upcoming_race(reference_time=reference)
        assert info.round == 2
        assert info.qualifying_completed is True

    def test_qualifying_completed_false_when_quali_time_has_not_passed(self):
        reference = datetime(2026, 3, 10, tzinfo=timezone.utc)  # before round 2's quali
        with patch(
            "src.data.fastf1_client.fastf1.get_event_schedule", return_value=_fake_schedule()
        ):
            info = get_upcoming_race(reference_time=reference)
        assert info.qualifying_completed is False

    def test_no_upcoming_race_raises_runtime_error(self):
        reference = datetime(2026, 4, 1, tzinfo=timezone.utc)  # after every event in the fake schedule
        with patch(
            "src.data.fastf1_client.fastf1.get_event_schedule", return_value=_fake_schedule()
        ):
            with pytest.raises(RuntimeError):
                get_upcoming_race(reference_time=reference)

    def test_schedule_fetch_failure_raises_runtime_error_not_silent_none(self):
        with patch(
            "src.data.fastf1_client.fastf1.get_event_schedule",
            side_effect=ConnectionError("network blocked"),
        ):
            with pytest.raises(RuntimeError):
                get_upcoming_race(reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc))

    def test_empty_schedule_raises_runtime_error(self):
        with patch(
            "src.data.fastf1_client.fastf1.get_event_schedule", return_value=pd.DataFrame()
        ):
            with pytest.raises(RuntimeError):
                get_upcoming_race(reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc))
