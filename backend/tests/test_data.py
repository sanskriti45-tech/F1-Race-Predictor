"""
Tests for src.data.fastf1_client and src.data.validation.

These tests mock fastf1.get_session so they verify our wrapper's logic
(retry-free error handling, ok/warnings semantics, validation rules)
without requiring live network access to FastF1's backends. A separate,
manually-run integration check (see README) hits the real API.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.fastf1_client import load_session, load_race_and_qualifying
from src.data.validation import validate_session_result, REQUIRED_RESULTS_COLUMNS


def _make_fake_session(results_df, laps_df, raise_on_load=False):
    fake = MagicMock()
    if raise_on_load:
        fake.load.side_effect = RuntimeError("simulated FastF1 backend failure")
    else:
        def _set_attrs(*args, **kwargs):
            fake.results = results_df
            fake.laps = laps_df
        fake.load.side_effect = _set_attrs
    return fake


def _full_results_df(n=3):
    return pd.DataFrame({
        "DriverNumber": [str(i) for i in range(1, n + 1)],
        "Abbreviation": [f"D{i}" for i in range(1, n + 1)],
        "TeamName": ["TeamA"] * n,
        "Position": list(range(1, n + 1)),
        "GridPosition": list(range(1, n + 1)),
        "Status": ["Finished"] * n,
        "Points": [25, 18, 15][:n],
    })


def _full_laps_df(n=3):
    return pd.DataFrame({
        "Driver": [f"D{i}" for i in range(1, n + 1)],
        "LapNumber": [1] * n,
        "LapTime": pd.to_timedelta(["0:01:30"] * n),
    })


class TestLoadSession:
    def test_successful_load_marks_ok_true(self):
        fake = _make_fake_session(_full_results_df(), _full_laps_df())
        with patch("src.data.fastf1_client.fastf1.get_session", return_value=fake):
            result = load_session(2023, "Monza", "R")
        assert result.ok is True
        assert result.error is None
        assert len(result.results) == 3
        assert len(result.laps) == 3
        assert result.retrieved_at is not None

    def test_empty_results_and_laps_marks_ok_false_without_raising(self):
        fake = _make_fake_session(pd.DataFrame(), pd.DataFrame())
        with patch("src.data.fastf1_client.fastf1.get_session", return_value=fake):
            result = load_session(2026, "Some Future GP", "R")
        assert result.ok is False
        assert result.error is None  # not an exception - just no data
        assert len(result.warnings) >= 1

    def test_partial_data_results_only_still_ok(self):
        # Results present but laps empty: e.g. qualifying session summary
        fake = _make_fake_session(_full_results_df(), pd.DataFrame())
        with patch("src.data.fastf1_client.fastf1.get_session", return_value=fake):
            result = load_session(2023, "Monza", "Q")
        assert result.ok is True
        assert "No lap data returned." in result.warnings

    def test_session_load_exception_is_caught_not_raised(self):
        fake = _make_fake_session(None, None, raise_on_load=True)
        with patch("src.data.fastf1_client.fastf1.get_session", return_value=fake):
            result = load_session(2023, "Monza", "R")
        assert result.ok is False
        assert "simulated FastF1 backend failure" in result.error

    def test_bad_identifier_reraises(self):
        with patch(
            "src.data.fastf1_client.fastf1.get_session",
            side_effect=ValueError("invalid event"),
        ):
            with pytest.raises(ValueError):
                load_session(2023, "Not A Real Grand Prix", "R")

    def test_load_race_and_qualifying_calls_both_session_types(self):
        fake = _make_fake_session(_full_results_df(), _full_laps_df())
        with patch("src.data.fastf1_client.fastf1.get_session", return_value=fake) as mock_get:
            race, quali = load_race_and_qualifying(2023, "Monza")
        called_session_types = [call.args[2] for call in mock_get.call_args_list]
        assert called_session_types == ["R", "Q"]
        assert race.ok and quali.ok


class TestValidation:
    def test_valid_session_passes(self):
        fake = _make_fake_session(_full_results_df(), _full_laps_df())
        with patch("src.data.fastf1_client.fastf1.get_session", return_value=fake):
            result = load_session(2023, "Monza", "R")
        report = validate_session_result(result)
        assert report.ok is True
        assert report.issues == []

    def test_missing_required_column_is_flagged(self):
        bad_results = _full_results_df().drop(columns=["GridPosition"])
        fake = _make_fake_session(bad_results, _full_laps_df())
        with patch("src.data.fastf1_client.fastf1.get_session", return_value=fake):
            result = load_session(2023, "Monza", "R")
        report = validate_session_result(result)
        assert report.ok is False
        assert any("GridPosition" in issue for issue in report.issues)

    def test_failed_load_is_flagged_without_crashing(self):
        fake = _make_fake_session(None, None, raise_on_load=True)
        with patch("src.data.fastf1_client.fastf1.get_session", return_value=fake):
            result = load_session(2023, "Monza", "R")
        report = validate_session_result(result)
        assert report.ok is False
        assert len(report.issues) == 1

    def test_required_columns_constant_matches_docstring_intent(self):
        # Guards against silent drift between the loader's expectations
        # and what validation actually checks.
        assert "Position" in REQUIRED_RESULTS_COLUMNS
        assert "GridPosition" in REQUIRED_RESULTS_COLUMNS
