"""
Manual verification checkpoint (Section 17, Step 4 of the project guide):
load ONE known historical race + qualifying session for real and print the
observed schema and row counts.

This is deliberately not a pytest test: it needs live network access to
FastF1's backends, which some sandboxed environments (including the one
this project was scaffolded in) block. Run it yourself with:

    python scripts/verify_live_fetch.py

Expected healthy output: non-empty results/laps frames, ok=True, no
warnings, real driver abbreviations and finishing positions printed.
"""
from __future__ import annotations

from src.data.fastf1_client import load_race_and_qualifying
from src.data.validation import validate_session_result


def main() -> None:
    season, event = 2023, "Monza"
    print(f"Loading {season} {event} race + qualifying...")
    race, quali = load_race_and_qualifying(season, event)

    for label, result in [("RACE", race), ("QUALIFYING", quali)]:
        print(f"\n--- {label} ---")
        print(result.summary())
        report = validate_session_result(result)
        print(f"validation ok={report.ok} issues={report.issues}")
        if result.ok and result.results is not None and len(result.results) > 0:
            print(result.results[["Abbreviation", "TeamName", "Position", "GridPosition"]].head(10))


if __name__ == "__main__":
    main()
