"""
Leakage tests (Section 6 + Section 14's "Does a feature test fail if
target-race data is intentionally injected?" checklist item).

These tests do not just check that features "look reasonable" — each one
deliberately corrupts a specific piece of future/target-race information
with an unmistakable sentinel value and asserts that value cannot be
found anywhere it shouldn't be. If any of these fail, it means a change
to cutoff.py, dataset.py, driver_form.py, team_form.py or circuit_form.py
introduced a leak.
"""
from __future__ import annotations

import numpy as np

from src.features.cutoff import races_before
from src.features.dataset import build_training_dataset, feature_columns
from tests.fixtures import make_synthetic_races_df

SENTINEL = 999999.0


class TestCutoffCannotLeakFutureRounds:
    def test_off_by_one_regression_guard(self):
        """If races_before's `<` were ever changed to `<=`, this test
        catches it: round 2 must never appear when cutting off at round 2."""
        df = make_synthetic_races_df()
        history = races_before(df, 2022, 2)
        assert not ((history["season"] == 2022) & (history["round"] >= 2)).any()

    def test_later_season_never_appears_in_earlier_cutoff(self):
        df = make_synthetic_races_df()
        history = races_before(df, 2022, 3)
        assert not (history["season"] > 2022).any()


class TestInjectedFutureDataDoesNotLeak:
    def test_sentinel_in_target_races_own_result_only_appears_as_label(self):
        """Corrupt VER's result in the LAST chronological race with an
        unmistakable sentinel, covering both the finish position and the
        points earned. Rebuild the full dataset. The sentinel must appear
        exactly once: as that single row's target_finish. It must not
        appear in any feature column, for that row or any other row
        (which would indicate the rolling-window logic somehow pulled in
        a race's own result, or a later race leaked into an earlier one).
        """
        df = make_synthetic_races_df()
        mask = (df["race_id"] == "2023_1") & (df["driver"] == "VER")
        assert mask.sum() == 1
        df.loc[mask, "finish_position"] = SENTINEL
        df.loc[mask, "points"] = SENTINEL

        dataset = build_training_dataset(df)
        feats = feature_columns(dataset)

        feature_values = dataset[feats].to_numpy(dtype=float)
        assert not np.any(feature_values == SENTINEL), (
            "Sentinel value leaked into a feature column — a future/target "
            "race result was used as a model input somewhere."
        )

        label_row = dataset[(dataset["race_id"] == "2023_1") & (dataset["driver"] == "VER")]
        assert label_row["target_finish"].iloc[0] == SENTINEL

    def test_corrupting_a_later_race_does_not_change_an_earlier_races_features(self):
        """Build the dataset once on clean data, then again with a later
        race corrupted, and assert every feature value for EARLIER races
        is byte-for-byte identical across both builds. This is the most
        direct test of "does the future affect the past" possible: it
        doesn't rely on knowing which columns might leak, it just diffs
        everything.
        """
        clean_df = make_synthetic_races_df()
        clean_dataset = build_training_dataset(clean_df)

        corrupted_df = make_synthetic_races_df()
        mask = corrupted_df["race_id"] == "2023_1"
        corrupted_df.loc[mask, "finish_position"] = SENTINEL
        corrupted_df.loc[mask, "points"] = SENTINEL
        corrupted_df.loc[mask, "qualifying_position"] = SENTINEL
        corrupted_dataset = build_training_dataset(corrupted_df)

        earlier_clean = clean_dataset[clean_dataset["race_id"] != "2023_1"].reset_index(drop=True)
        earlier_corrupted = corrupted_dataset[
            corrupted_dataset["race_id"] != "2023_1"
        ].reset_index(drop=True)

        feats = feature_columns(clean_dataset)
        pd_equal = earlier_clean[feats].equals(earlier_corrupted[feats])
        assert pd_equal, "Corrupting a later race changed features computed for an earlier race."

    def test_first_race_in_dataset_has_no_leakage_possible_by_construction(self):
        """The very first race chronologically has empty history by
        definition (see TestCutoff.test_first_race_of_dataset_has_empty_history
        in test_features.py) — so every rolling feature for it must be the
        cold-start NaN, never a real number that could only have come from
        peeking at this race or a later one."""
        df = make_synthetic_races_df()
        dataset = build_training_dataset(df)
        first_race_rows = dataset[dataset["race_id"] == "2022_1"]
        for col in ("driver_form_3", "team_form_3", "circuit_driver_avg_finish"):
            assert first_race_rows[col].isna().all(), f"{col} should be NaN for the first race"
