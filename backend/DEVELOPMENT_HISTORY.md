# F1 Race Predictor

Pre-race finishing-position and win-probability prediction for Formula 1,
built on [FastF1](https://docs.fastf1.dev/), with time-aware,
leakage-safe evaluation. See the full project guide for scope and
architecture; this README tracks implementation status.

## Status: Phase 6 (dashboard) complete — all 6 phases complete

What exists right now:

- `src/config.py` — paths, `.env` loading, shared logger.
- `src/data/fastf1_client.py` — cached, logged, defensive loader for one
  FastF1 session (`load_session`) or a race+qualifying pair
  (`load_race_and_qualifying`). Never raises on ordinary data-availability
  problems; reports `ok=False` with warnings instead.
- `src/data/validation.py` — structural checks on the returned frames
  (required columns present, no unexpected nulls).
- `tests/test_data.py` — 10 tests, all passing, covering successful loads,
  empty/partial data, backend exceptions, and validation rules. These mock
  `fastf1.get_session` so they run without network access.
- `scripts/verify_live_fetch.py` — a manual, non-pytest script that hits
  the **real** FastF1 API for one known race (2023 Monza) and prints the
  observed schema/row counts, per Section 17 Step 4 of the guide.

## Important: network access

This project was scaffolded in a sandboxed environment whose outbound
network allowlist does not include FastF1's backends (live timing /
Ergast). Running `scripts/verify_live_fetch.py` here returns `ok=False`
for every session — not because the code is wrong, but because the HTTP
requests never reach their destination.

**You need to run this yourself in an environment with normal internet
access** (your own machine, Claude Code locally, a CI runner, etc.) to
confirm live data actually loads. Do this before trusting any downstream
phase:

```bash
pip install -e .
python scripts/verify_live_fetch.py
```

Expected healthy output: `ok=True` for both RACE and QUALIFYING, non-zero
`n_laps`/`n_results`, and a printed table of driver abbreviations,
positions and teams. If you still see `ok=False` outside a sandbox, it's
a genuine data issue worth investigating (bad event name, FastF1 version
mismatch, etc.) — not a network problem.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Running tests

```bash
pytest tests/ -v
```

## Using the CLI (Phase 5)

```bash
python scripts/build_dataset_and_train.py --seasons 2022 2023  # needs live network
python -m src.app.cli --season 2023 --round 6
python -m src.app.cli --next-race
```

## Using the dashboard (Phase 6)

```bash
pip install -e ".[dashboard]"
python scripts/build_dataset_and_train.py --seasons 2022 2023  # needs live network
python scripts/run_backtest.py
streamlit run src/app/dashboard.py
```

The dashboard runs fine (with a clear "not built yet" message instead of
crashing) even before you've run the two scripts above — that's tested,
see Phase 6 below.

## Verification checkpoint (Phase 1)

| Check | Status |
|---|---|
| FastF1 installed and importable | ✅ |
| One historical race + qualifying session can be requested through our wrapper | ✅ (code path verified; live data pull needs to be confirmed in a network-unrestricted environment — see above) |
| Loader is robust to a session with no usable data (doesn't crash) | ✅ — caught a real bug here: `session.laps`/`session.results` raise `DataNotLoadedError` (not `AttributeError`) when FastF1 "succeeds" but returns nothing; fixed in `_safe_get_loaded_frame` |
| Caching enabled once per process | ✅ |
| Exact season/event/session/retrieval timestamp logged | ✅ |
| Validation checks for required columns and nulls | ✅ |
| Tests for the above, runnable without live network | ✅ 10/10 passing |

## Phase 2 — leakage-safe dataset (complete)

- `src/data/loaders.py` — normalizes a FastF1 race + qualifying `results`
  pair into the unified `races_df` table (one row per driver per race;
  see the module docstring for the exact schema).
- `src/features/cutoff.py` — the single chokepoint every feature builder
  must go through to get "what happened before this race". Ordering key
  is `(season, round)`, never raw calendar date.
- `src/features/driver_form.py`, `team_form.py`, `circuit_form.py` —
  rolling driver/team form (Sections 5.2–5.3) and circuit history
  (Section 5.4), each computed only from data already passed through
  `races_before`.
- `src/features/dataset.py` — `build_training_dataset()` assembles one
  row per driver per race (season-form features per Section 5.5 included
  directly), with `target_finish` clearly separated as the label via
  `feature_columns()`.
- `tests/test_features.py` — 12 tests checking feature values against
  hand-computed expected numbers on a small synthetic dataset.
- `tests/test_leakage.py` — 5 tests that deliberately inject sentinel
  values into target/future races and assert they never appear in any
  feature column. **Verified these actually catch bugs**: I temporarily
  changed `races_before`'s `<` to `<=` (a classic off-by-one leak) and
  confirmed 3 of the 5 leakage tests failed immediately, then reverted
  and confirmed all 28 tests pass again — see commit history / this
  session's transcript for the exact diff used.

All of Phase 2 runs on synthetic fixture data (`tests/fixtures.py`) and
needs no network access, so every test above passes in this sandbox.
What Phase 2 has *not* been validated against yet is real FastF1 data —
that only becomes possible once you've confirmed Phase 1's live fetch
(see above) and wired `src/data/loaders.py` up to real `SessionLoadResult`
objects across multiple real races.

## Phase 3 — baseline model (complete)

- `src/evaluation/metrics.py` — MAE (pooled), rank correlation and top-k
  agreement (both computed per-race then averaged — ranking only means
  something within one race's field, see the module docstring), plus
  Brier score / log loss / a reliability table for win-probability
  calibration (Section 8.2, 8.3).
- `src/models/baseline.py` — `build_gradient_boosting_model()` (sklearn's
  `HistGradientBoostingRegressor`, chosen specifically because it handles
  NaN features natively — no separate imputation step, which would
  otherwise risk its own train/test leakage), plus two simple comparison
  baselines: predict-from-qualifying-position and predict-from-recent-form.
  Also `win_probabilities_from_predictions()` — a per-race softmax over
  predicted finish position, explicitly documented as an **uncalibrated
  heuristic** until validated on real held-out data.
- `src/models/train.py` — `chronological_split()` (splits by *race*, never
  by row, so a field is never split across train/test), `train_baseline_gbm()`,
  and `compare_against_baselines()`, which trains the GBM and evaluates it
  against both simple baselines on the same held-out test split.
- `tests/test_metrics.py` (16 tests) and `tests/test_models.py` (15 tests)
  — including a check that races never span multiple splits, that splits
  stay chronologically ordered, and that win probabilities sum to 1 per
  race.

**Honest result on the toy synthetic dataset** (`tests/fixtures.py`'s
`make_large_synthetic_races_df`, 24 races): the plain qualifying-position
baseline currently **beats** the trained GBM (MAE 0.20 vs 0.37, top-1
agreement 100% vs 80%). This is expected, not a bug — 56 training rows
against 30 features is a classic overfitting setup, and the toy data was
generated as `finish = qualifying + noise`, so qualifying position is
almost the ground truth by construction. This is exactly the comparison
Section 8.2 asks for, and it's a real signal that Phase 4's backtest on
real, much larger multi-season data is what will actually tell us whether
the GBM adds value over the naive baseline — don't take the GBM's
usefulness on faith until that's been run.

## Phase 4 — walk-forward backtesting (complete)

- `src/evaluation/backtest.py` — `walk_forward_backtest()` advances
  race-by-race through the dataset after an initial `min_train_races`
  history, retraining every `retrain_every` races (default: every race)
  and collecting out-of-sample predictions for every subsequent race.
  Reuses Phase 2's per-row cutoff-safe features directly — "train on
  races before race i" is just a race_id filter, since leakage-safety was
  already guaranteed when the dataset was built.
- `backtest_summary()` — MAE, rank correlation, top-k agreement, and
  win-probability calibration, reported **overall AND per season**
  (Section 9: a single pooled number can hide a season where the model
  quietly fell apart, e.g. after a regulation change).
- `worst_predictions()` / `worst_races()` — Section 9's explicit
  requirement to investigate failure cases rather than stop at aggregate
  metrics. These surface the individual driver-races and whole races with
  the largest error, as concrete starting points for "why did the model
  miss this one" rather than a single opaque MAE number.
- `tests/test_backtest.py` — 14 tests, including two distinct leakage
  checks specific to the backtest loop (not just reusing Phase 2's):
  one confirms corrupting a late race never changes an earlier
  out-of-sample prediction, the other confirms a race is never present in
  its own training set. **Both were verified to actually catch bugs**: I
  deliberately changed the training-set slice from `iloc[:i]` to
  `iloc[:i+1]` (an off-by-one that trains on the target race itself) and
  confirmed exactly the self-inclusion test failed while the
  corrupt-a-late-race test correctly did NOT fail — proving they check
  genuinely different things rather than one subsuming the other. Both
  pass again after reverting.

**Backtest result on the toy dataset** (24 races, `min_train_races=10`,
so 14 out-of-sample races / 56 predictions): overall MAE 0.35, rank
correlation 0.90, top-1 agreement 93%. Consistent with Phase 3's finding,
this is on synthetic toy data where qualifying position is close to
ground truth by construction — encouraging that the walk-forward
machinery works correctly, but not evidence the model is good at
predicting real F1 races. That verdict still requires real historical
data (see the network-access note above).

## Phase 5 — upcoming-race prediction / CLI (complete)

- `src/features/dataset.py` — refactored to expose `build_feature_row()`,
  the single row-building function now shared by BOTH
  `build_training_dataset` (historical, with a label) and live prediction
  (below, no label yet). This was a deliberate fix during this phase: an
  earlier version had two separate code paths for "build a training row"
  vs "build a prediction row", which is exactly the kind of drift that
  could silently make live predictions inconsistent with what the model
  was trained on. There's only one path now.
- `src/data/fastf1_client.py` — `get_upcoming_race()` identifies the next
  event from FastF1's schedule and infers whether qualifying has already
  happened, raising `RuntimeError` (never a silent `None`) if the
  schedule can't be fetched.
- `src/models/predict.py` — `ModelBundle` + `save_model_bundle` /
  `load_model_bundle` (joblib) persist the model, its exact feature
  schema, and its permutation-importance ranking together, per Section
  8.1's "save the exact model configuration and feature schema".
  `predict_upcoming_race()` builds features via the shared
  `build_feature_row()` and **refuses** (raises `PredictionCutoffError`)
  rather than fabricates when qualifying hasn't happened yet or any
  entrant is missing a qualifying position — Section 10's "report that
  instead of fabricating a prediction", taken literally.
- `src/models/train.py` — added `compute_feature_importance()`
  (permutation importance; `HistGradientBoostingRegressor` has no
  built-in `feature_importances_`) so the CLI's "Top factors" list is
  genuinely derived from the data, not hard-coded (Section 2's explicit
  requirement).
- `src/app/cli.py` — `python -m src.app.cli --next-race` or
  `--season Y --round N`, producing the exact report shape from Section
  2's product vision (predicted finish, P1 probabilities, top factors,
  "No future race-result data was used."). No dashboard — Section 11 and
  Section 18 both say build that only after the pipeline is trustworthy.
- `scripts/build_dataset_and_train.py` — the orchestration script that
  actually needs live network: fetches N seasons via FastF1, builds
  `data/processed/races.csv`, trains the model, saves
  `models/baseline_gbm.joblib`. Run this yourself before the CLI can do
  anything real (see network-access note above).
- 24 new tests across `tests/test_predict.py`, `tests/test_upcoming_race.py`,
  `tests/test_cli.py`. The standout: **a consistency test proving live
  predictions and training rows are built identically** — it takes a real
  historical race, predicts it through the "upcoming race" code path, and
  asserts every single feature value matches what `build_training_dataset`
  computed for that same race, feature-by-feature. That's the test that
  would catch the exact kind of drift the `build_feature_row` refactor
  above was written to prevent.

I ran the CLI end-to-end against synthetic data + a mocked FastF1 session
and confirmed the output matches Section 2's product vision shape,
including deriving "Top factors" from real permutation importance rather
than a hard-coded list.

## Phase 6 — dashboard (complete)

- `src/app/dashboard_data.py` — every real computation the dashboard
  needs (artifact status, table formatting, backtest summarization),
  written as plain functions with **no `streamlit` import**, specifically
  so they're directly unit-testable without spinning up a Streamlit
  runtime. 12 tests in `tests/test_dashboard_data.py`.
- `src/app/dashboard.py` — thin Streamlit UI on top of
  `dashboard_data.py`. Covers every element Section 11 lists:
  - upcoming/selected race + prediction cutoff
  - predicted finishing order + win probabilities
  - recent driver/team form (shown alongside the prediction table)
  - top model features (from the same permutation importance the CLI uses)
  - historical backtest performance, overall and per season
  - race-by-race prediction-vs-actual for any backtested race
  - a data/model freshness status indicator that never crashes, even when
    nothing has been built yet
  - a prominent disclaimer: predictions are uncertain estimates, not
    guaranteed outcomes
- `src/app/labels.py` — the feature-name-to-human-label mapping, pulled
  out of `cli.py` (Phase 5) into its own module so the CLI's "Top
  factors" and the dashboard's "Top factors" can never drift apart.
- `scripts/run_backtest.py` — generates the backtest reports
  (`reports/backtest_summary.json`, `reports/backtest_predictions.csv`)
  the dashboard reads, so the dashboard never re-runs a multi-minute
  backtest on every page load.
- `tests/test_dashboard_app.py` — 5 end-to-end smoke tests using
  Streamlit's own `AppTest` (installed in this sandbox and used for real,
  not just described): one set confirms the dashboard runs without
  exception and shows a clear "not built yet" message on a completely
  empty project, another confirms it runs without exception against a
  fully populated project (real synthetic data, a real trained model, a
  real backtest report) and that the status indicator, prediction table,
  and disclaimer all actually render.

This closes the loop from Section 18's ordering requirement — dashboard
built last, only after Phases 1–5 (data, features, leakage tests, model,
backtest, CLI) were already in place and tested.

## What's real vs what's still a toy

Worth restating plainly, now that all 6 phases exist: **every number in
this project — the Phase 3 baseline comparison, the Phase 4 backtest, the
Phase 6 dashboard's metrics — has only ever been computed on synthetic
toy data** (`tests/fixtures.py`), because this sandbox's network can't
reach FastF1's backends (confirmed directly in Phase 1). The code paths
are real and thoroughly tested (105 tests, including leakage tests
verified to catch real injected bugs), but nothing here is evidence the
model is any good at predicting real Formula 1 races. That verdict
requires you to run `scripts/build_dataset_and_train.py` and
`scripts/run_backtest.py` yourself against real FastF1 data in an
unrestricted environment, then look at the dashboard's backtest tab with
real results in it — not before.

## Repository structure

```
f1-race-predictor/
├── README.md
├── pyproject.toml
├── .env.example
├── data/{raw,processed,cache}/
├── models/
├── reports/
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── fastf1_client.py
│   │   ├── loaders.py
│   │   └── validation.py
│   ├── features/
│   │   ├── cutoff.py
│   │   ├── driver_form.py
│   │   ├── team_form.py
│   │   ├── circuit_form.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── backtest.py
│   └── app/
│       ├── cli.py
│       ├── dashboard.py
│       ├── dashboard_data.py
│       └── labels.py
├── scripts/
│   ├── verify_live_fetch.py
│   ├── build_dataset_and_train.py
│   └── run_backtest.py
└── tests/
    ├── fixtures.py
    ├── test_data.py
    ├── test_features.py
    ├── test_leakage.py
    ├── test_metrics.py
    ├── test_models.py
    ├── test_backtest.py
    ├── test_predict.py
    ├── test_upcoming_race.py
    ├── test_cli.py
    ├── test_dashboard_data.py
    └── test_dashboard_app.py
```
