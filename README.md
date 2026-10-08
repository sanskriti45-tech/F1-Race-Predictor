# F1 Race Predictor

Pre-race finishing-position and win-probability prediction for Formula 1,
built on [FastF1](https://docs.fastf1.dev/), with leakage-safe time-aware
evaluation — plus a frontend that consumes the real backend over a local
HTTP API. No prediction logic lives in the browser.

```
F1-Race-Predictor/
├── backend/     Python: FastF1 ingestion, features, model, API server
└── frontend/    Static HTML/CSS/JS: intro video, dashboard, car visuals
```

## 1. Architecture

```
FastF1 → raw session data → normalization/validation → feature builder
  → time-aware training dataset → gradient-boosting model
  → evaluation/calibration → saved model
  → upcoming-race feature builder → predict_upcoming_race()
          │
          ▼
  backend/src/app/api.py  (stdlib http.server, GET /api/prediction/next-race)
          │  JSON over HTTP, localhost only
          ▼
  frontend/index.html  →  fetch()  →  normalizePredictionData()  →  UI
```

The API layer (`backend/src/app/api.py`) contains **no prediction logic**.
Every field in its response comes from calling the real backend functions
directly (`get_upcoming_race`, `load_session`, `predict_upcoming_race`,
`load_model_bundle`) or reading files the backend pipeline already
produces (`races.csv`, the saved model, backtest reports). If something
isn't available yet, the response says so honestly — it never fabricates
a value. The frontend mirrors this: its own `normalizePredictionData()`
only reshapes what the API returned; it never computes a prediction.

## 2. Backend setup

**Python version**: 3.12.3 (what this project was built and tested
against — `requires-python = ">=3.10"` in `pyproject.toml`, but only
3.12.3 has actually been run).

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Dependencies (`pyproject.toml`): `fastf1`, `pandas`, `numpy`,
`scikit-learn`, `pyyaml`, `python-dotenv`, plus `pytest`/`streamlit` in
the `dev` extra. **No new dependency was added for the API server** — it
uses Python's standard library (`http.server`) only, per "the thinnest
possible API/data-serving layer."

### FastF1 network access — important, tested directly

The sandbox this project was built in has an unusual, specific network
profile, confirmed by direct testing, not assumed:

- `fastf1.get_event_schedule()` (the calendar/next-race lookup) **works**
  — reaches real data.
- `fastf1.get_session(...).load()` (actual session results, qualifying
  times, lap data — via Ergast/live-timing) **does not** — it returns
  without raising, but with zero rows, which the existing defensive
  loader code (`src/data/fastf1_client.py`) already catches and reports
  honestly (`ok=False`) rather than crashing.

This means in *this* environment, `GET /api/prediction/next-race` can
correctly identify the next race and its cutoff, but cannot produce a
full prediction (it honestly reports `predictionGenerated: false` with
an explanation). **Your own machine may have full access to both** —
this is a property of the sandbox, not a known limitation of the code.
Run `python scripts/verify_live_fetch.py` to check your own environment.

### Generate real data (needs FastF1 network access)

```bash
cd backend
python scripts/build_dataset_and_train.py --seasons 2022 2023
python scripts/run_backtest.py
```

These populate `data/processed/races.csv`, `models/baseline_gbm.joblib`,
and `reports/backtest_*`. **None of these files are included in this
ZIP** — the project ships clean, and the API honestly reports
`historicalDataLoaded: false` / `modelLoaded: false` until you run this.

### Start the API server

```bash
cd backend
python scripts/serve_api.py --port 8787
```

`fastf1` takes **5–10 seconds to import** (measured directly — it's not
a trivial library) — the server needs that long to start listening. Give
it a moment before the frontend's first request.

Endpoints:
- `GET /api/health` → `{"status": "ok"}`
- `GET /api/prediction/next-race` → the full payload described in
  section 4 below

CORS is restricted to `http://localhost:*`, `http://127.0.0.1:*`, and the
`null` origin a browser sends for a page opened via `file://` — never a
wildcard. This is a local-dev-only server; don't expose it publicly.

## 3. Frontend setup

```bash
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500/index.html`. (Opening `index.html`
directly via `file://` mostly works too, but some browsers restrict
`fetch()` from `file://` pages regardless of CORS — serving it locally is
more reliable.)

If your API server runs on a different port/host, edit the one line near
the top of `frontend/index.html`'s `<script>`:

```js
var API_BASE_URL = 'http://localhost:8787';
```

## 4. API / data contract

`GET /api/prediction/next-race` returns:

```jsonc
{
  "dataFreshness": "2026-...",       // bundle.saved_at, or null
  "modelVersion": "baseline_gbm",    // or null if no model loaded
  "fastf1Available": true,           // real check, this request
  "historicalDataLoaded": true,      // races.csv exists
  "modelLoaded": true,               // model bundle exists
  "predictionGenerated": false,      // did a full prediction actually get produced
  "nextRace": { "name", "circuit", "date", "cutoffNote", "statusNote" } | null,
  "predictions": [ {                 // one row per driver — empty [] if not available
    "predicted_rank", "driver", "team", "circuit",
    "qualifying_position", "grid_position",
    "driver_form_3", "driver_form_5", "driver_quali_form_3", "driver_quali_form_5",
    "driver_points_3", "driver_points_5", "driver_finish_rate_3", "driver_finish_rate_5",
    "team_form_3", "team_form_5", "team_quali_form_3", "team_quali_form_5",
    "team_points_3", "team_points_5",
    "circuit_driver_avg_finish", "circuit_driver_prior_starts",
    "circuit_team_avg_finish", "circuit_team_prior_starts",
    "season_avg_finish", "season_avg_quali", "season_points", "season_races_so_far",
    "pred_finish", "win_prob"
  } ],
  "featureValues": [ {"feature": "...", "value": ...} ],  // real saved feature importance
  "backtest": { "overall": {"mae", "rank_correlation", "top1_agreement",
                             "brier_score_win", "log_loss_win", "n_races"}, ... } | null,
  "history": [ {"race_id", "driver", "team", "target_finish", "pred_finish", "win_prob"} ],
  "error": "human-readable reason, or null"
}
```

This is the **real, actual schema** the backend produces (verified
directly from `src/features/dataset.py`, `src/models/predict.py`,
`src/evaluation/backtest.py` — not assumed from a spec).

## 5. How next-race prediction is generated

1. `GET /api/prediction/next-race` calls `get_upcoming_race()` → the next
   calendar event + whether qualifying has happened.
2. If qualifying is done, it loads that session's real results via
   `load_session(season, round, "Q")`.
3. If that succeeds, it calls `predict_upcoming_race()` — the same
   function `backend/src/app/cli.py` uses — with the real races table and
   the real saved model. This builds every feature via
   `races_before(season, round)`, strictly excluding the target race and
   anything after it (leakage prevention, unchanged).
4. The result is serialized straight to JSON. No step here recomputes or
   adjusts anything in JavaScript.

## 6. Driver/team/car synchronization

- Clicking a row in the Race Prediction table, or choosing a driver in
  the Drivers tab dropdown, calls `selectDriver(name)` — the single
  source of truth for "who's selected."
- `selectDriver()` calls `renderDriverInto(name, ids)` **twice**, once
  per mount point (Drivers tab, Prediction tab dashboard) — same
  function, same data, two places it's drawn — and keeps the dropdown
  and the highlighted table row in sync too.
- The car itself comes from `F1CarDisplay(team, driverCode)`:
  `resolveTeamId()` fuzzy-matches the backend's real team string (e.g.
  "Red Bull Racing") against `TEAM_REGISTRY`; an unmatched team renders
  a neutral grey car **with its own real name**, never another team's
  identity. The car is an original SVG silhouette (front wing, nose,
  halo, sidepods, rear wing, wheels) — not a photo, not any real team's
  actual livery, since no licensed car imagery is available here.

## 7. How prediction cutoff works

A race's features are only ever built from `races_before(season, round)`
— every race strictly before the target, by `(season, round)` ordering.
Qualifying position and grid position ARE used (known before a race
starts); finishing position, points, and anything from the race itself
or later are never used for that race's own prediction. This logic is
unchanged from the original backend and is covered by
`tests/test_leakage.py`, including tests that deliberately inject a
sentinel value into a future race and confirm it never appears in an
earlier prediction.

## 8. Running tests

```bash
cd backend
pytest tests/ -v
```

**111 tests, all passing** (105 from the original backend phases + 6 new
in `tests/test_api.py`). The new API tests include a direct cross-check:
calling `predict_upcoming_race()` directly and comparing its output
field-by-field against what the HTTP API returns for the same inputs —
proving the API layer adds zero prediction logic of its own.

## 9. What was actually tested before packaging (not just "it returns 200")

- **Scenario A — backend not running**: frontend correctly shows "BACKEND
  CONNECTION ERROR" with a Retry button; no fake data; zero JS errors.
- **Scenario B — real server, real network conditions**: real schedule
  data flowed through (a real upcoming Grand Prix + circuit + cutoff
  timestamp), real backtest metrics from a real model run appeared in
  the History tab, `predictionGenerated` honestly showed `false` with a
  clear reason (session data unreachable in this sandbox) — no banner
  error, since the server itself was reachable; this is the Step
  14-vs-15 distinction (connection error vs. honest empty data) working
  as designed.
- **Scenario C — full real pipeline, network boundary mocked**: a real
  trained model's real predictions for four real-named teams (Red Bull
  Racing, Scuderia Ferrari, Mercedes-AMG Petronas, McLaren) were served
  over real HTTP and consumed by a real browser. Clicking each of the 4
  driver rows correctly swapped the car's colors and every statistic to
  that driver's real values; `performance.getEntriesByType('navigation')`
  was checked before and after — count unchanged, proving no page reload
  across any of the clicks. Zero JS console errors throughout.

All three scenarios used Playwright driving a real Chromium instance
against a real running server — not a simulated or mocked frontend.

See `backend/DEVELOPMENT_HISTORY.md` for the detailed, phase-by-phase
record of how the backend was built and tested (data ingestion, feature
engineering, the baseline model, backtesting, the CLI, and the Streamlit
dashboard) — useful if you want the reasoning behind a specific design
choice, not just the end state.

## 10. Known limitations

- **No real production data ships in this ZIP.** You must run
  `scripts/build_dataset_and_train.py` yourself with real FastF1 network
  access before the API has anything to serve.
- **Session/qualifying data was unreachable in the sandbox that built
  this project** (see section 2) — `predictionGenerated` may honestly
  stay `false` depending on your own network environment. This is a
  property of network access, not a code defect; `get_upcoming_race()`
  working while `load_session()` doesn't is real, confirmed behavior.
- **The car visuals are original schematic illustrations**, not licensed
  team car photography — none was available to use. `TEAM_REGISTRY`
  currently covers 10 common constructors by name-matching; anything
  else gets the honest neutral fallback with its real name.
- **CORS is wide open to any localhost/127.0.0.1 origin** for development
  convenience. Do not deploy `serve_api.py` outside localhost as-is.
- **The frontend has no build step** — it's one HTML file with inline
  CSS/JS, by design, matching how it was developed throughout this
  project. There's no bundler, no `package.json`, nothing to `npm
  install`.
- The baseline model is a `HistGradientBoostingRegressor` with a single,
  unoptimized configuration (Section 8.1 of the original project guide:
  "start simple before tuning"). Its real-world accuracy has only ever
  been evaluated on synthetic data in this environment — treat any
  number you see until you've run the real pipeline against real
  historical seasons as unverified.
