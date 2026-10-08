"""
Dashboard (Section 11 of the project guide). Run with:

    streamlit run src/app/dashboard.py

Built only after the data pipeline and backtest are trustworthy (Sections
11 and 18 are both explicit about that ordering) — Phase 6, last.

Covers every element Section 11 lists: upcoming GP + prediction cutoff,
predicted finishing order, win probabilities, recent driver/team form,
top model features, historical backtest performance, race-by-race
prediction-vs-actual, a data freshness/status indicator, and a clear
disclaimer that predictions are uncertain estimates.

This file is deliberately thin — every real computation lives in
src.app.dashboard_data (pure functions, unit-tested directly) or earlier
phases' modules. If you're reading this file to understand the actual
logic, you're in the wrong place; start at dashboard_data.py instead.
"""
from __future__ import annotations

import streamlit as st

from src.app import dashboard_data as dd
from src.config import settings
from src.models.predict import PredictionCutoffError, predict_upcoming_race

st.set_page_config(page_title="F1 Race Predictor", layout="wide")

RACES_PATH = settings.data_processed_dir / "races.csv"
MODEL_PATH = settings.models_dir / "baseline_gbm.joblib"
BACKTEST_SUMMARY_PATH = settings.reports_dir / "backtest_summary.json"
BACKTEST_PREDICTIONS_PATH = settings.reports_dir / "backtest_predictions.csv"

st.title("🏁 F1 Race Predictor")
st.warning(
    "**Predictions are uncertain estimates, not guaranteed outcomes.** "
    "This is a prediction-and-evaluation project — see the backtest performance "
    "below before trusting anything on this page.",
    icon="⚠️",
)

# ---- Data freshness / status (Section 11) ----
with st.expander("Data & model status", expanded=False):
    for status in dd.check_artifact_status(RACES_PATH, MODEL_PATH, BACKTEST_SUMMARY_PATH):
        icon = "✅" if status.exists else "❌"
        line = f"{icon} **{status.name}** — `{status.path}`"
        if status.exists:
            line += f" (updated {status.last_modified}; {status.detail})"
        else:
            line += " — not yet built"
        st.markdown(line)

races_df = dd.load_races_df(RACES_PATH)
bundle = dd.load_bundle(MODEL_PATH)

if races_df is None or bundle is None:
    st.error(
        "No processed data / trained model found yet. Run "
        "`python scripts/build_dataset_and_train.py --seasons ...` first "
        "(requires live FastF1 network access — see README)."
    )
    st.stop()

tab_predict, tab_backtest, tab_races = st.tabs(
    ["Upcoming / selected race", "Backtest performance", "Race-by-race comparison"]
)

# ---- Predicted finishing order + win probabilities (Section 11) ----
with tab_predict:
    st.subheader("Generate a prediction")
    st.caption(
        "Pick a race already present in the historical table below to see what the "
        "model would have predicted using only data available before that race — "
        "this is the same code path a real upcoming-race prediction uses."
    )

    race_options = (
        races_df[["season", "round", "race_id", "circuit"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
    )
    labels = [f"{r.season} round {r.round} — {r.circuit}" for r in race_options.itertuples()]
    choice = st.selectbox("Race", options=range(len(labels)), format_func=lambda i: labels[i])
    chosen = race_options.iloc[choice]

    entrants_df = races_df[races_df["race_id"] == chosen["race_id"]]
    entrants = [
        {
            "driver": r["driver"], "team": r["team"],
            "qualifying_position": r["qualifying_position"], "grid_position": r["grid_position"],
        }
        for _, r in entrants_df.iterrows()
    ]

    try:
        predictions = predict_upcoming_race(
            races_df, bundle,
            season=int(chosen["season"]), round_number=int(chosen["round"]), circuit=chosen["circuit"],
            entrants=entrants, qualifying_completed=True,
        )
        st.caption(
            f"Prediction cutoff: after qualifying, before the race, for "
            f"{chosen['season']} round {chosen['round']} ({chosen['circuit']}). "
            "No future race-result data was used."
        )
        st.subheader("Predicted finishing order")
        st.dataframe(dd.format_prediction_table(predictions), use_container_width=True, hide_index=True)

        if bundle.feature_importance is not None:
            st.subheader("Top factors")
            st.dataframe(
                dd.format_feature_importance_table(bundle.feature_importance),
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No feature-importance ranking was saved with this model.")
    except PredictionCutoffError as exc:
        st.error(f"Cannot generate a prediction: {exc}")

# ---- Historical backtest performance (Section 11) ----
with tab_backtest:
    st.subheader("Backtest performance (overall and by season)")
    summary = dd.load_backtest_summary(BACKTEST_SUMMARY_PATH)
    if summary is None:
        st.info(
            "No backtest report found. Run `python scripts/run_backtest.py` "
            "after building the dataset."
        )
    else:
        st.dataframe(dd.format_backtest_summary_table(summary), use_container_width=True, hide_index=True)
        overall = summary["overall"]
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE (finishing position)", f"{overall['mae']:.2f}")
        col2.metric("Rank correlation", f"{overall['rank_correlation']:.2f}")
        col3.metric("Top-1 agreement", f"{overall['top1_agreement'] * 100:.0f}%")

# ---- Race-by-race prediction vs actual (Section 11) ----
with tab_races:
    st.subheader("Race-by-race prediction vs actual")
    backtest_predictions = dd.load_backtest_predictions(BACKTEST_PREDICTIONS_PATH)
    if backtest_predictions is None:
        st.info("No backtest predictions found. Run `python scripts/run_backtest.py` first.")
    else:
        backtested_race_ids = sorted(backtest_predictions["race_id"].unique())
        race_id = st.selectbox("Backtested race", options=backtested_race_ids)
        st.dataframe(
            dd.race_prediction_vs_actual(backtest_predictions, race_id),
            use_container_width=True, hide_index=True,
        )
