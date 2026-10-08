"""
Evaluation metrics (Section 8.2, 8.3 of the project guide).

Finishing position is an ordered, per-race outcome — a driver's predicted
position only means something relative to the other drivers in THE SAME
race. Every ranking-flavored metric here (rank correlation, top-k
agreement) is therefore computed per race_id and then averaged, never
computed by pooling predicted/actual positions across different races
into one flat correlation (which would be comparing apples to oranges:
"predicted P3" in a 20-car field and "predicted P3" in a hypothetical
10-car field are not the same thing).

MAE is the one exception — mean absolute error between predicted and
actual finishing position is well-defined pooled across races, since it
doesn't rely on relative ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss as sk_log_loss


def mae(df: pd.DataFrame, true_col: str = "target_finish", pred_col: str = "pred_finish") -> float:
    """Mean absolute error between predicted and actual finishing position,
    pooled across all races."""
    return float((df[true_col] - df[pred_col]).abs().mean())


def race_level_rank_correlation(
    df: pd.DataFrame,
    race_id_col: str = "race_id",
    true_col: str = "target_finish",
    pred_col: str = "pred_finish",
) -> float:
    """Spearman rank correlation between predicted and actual finishing
    order, computed within each race and averaged across races.

    A race where either column has zero variance (e.g. a 1-car "field" in
    a synthetic test, or every prediction tied) has undefined correlation;
    those races are excluded from the average rather than contributing a
    fabricated 0 or 1, and this function returns NaN if EVERY race was
    excluded that way (nothing to average).
    """
    per_race_scores = []
    for _, race_df in df.groupby(race_id_col):
        if race_df[true_col].nunique() < 2 or race_df[pred_col].nunique() < 2:
            continue
        corr, _ = spearmanr(race_df[true_col], race_df[pred_col])
        if not np.isnan(corr):
            per_race_scores.append(corr)
    return float(np.mean(per_race_scores)) if per_race_scores else float("nan")


def topk_agreement(
    df: pd.DataFrame,
    k: int = 1,
    race_id_col: str = "race_id",
    true_col: str = "target_finish",
    pred_col: str = "pred_finish",
) -> float:
    """Fraction of races where the actual winner (true_col == 1) is among
    the model's top-k predicted drivers (lowest k values of pred_col).
    k=1 is "did we pick the actual winner as our predicted winner"."""
    hits = 0
    n_races = 0
    for _, race_df in df.groupby(race_id_col):
        actual_winner_rows = race_df[race_df[true_col] == 1]
        if actual_winner_rows.empty:
            continue  # race with no recorded winner in this data - skip, don't fabricate
        n_races += 1
        predicted_top_k = race_df.nsmallest(k, pred_col)
        if actual_winner_rows.index.isin(predicted_top_k.index).any():
            hits += 1
    return float(hits / n_races) if n_races > 0 else float("nan")


def brier_score_win(
    df: pd.DataFrame, prob_col: str, true_col: str = "target_finish"
) -> float:
    """Brier score for P1 win probability: mean squared error between
    predicted win probability and the binary "actually won" outcome."""
    is_winner = (df[true_col] == 1).astype(float)
    return float(((df[prob_col] - is_winner) ** 2).mean())


def log_loss_win(df: pd.DataFrame, prob_col: str, true_col: str = "target_finish") -> float:
    """Log loss for P1 win probability. Probabilities are clipped away
    from exactly 0/1 before scoring — an unclipped 0 or 1 would produce
    infinite loss on a single miss, which is a numerical-stability
    convention, not a claim about the model's actual confidence."""
    is_winner = (df[true_col] == 1).astype(int)
    probs = df[prob_col].clip(1e-6, 1 - 1e-6)
    return float(sk_log_loss(is_winner, probs, labels=[0, 1]))


def reliability_table(
    df: pd.DataFrame, prob_col: str, true_col: str = "target_finish", n_bins: int = 5
) -> pd.DataFrame:
    """Calibration diagnostic (Section 8.3): bucket predictions into
    `n_bins` probability ranges and compare mean predicted probability to
    actual win rate in each bucket. A well-calibrated model has these two
    columns close together in every bucket with enough rows to be
    meaningful. This is a diagnostic to inspect, not a pass/fail check —
    Section 8.3 requires validating calibration on held-out data before
    labeling any score a "probability"; this table is how you do that."""
    is_winner = (df[true_col] == 1).astype(float)
    bins = pd.cut(df[prob_col], bins=n_bins, include_lowest=True)
    table = pd.DataFrame({"predicted_prob": df[prob_col], "is_winner": is_winner, "bucket": bins})
    summary = table.groupby("bucket", observed=True).agg(
        n=("is_winner", "size"),
        mean_predicted_prob=("predicted_prob", "mean"),
        actual_win_rate=("is_winner", "mean"),
    )
    return summary.reset_index()
