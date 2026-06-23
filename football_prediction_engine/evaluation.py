from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import OUTCOMES


@dataclass
class BankrollResult:
    bets: int
    staked: float
    profit: float
    roi: float
    final_bankroll: float
    max_drawdown: float


def one_hot(y: pd.Series | np.ndarray) -> np.ndarray:
    labels = list(y)
    return np.asarray([[1.0 if label == outcome else 0.0 for outcome in OUTCOMES] for label in labels], dtype=float)


def brier_score(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean(np.sum((probabilities - one_hot(y_true)) ** 2, axis=1)))


def log_loss(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, eps: float = 1e-12) -> float:
    encoded = one_hot(y_true)
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(np.sum(encoded * np.log(clipped), axis=1)))


def accuracy(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def market_probabilities(features: pd.DataFrame) -> np.ndarray:
    probs = features[["odds_home_implied", "odds_draw_implied", "odds_away_implied"]].to_numpy(dtype=float)
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / np.where(row_sums == 0, 1.0, row_sums)


def market_predictions(features: pd.DataFrame) -> np.ndarray:
    probs = market_probabilities(features)
    return np.asarray([OUTCOMES[int(np.argmax(row))] for row in probs])


def reliability_bins(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, bins: int = 5) -> list[dict[str, float | int]]:
    predictions = np.argmax(probabilities, axis=1)
    confidences = probabilities[np.arange(len(probabilities)), predictions]
    hits = np.asarray([OUTCOMES[pred] for pred in predictions]) == np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidences >= low) & (confidences < high if high < 1.0 else confidences <= high)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": round(float(low), 2),
                "bin_high": round(float(high), 2),
                "samples": int(mask.sum()),
                "mean_confidence": round(float(confidences[mask].mean()), 4),
                "hit_rate": round(float(hits[mask].mean()), 4),
            }
        )
    return rows


def tipster_market_correlation(features: pd.DataFrame) -> dict[str, float | str]:
    pairs = [
        ("home", "tipster_home_consensus", "odds_home_implied"),
        ("draw", "tipster_draw_consensus", "odds_draw_implied"),
        ("away", "tipster_away_consensus", "odds_away_implied"),
    ]
    values: dict[str, float | str] = {}
    corrs = []
    for name, tip_col, odds_col in pairs:
        if tip_col not in features or odds_col not in features:
            continue
        corr = float(features[tip_col].corr(features[odds_col]))
        values[f"{name}_correlation"] = round(corr, 4)
        if not np.isnan(corr):
            corrs.append(corr)
    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    values["mean_correlation"] = round(mean_corr, 4)
    values["independence_warning"] = "high_double_counting_risk" if mean_corr >= 0.8 else "moderate" if mean_corr >= 0.55 else "low"
    return values


def closing_line_edges(probabilities: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    return probabilities - market_probabilities(features)


def simulate_bankroll(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    features: pd.DataFrame,
    edge_threshold: float = 0.04,
    starting_bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_stake_fraction: float = 0.03,
) -> BankrollResult:
    bankroll = starting_bankroll
    peak = starting_bankroll
    max_drawdown = 0.0
    staked = 0.0
    profit = 0.0
    bets = 0
    markets = market_probabilities(features)
    y_values = np.asarray(y_true)

    for idx, model_probs in enumerate(probabilities):
        edge = model_probs - markets[idx]
        pick = int(np.argmax(edge))
        if edge[pick] < edge_threshold:
            continue
        decimal_odds = 1.0 / max(markets[idx, pick], 1e-6)
        net_odds = decimal_odds - 1.0
        kelly = max(0.0, (model_probs[pick] * decimal_odds - 1.0) / max(net_odds, 1e-6))
        stake = bankroll * min(max_stake_fraction, kelly * kelly_fraction)
        if stake <= 0:
            continue
        bets += 1
        staked += stake
        if y_values[idx] == OUTCOMES[pick]:
            gain = stake * net_odds
            bankroll += gain
            profit += gain
        else:
            bankroll -= stake
            profit -= stake
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak else 0.0)

    return BankrollResult(
        bets=bets,
        staked=round(float(staked), 2),
        profit=round(float(profit), 2),
        roi=round(float(profit / staked), 4) if staked else 0.0,
        final_bankroll=round(float(bankroll), 2),
        max_drawdown=round(float(max_drawdown), 4),
    )


def evaluation_report(y_true: pd.Series, probabilities: np.ndarray, features: pd.DataFrame) -> dict[str, Any]:
    y_pred = np.asarray([OUTCOMES[int(np.argmax(row))] for row in probabilities])
    market_probs = market_probabilities(features)
    market_pred = market_predictions(features)
    edges = closing_line_edges(probabilities, features)
    max_edges = edges.max(axis=1)
    bankroll = simulate_bankroll(y_true, probabilities, features)
    return {
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "market_accuracy": round(accuracy(y_true, market_pred), 4),
        "brier_score": round(brier_score(y_true, probabilities), 4),
        "market_brier_score": round(brier_score(y_true, market_probs), 4),
        "log_loss": round(log_loss(y_true, probabilities), 4),
        "market_log_loss": round(log_loss(y_true, market_probs), 4),
        "average_positive_edge": round(float(np.mean(np.maximum(max_edges, 0.0))), 4),
        "value_candidates": int(np.sum(max_edges >= 0.04)),
        "bankroll": bankroll.__dict__,
        "reliability": reliability_bins(y_true, probabilities),
        "tipster_market_correlation": tipster_market_correlation(features),
    }
