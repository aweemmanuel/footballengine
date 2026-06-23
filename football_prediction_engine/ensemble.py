from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .features import OUTCOMES


@dataclass
class EnsembleResult:
    probabilities: dict[str, float]
    most_likely: str
    confidence: str
    consensus_percentage: float
    engine_breakdown: dict[str, int]
    predicted_score: str
    over_under_2_5: str
    top_3_exact_scores: list[dict[str, Any]]
    uncertainty: dict[str, Any]
    market_edge: dict[str, float]


class EnsembleVoter:
    def __init__(self, engines: list[Any], weights: list[float] | None = None):
        self.engines = engines
        self.weights = np.asarray(weights if weights is not None else [getattr(e.spec, "validation_weight", 1.0) for e in engines], dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleVoter":
        for engine in self.engines:
            engine.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.asarray([engine.predict(X) for engine in self.engines]).T
        final = []
        for sample in predictions:
            final.append(Counter(sample).most_common(1)[0][0])
        return np.asarray(final)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        weighted = []
        for weight, engine in zip(self.weights, self.engines):
            weighted.append(engine.predict_proba(X) * weight)
        return np.sum(weighted, axis=0) / self.weights.sum()

    def explain_one(self, X: pd.DataFrame, row_index: int = 0) -> EnsembleResult:
        sample = X.iloc[[row_index]]
        engine_predictions = [str(engine.predict(sample)[0]) for engine in self.engines]
        engine_probabilities = np.asarray([engine.predict_proba(sample)[0] for engine in self.engines])
        votes = Counter(engine_predictions)
        proba = self.predict_proba(sample)[0]
        probabilities = {label: round(float(proba[i]), 4) for i, label in enumerate(OUTCOMES)}
        most_likely = max(votes, key=votes.get)
        consensus = votes[most_likely] / len(self.engines) * 100
        confidence = "high" if consensus >= 70 else "medium" if consensus >= 50 else "low"
        expected_home = max(0.2, 1.15 + probabilities["home_win"] * 1.8 - probabilities["away_win"] * 0.8)
        expected_away = max(0.2, 0.95 + probabilities["away_win"] * 1.8 - probabilities["home_win"] * 0.7)
        predicted_score = f"{int(round(expected_home))}-{int(round(expected_away))}"
        total_goals = expected_home + expected_away
        exact_scores = self._exact_scores(expected_home, expected_away)
        probability_std = engine_probabilities.std(axis=0)
        intervals = {
            label: [
                round(float(max(0.0, proba[i] - 1.64 * probability_std[i])), 4),
                round(float(min(1.0, proba[i] + 1.64 * probability_std[i])), 4),
            ]
            for i, label in enumerate(OUTCOMES)
        }
        market = np.asarray(
            [
                float(sample.iloc[0].get("odds_home_implied", 0.40)),
                float(sample.iloc[0].get("odds_draw_implied", 0.28)),
                float(sample.iloc[0].get("odds_away_implied", 0.32)),
            ]
        )
        market = market / market.sum()
        return EnsembleResult(
            probabilities=probabilities,
            most_likely=most_likely,
            confidence=confidence,
            consensus_percentage=round(consensus, 1),
            predicted_score=predicted_score,
            over_under_2_5="over" if total_goals > 2.5 else "under",
            engine_breakdown={f"{label}_votes": int(votes.get(label, 0)) for label in OUTCOMES},
            top_3_exact_scores=exact_scores[:3],
            uncertainty={
                "probability_intervals": intervals,
                "mean_engine_std": round(float(probability_std.mean()), 4),
                "directional_disagreement": len(votes) > 1,
                "stability": "high" if probability_std.mean() < 0.06 and consensus >= 70 else "medium" if consensus >= 50 else "low",
            },
            market_edge={label: round(float(proba[i] - market[i]), 4) for i, label in enumerate(OUTCOMES)},
        )

    def format_match_output(self, match_features: pd.DataFrame, row_index: int = 0) -> dict[str, Any]:
        row = match_features.iloc[row_index]
        result = self.explain_one(match_features, row_index)
        return {
            "match_id": str(row.get("match_id", "future_match")),
            "home_team": str(row.get("home_team", "")),
            "away_team": str(row.get("away_team", "")),
            "predictions": result.probabilities,
            "most_likely": result.most_likely,
            "confidence": result.confidence,
            "consensus_percentage": result.consensus_percentage,
            "predicted_score": result.predicted_score,
            "over_under_2.5": result.over_under_2_5,
            "engine_breakdown": result.engine_breakdown,
            "uncertainty": result.uncertainty,
            "market_edge": result.market_edge,
            "top_3_exact_scores": result.top_3_exact_scores,
        }

    @staticmethod
    def _exact_scores(home_lambda: float, away_lambda: float) -> list[dict[str, Any]]:
        scores = []
        for home in range(5):
            for away in range(5):
                prob = _poisson_pmf(home, home_lambda) * _poisson_pmf(away, away_lambda)
                scores.append({"score": f"{home}-{away}", "probability": round(float(prob), 4)})
        return sorted(scores, key=lambda item: item["probability"], reverse=True)


def _poisson_pmf(k: int, lam: float) -> float:
    return float((lam**k) * np.exp(-lam) / math.factorial(k))
